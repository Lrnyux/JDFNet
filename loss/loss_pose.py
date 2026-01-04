import cv2
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import lpips
import matplotlib.pyplot as plt
from skimage.transform import resize
from tqdm import tqdm
import time

class BackprojectDepth(nn.Module):
    """Layer to transform a depth image into a point cloud
    """
    def __init__(self, batch_size, height, width):
        super(BackprojectDepth, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width

        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        self.id_coords = nn.Parameter(torch.from_numpy(self.id_coords),
                                      requires_grad=False)

        self.ones = nn.Parameter(torch.ones(self.batch_size, 1, self.height * self.width),
                                 requires_grad=False)

        self.pix_coords = torch.unsqueeze(torch.stack(
            [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0)
        self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1)
        self.pix_coords = nn.Parameter(torch.cat([self.pix_coords, self.ones], 1),
                                       requires_grad=False)

    def forward(self, depth, inv_K):
        # print(depth.device,inv_K.device)
        cam_points = torch.matmul(inv_K[:, :3, :3], self.pix_coords)
        cam_points = depth.view(self.batch_size, 1, -1) * cam_points
        cam_points = torch.cat([cam_points, self.ones], 1)

        return cam_points

class Project3D(nn.Module):
    """Layer which projects 3D points into a camera with intrinsics K and at position T
    """
    def __init__(self, batch_size, height, width, eps=1e-7):
        super(Project3D, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.eps = eps

    def forward(self, points, K, T):
        P = torch.matmul(K, T)[:, :3, :]

        cam_points = torch.matmul(P, points)

        pix_coords = cam_points[:, :2, :] / (cam_points[:, 2, :].unsqueeze(1) + self.eps)
        pix_coords = pix_coords.view(self.batch_size, 2, self.height, self.width)
        pix_coords = pix_coords.permute(0, 2, 3, 1)
        pix_coords[..., 0] /= self.width - 1
        pix_coords[..., 1] /= self.height - 1
        pix_coords = (pix_coords - 0.5) * 2
        return pix_coords


class FusionLoss_dVRK(nn.Module):
    def __init__(self, args, batch_size, height, width, K=None, fl_bl=None):
        super(FusionLoss_dVRK, self).__init__()

        T_x = args.T_x
        f = args.f
        c_x = args.c_x
        c_y = args.c_y

        self.args = args
        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.K = np.array([[f,0,c_x,0],
                           [0,f,c_y,0],
                           [0,0,1,0],
                           [0,0,0,1]],dtype=np.float32)
        self.K = torch.from_numpy(self.K)
        self.fl_bl = T_x * f
        self.K_inv = torch.linalg.inv(self.K)

        self.K = self.K.view(1,4,4).expand(batch_size,4,4)
        self.K_inv = self.K_inv.view(1,4,4).expand(batch_size,4,4)

        self.backproject_depth = BackprojectDepth(batch_size=self.batch_size, height=self.height, width=self.width)
        self.project_3d = Project3D(batch_size=self.batch_size, height=self.height, width=self.width)

        self.K = nn.Parameter(self.K,requires_grad=False)
        self.K_inv = nn.Parameter(self.K_inv,requires_grad=False)

        # sample grid for computing loss
        self.x_base = torch.linspace(-1, 1, width).repeat(batch_size, height, 1)
        self.y_base = torch.linspace(-1, 1, height).repeat(batch_size, width, 1).transpose(1, 2)
        self.x_base = nn.Parameter(self.x_base, requires_grad=False)
        self.y_base = nn.Parameter(self.y_base, requires_grad=False)


    def forward(self,img_sta,img_end,disp_pred,flow_pred,pose_pred, save_results=False):
        """
            :param disp:        B x H x W   range[-1,1]
            :param flow:        B x 4 x H x W  range[-1,1]
            :param K:           B x 4 x 4
            :param K_inv:       B x 4 x 4
            :fl_bl:             1
            :pose_pred:         B x 4 x 4
        """
        disp_pred = disp_pred.view(self.batch_size, self.height, self.width)
        flow_pred = flow_pred[:,:2,:,:].view(self.batch_size, 2, self.height, self.width)
        pose_pred = pose_pred.view(self.batch_size, 4, 4)

        # depth_pred = self.fl_bl / (self.width * disp_pred)
        depth_pred = self.fl_bl / ((self.width/2) * torch.abs(disp_pred) + 1e-5)

        depth_pred = torch.clamp(depth_pred,min=1e-4,max=1e4)
        # valid_mask = torch.zeros_like(depth_pred)
        # valid_mask[depth_pred!=0] = 1
        valid_mask =  depth_pred!=0


        cam_points = self.backproject_depth(depth_pred, self.K_inv)
        pix_points = self.project_3d(cam_points,self.K, pose_pred)

        valid_mask &= (pix_points[:, :, :, 0] < 1) & (pix_points[:, :, :, 0] > -1) & \
                      (pix_points[:, :, :, 1] < 1) & (pix_points[:, :, :, 1] > -1)
        # valid_mask_img = torch.unsqueeze(valid_mask,1).expand(-1,3,-1,-1)
        valid_mask_pts = torch.unsqueeze(valid_mask,3).expand(-1,-1,-1,2)
        # print(valid_mask.shape, valid_mask_img.shape, valid_mask_pts.shape)

        x_shifts = flow_pred[:, 0, :, :].view(self.batch_size, self.height, self.width)
        y_shifts = flow_pred[:, 1, :, :].view(self.batch_size, self.height, self.width)
        flow_points = torch.stack((self.x_base + x_shifts, self.y_base + y_shifts), dim=3)

        fusion_recon = F.grid_sample(img_sta,pix_points,mode='bilinear')
        flow_recon = F.grid_sample(img_sta, flow_points, mode='bilinear')

        valid_mask_img = torch.unsqueeze((torch.sum(fusion_recon,dim=1)!=0),1).expand(-1,3,-1,-1)

        # image_loss = F.l1_loss(fusion_recon,img_end)
        # points_loss = F.l1_loss(pix_points,flow_points)

        image_loss = F.l1_loss(fusion_recon*valid_mask_img, img_end*valid_mask_img)
        points_loss = F.l1_loss(pix_points*valid_mask_pts,flow_points*valid_mask_pts)
        ref_loss = F.l1_loss(img_sta*valid_mask_img, img_end*valid_mask_img)

        loss = {}
        loss['loss_fuse_image'] = image_loss
        loss['loss_fuse_points'] = points_loss
        loss['loss_ref'] = ref_loss

        if save_results:
            loss['fusion_recon'] = fusion_recon
            loss['flow_recon'] = flow_recon
            loss['valid_img'] = img_end

            # loss['fusion_recon'] = fusion_recon*valid_mask_img
            # loss['flow_recon'] = flow_recon*valid_mask_img
            # loss['valid_img'] = img_end*valid_mask_img

        return loss




def get_valid_depth(gt, crop=False):
    valid = (gt > 0) & (gt < 80)
    if crop:
        h, w = gt.shape[:2]
        crop_mask = gt != gt
        y1, y2 = int(0.40810811 * h), int(0.99189189 * h)
        x1, x2 = int(0.03594771 * w), int(0.96405229 * w)
        crop_mask[y1:y2, x1:x2] = 1
        valid = valid & crop_mask
    return valid


def depth_flow2pose(disp, flow, K, K_inv, fl_bl=1., gs=16, th=1., method='AP3P', depth2=None):
    """

    :param disp:        H x W   range[0,1]
    :param flow:        h x w x2  range[-1,1]
    :param K:           3 x 3
    :param K_inv:       3 x 3
    :fl_bl:             1 x 1
    :param gs:          grad size for sampling
    :param th:          threshold for RANSAC
    :param method:      PnP method
    :return:
    """
    if method == 'PnP':
        PnP_method = cv2.SOLVEPNP_ITERATIVE
    elif method == 'AP3P':
        PnP_method = cv2.SOLVEPNP_AP3P
    elif method == 'EPnP':
        PnP_method = cv2.SOLVEPNP_EPNP
    else:
        raise ValueError('PnP method ' + method)



    H, W = disp.shape[:2]
    depth = (fl_bl / ( W * disp )).clamp(min=1e-5)

    valid_mask = get_valid_depth(depth)
    sample_mask = np.zeros_like(valid_mask)
    sample_mask[::gs, ::gs] = 1
    valid_mask &= sample_mask == 1

    h, w = flow.shape[:2]
    # flow[:, :, 0] = flow[:, :, 0] / w * W
    # flow[:, :, 1] = flow[:, :, 1] / h * H
    flow = cv2.resize(flow, (W, H), interpolation=cv2.INTER_LINEAR)
    flow[:, :, 0] = flow[:, :, 0] * W
    flow[:, :, 1] = flow[:, :, 1] * H

    grid = np.stack(np.meshgrid(range(W), range(H)), 2).astype(
        np.float32)  # HxWx2
    one = np.expand_dims(np.ones_like(grid[:, :, 0]), 2)
    homogeneous_2d = np.concatenate([grid, one], 2)
    d = np.expand_dims(depth, 2)
    points_3d = d * (K_inv @ homogeneous_2d.reshape(-1, 3).T).T.reshape(H, W, 3)

    points_2d = grid + flow
    valid_mask &= (points_2d[:, :, 0] < W) & (points_2d[:, :, 0] >= 0) & \
                  (points_2d[:, :, 1] < H) & (points_2d[:, :, 1] >= 0)

    ret, rvec, tvec, inliers = cv2.solvePnPRansac(points_3d[valid_mask],
                                                  points_2d[valid_mask],
                                                  K, np.zeros([4, 1]),
                                                  reprojectionError=th,
                                                  flags=PnP_method)
    if not ret:
        inlier_ratio = 0.
    else:
        inlier_ratio = len(inliers) / np.sum(valid_mask)
    pose_mat = np.eye(4, dtype=np.float32)
    pose_mat[:3, :] = cv2.hconcat([cv2.Rodrigues(rvec)[0], tvec])

    return pose_mat, np.concatenate([rvec, tvec]), inlier_ratio

def depth_flow2pose_pt(depth, flow, K, K_inv, fl_bl=1., gs=16, th=1., method='AP3P'):
    """
    This operation is non-differentiable and is run with the original image size only.
    :param depth:       B x H x W
    :param flow:        B x 2 x h x w
    :param K:           B x 3 x 3
    :param K_inv:       B x 3 x 3
    :param gs:          grad size for sampling
    :param th:          threshold for RANSAC
    :param method:      PnP method
    :return:
    """

    B = depth.size(0)
    dtype = depth.type()
    depth = [arr.squeeze(0) for arr in
             np.split(depth.detach().cpu().numpy(), B, axis=0)]
    flow = [arr.squeeze(0) for arr in
            np.split(flow.detach().cpu().numpy().transpose([0, 2, 3, 1]), B, axis=0)]
    K = [arr.squeeze(0) for arr in
         np.split(K.detach().cpu().numpy(), B, axis=0)]
    K_inv = [arr.squeeze(0) for arr in
             np.split(K_inv.detach().cpu().numpy(), B, axis=0)]

    pose_mat = []
    pose_vec = []
    inlier_ratio = []
    for i, (a, b, c, d) in enumerate(zip(depth, flow, K, K_inv)):
        mat, vec, r = depth_flow2pose(a, b, c, d, gs=gs, th=th, method=method)
        pose_mat.append(mat)
        pose_vec.append(vec)
        inlier_ratio.append(r)

    pose_mat = torch.tensor(np.stack(pose_mat)).type(dtype)
    pose_vec = torch.tensor(np.stack(pose_vec)).type(dtype)
    inlier_ratio = torch.tensor(np.stack(inlier_ratio)).type(dtype)
    return pose_mat, pose_vec, inlier_ratio


