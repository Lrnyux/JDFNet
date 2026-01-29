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

def apply_disparity(img,disp,max_disp=96):
    # img [B,C,H,W]
    # disp [B,1,H,W]
    batch_size,channel,height,width = img.shape
    # Original coordinates of pixels
    x_base = torch.linspace(-1, 1, width).repeat(batch_size, height, 1).type_as(img)
    y_base = torch.linspace(-1, 1, height).repeat(batch_size, width, 1).transpose(1,2).type_as(img)
    # Apply shift in X direction
    x_shifts = disp.view(batch_size,height,width)
    flow_field = torch.stack((x_base + x_shifts, y_base), dim=3)


    # output = F.grid_sample(img, flow_field, mode='bilinear', padding_mode='border')
    # output = F.grid_sample(img, flow_field, mode='bilinear', align_corners = True)
    output = F.grid_sample(img, flow_field, mode='bilinear')

    return output


def apply_flow(img,flow,max_disp=96):
    # img [B,C,H,W]
    # disp [B,2,H,W]
    batch_size, channel, height, width = img.shape
    # Original coordinates of pixels
    x_base = torch.linspace(-1, 1, width).repeat(batch_size, height, 1).type_as(img)
    y_base = torch.linspace(-1, 1, height).repeat(batch_size, width, 1).transpose(1, 2).type_as(img)
    # Apply shift in X and Y direction
    x_shifts = flow[:,0,:,:].view(batch_size, height, width)
    y_shifts = flow[:,1,:,:].view(batch_size, height, width)

    flow_field = torch.stack((x_base + x_shifts, y_base + y_shifts), dim=3)


    # output = F.grid_sample(img, flow_field, mode='bilinear', padding_mode='border')
    output = F.grid_sample(img, flow_field, mode='bilinear', align_corners=True)
    # output = F.grid_sample(img, flow_field, mode='bilinear')
    return output

def gradient_x(img):
    # img [B,C,H,W]
    # Pad input to keep output size consistent
    img = F.pad(img, (0, 1, 0, 0), mode="replicate")
    gx = img[:, :, :, :-1] - img[:, :, :, 1:]  # NCHW
    return gx

def gradient_y(img):
    # img [B,C,H,W]
    # Pad input to keep output size consistent
    img = F.pad(img, (0, 0, 0, 1), mode="replicate")
    gy = img[:, :, :-1, :] - img[:, :, 1:, :]  # NCHW
    return gy

def disp_smoothness(img,disp):
    # img [B,C,H,W]
    # disp [B,1,H,W]
    disp_gradients_x = gradient_x(disp)
    disp_gradients_y = gradient_y(disp)

    image_gradients_x = gradient_x(img)
    image_gradients_y = gradient_y(img)

    weights_x = torch.exp(-torch.mean(torch.abs(image_gradients_x), 1,keepdim=True))
    weights_y = torch.exp(-torch.mean(torch.abs(image_gradients_y), 1,keepdim=True))

    smoothness_x = disp_gradients_x * weights_x
    smoothness_y = disp_gradients_y * weights_y

    return torch.abs(smoothness_x) + torch.abs(smoothness_y)

def flow_smoothness(img,flow):
    # img [B,C,H,W]
    # disp [B,2,H,W]
    disp_gradients_x = gradient_x(flow)
    disp_gradients_y = gradient_y(flow)

    image_gradients_x = gradient_x(img)
    image_gradients_y = gradient_y(img)

    weights_x = torch.exp(-torch.mean(torch.abs(image_gradients_x), 1,keepdim=True))
    weights_y = torch.exp(-torch.mean(torch.abs(image_gradients_y), 1,keepdim=True))

    smoothness_x = disp_gradients_x * weights_x
    smoothness_y = disp_gradients_y * weights_y

    return torch.abs(smoothness_x) + torch.abs(smoothness_y)


def SSIM_loss(x, y):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_x = nn.AvgPool2d(3, 1)(x)
    mu_y = nn.AvgPool2d(3, 1)(y)
    mu_x_mu_y = mu_x * mu_y
    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)

    sigma_x = nn.AvgPool2d(3, 1)(x * x) - mu_x_sq
    sigma_y = nn.AvgPool2d(3, 1)(y * y) - mu_y_sq
    sigma_xy = nn.AvgPool2d(3, 1)(x * y) - mu_x_mu_y

    SSIM_n = (2 * mu_x_mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x_sq + mu_y_sq + C1) * (sigma_x + sigma_y + C2)
    SSIM = SSIM_n / SSIM_d

    return torch.clamp((1 - SSIM) / 2, 0, 1)





def StereodepthLoss_singdr_with_spec(left,right,left_spec,right_spec,disp_pred,mask_pred,max_disp=96,recon_right=0):

    # left [B,C,H,W]
    # disp_pred [B,1,H,W]
    # mask_pred [B,1,H,W]

    b,c,h,w = left.shape
    spec_ID = 1-mask_pred
    spec_OOD = mask_pred


    if recon_right ==1:
        disp_left_pred = disp_pred[:,0,:,:].view(b,1,h,w)

        right_recon = apply_disparity(left,disp_left_pred,max_disp)

        disp_left_smoothness = disp_smoothness(left,disp_left_pred)

        loss_l1_right = torch.mean(torch.abs(right_recon * spec_ID-right * spec_ID))

        loss_ssim_right = torch.mean(SSIM_loss(right_recon * spec_ID,right * spec_ID))

        loss_grad = torch.mean(torch.abs(disp_left_smoothness))

        right_recon_spec = apply_disparity(left_spec, disp_left_pred, max_disp)

        disp_left_smoothness_spec = disp_smoothness(left_spec, disp_left_pred)

        loss_l1_right_spec = torch.mean(torch.abs(right_recon_spec * spec_OOD - right_spec * spec_OOD))

        loss_ssim_right_spec = torch.mean(SSIM_loss(right_recon_spec * spec_OOD, right_spec * spec_OOD))

        loss_grad_spec = torch.mean(torch.abs(disp_left_smoothness_spec))

        loss = {}
        loss['loss_l1_right'] = loss_l1_right
        loss['loss_ssim_right'] = loss_ssim_right
        loss['loss_grad'] = loss_grad

        loss['loss_l1_right_spec'] = loss_l1_right_spec
        loss['loss_ssim_right_spec'] = loss_ssim_right_spec
        loss['loss_grad_spec'] = loss_grad_spec

    else:
        disp_right_pred = -1 * disp_pred[:, 0, :, :].view(b, 1, h, w)

        left_recon = apply_disparity(right, disp_right_pred, max_disp)

        disp_right_smoothness = disp_smoothness(right, disp_right_pred)

        loss_l1_left = torch.mean(torch.abs(left_recon * spec_ID - left * spec_ID))

        loss_ssim_left = torch.mean(SSIM_loss(left_recon * spec_ID, left * spec_ID))

        loss_grad = torch.mean(torch.abs(disp_right_smoothness))

        left_recon_spec = apply_disparity(right_spec, disp_right_pred, max_disp)

        disp_right_smoothness_spec = disp_smoothness(right_spec, disp_right_pred)

        loss_l1_left_spec = torch.mean(torch.abs(left_recon_spec * spec_OOD - left_spec * spec_OOD))

        loss_ssim_left_spec = torch.mean(SSIM_loss(left_recon_spec * spec_OOD, left_spec * spec_OOD))

        loss_grad_spec = torch.mean(torch.abs(disp_right_smoothness_spec))

        loss = {}
        loss['loss_l1_left'] = loss_l1_left
        loss['loss_ssim_left'] = loss_ssim_left
        loss['loss_grad'] = loss_grad

        loss['loss_l1_left_spec'] = loss_l1_left_spec
        loss['loss_ssim_left_spec'] = loss_ssim_left_spec
        loss['loss_grad_spec'] = loss_grad_spec

    return loss



def FrameFlowLoss_singdr_with_spec(start,end,start_spec,end_spec,flow_pred,mask_pred,max_disp=96):

    # start [B,C,H,W]
    # flow_pred [B,2,H,W]
    # mask_pred [B,1,H,W]

    b,c,h,w = start.shape
    spec_ID = 1 - mask_pred
    spec_OOD = mask_pred

    forward_flow = flow_pred[:,0:2,:,:].view(b,2,h,w)

    end_recon = apply_flow(start,forward_flow,max_disp=max_disp)

    forward_flow_smoothness = flow_smoothness(start,forward_flow)

    end_recon_spec = apply_flow(start_spec, forward_flow, max_disp=max_disp)

    forward_flow_smoothness_spec = flow_smoothness(start_spec, forward_flow)

    loss_l1_end = torch.mean(torch.abs(end_recon * spec_ID - end * spec_ID))

    loss_ssim_end = torch.mean(SSIM_loss(end_recon * spec_ID,end * spec_ID))

    loss_grad = torch.mean(torch.abs(forward_flow_smoothness))

    loss_l1_end_spec = torch.mean(torch.abs(end_recon_spec * spec_OOD - end_spec * spec_OOD))

    loss_ssim_end_spec = torch.mean(SSIM_loss(end_recon_spec * spec_OOD, end_spec * spec_OOD))

    loss_grad_spec = torch.mean(torch.abs(forward_flow_smoothness_spec))

    loss = {}
    loss['loss_l1_end'] = loss_l1_end
    loss['loss_ssim_end'] = loss_ssim_end
    loss['loss_grad'] = loss_grad
    loss['loss_l1_end_spec'] = loss_l1_end_spec
    loss['loss_ssim_end_spec'] = loss_ssim_end_spec
    loss['loss_grad_spec'] = loss_grad_spec

    return loss


def CrossCycleLoss(l_sta,l_end,r_sta,r_end,d_sta_pred,d_end_pred,f_l_pred,f_r_pred,max_disp=96):
    b,c,h,w = l_sta.shape
    disp_sta_pred = -1 * d_sta_pred[:,0,:,:].view(b,1,h,w)
    disp_end_pred = -1 * d_end_pred[:,0,:,:].view(b,1,h,w)
    flow_l_pred = f_l_pred[:,:2,:,:].view(b,2,h,w)
    flow_r_pred = f_r_pred[:,:2,:,:].view(b,2,h,w)


    l_sta_recon_depth = apply_disparity(r_sta,disp_sta_pred,max_disp=max_disp)
    l_end_recon_l_sta_recon = apply_flow(l_sta_recon_depth,flow_l_pred,max_disp=max_disp)
    r_end_recon_flow = apply_flow(r_sta,flow_r_pred,max_disp=max_disp)
    l_end_recon_r_end_recon = apply_disparity(r_end_recon_flow,disp_end_pred,max_disp=max_disp)

    loss_img = torch.mean(torch.abs(l_end_recon_l_sta_recon - l_end_recon_r_end_recon)) + \
                torch.mean(SSIM_loss(l_end_recon_l_sta_recon, l_end_recon_r_end_recon))
    loss_pts_x_axis = torch.mean(torch.abs(disp_sta_pred[:,0,:,:] + flow_l_pred[:,0,:,:] -\
                                           disp_end_pred[:,0,:,:] + flow_r_pred[:,0,:,:]))
    loss_pts_y_axis = torch.mean(torch.abs(flow_l_pred[:,1,:,:] - flow_r_pred[:,1,:,:]))
    loss_pts = loss_pts_x_axis + loss_pts_y_axis

    loss = {}
    loss['loss_cross_img'] = loss_img
    loss['loss_cross_pts'] = loss_pts

    return loss
