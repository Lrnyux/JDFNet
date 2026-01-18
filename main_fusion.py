import argparse
import torch
import torch.nn
from torch.utils.data import Dataset, DataLoader
from utils.utils import AverageMeter, ConsoleLogger
from models.model_flow import FlowNet
from models.model_depth import DepthNet
from models.model_pose import PoseNet
from models.model_specular import SpecNet
from dataset.dataset_dVRK import dVRK_dataset
from loss.loss_stereo import *
from loss.loss_pose import *



def get_args():
    parser = argparse.ArgumentParser()

    # =========for hyper parameters===
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--trial', type=str, default='base')
    parser.add_argument('--mode', type=str, default='train', choices=['train','com'])
    parser.add_argument('--seed', type=int, default=1)

    # ==========define the task==============
    parser.add_argument('--task', type=str, default='End_to_End', choices=[ 'End_to_End'])
    parser.add_argument('--percent', type=float, default=1.0, choices=[1.0, 0.5, 0.3, 0.2])
    # =========for SCARED dataset============
    parser.add_argument('--step', type=int, default=3, help='time step for frames')

    parser.add_argument('--train_img_root', type=str,
                        default='')
    parser.add_argument('--val_img_root', type=str,
                        default='')
    parser.add_argument('--test_img_root', type=str,
                        default='')

    parser.add_argument('--co_transform', type=list, default=[False, False, False, False, False],
                        help='Trans, Rotate, Vflip, Hflip, Swap')

    # =========for training===========
    parser.add_argument('--train_batchsize', type=int, default=12)
    parser.add_argument('--val_batchsize', type=int, default=1)
    parser.add_argument('--max_epoch', default=10, help='max training epoch', type=int)
    parser.add_argument('--lr', default=1e-4, help='learning rate', type=float)
    parser.add_argument('--lr_depth', default=1e-5, help='learning rate for depth estimation model', type=float)
    parser.add_argument('--lr_flow', default=1e-5, help='learning rate for flow estimation model', type=float)
    parser.add_argument('--lr_pose', default=1e-5, help='learning rate for pose estimation model', type=float)

    parser.add_argument('--weight_decay', default=0.0001, help='decay of learning rate', type=float)
    parser.add_argument('--freq_print_train', default=20, help='Printing frequency for training', type=int)
    parser.add_argument('--freq_print_val', default=20, help='Printing frequency for validation', type=int)
    parser.add_argument('--freq_print_test', default=50, help='Printing frequency for test', type=int)
    parser.add_argument('--freq_perform_val', default=10, help='Frequency of Validation', type=int)

    # ==========loss function============
    parser.add_argument('--recon_right', type=int, default=0)

    parser.add_argument('--w_l1', type=float, default=1.0)
    parser.add_argument('--w_ssim', type=float, default=1.0)
    parser.add_argument('--w_grad', type=float, default=10.0)
    parser.add_argument('--w_fuse_img', type=float, default=1.0)
    parser.add_argument('--w_fuse_pts', type=float, default=0.2)
    parser.add_argument('--w_cycle_img', type=float, default=1.0)
    parser.add_argument('--w_cycle_pts', type=float, default=0.5)
    parser.add_argument('--w_tasks', type=list, default=[1.0, 1.0, 0.5])
    parser.add_argument('--w_cross', type=float, default=[1.0, 0.5, 0.5])


    parser.add_argument('--w_ID', type=float, default=1.0)
    parser.add_argument('--w_OOD', type=float, default=1.0)


    parser.add_argument('--clip_min', type=float, default=0.1)
    parser.add_argument('--clip_max', type=float, default=30.0)

    # ========for model ==============
    parser.add_argument('--max_disp', type=int, default=64)
    parser.add_argument('--image_size', type=list, default=[320, 256])
    parser.add_argument('--save_epoch', type=int, default=1)

    return args



def train_end_to_end():
    args = get_args()
    LOGGER = ConsoleLogger('train_integrate', 'train')
    logdir = LOGGER.getLogFolder()
    LOGGER.info(args)
    torch.manual_seed(args.seed)
    # ==================================dataset============================================================
    # Place the dataset and dataloader here
    train_set = dVRK_dataset(args, stage='Train')
    train_dataloader = DataLoader(train_set, batch_size=args.train_batchsize, shuffle=True, num_workers=16,
                                  drop_last=True)
    val_set = dVRK_dataset(args, stage='Val')
    val_dataloader = DataLoader(val_set, batch_size=args.val_batchsize, shuffle=False, num_workers=16,
                                drop_last=False)

    # ==================================model============================================================
    device = torch.device(f'cuda:{args.gpu}')

    model_depth = DepthNet(device=device, maxdisp=args.max_disp)
    model_flow = FlowNet(device=device, maxrange=args.max_disp)
    model_pose = PoseNet(num_input_frames=2)
    model_spec = SpecNet(in_channel=3, out_channel=3, inner_channel=16, norm_groups=16,
                                   channel_mults=[1, 2, 4, 8], res_blocks=3)

    model_depth = model_depth.cuda(device)
    model_flow = model_flow.cuda(device)
    model_pose = model_pose.cuda(device)
    model_spec = model_spec.cuda(device)

    # ==================================loss function============================================================

    Loss_func_flow = FrameFlowLoss_singdr_with_spec
    Loss_func_depth = StereodepthLoss_singdr_with_spec
    Loss_func_flow_cross = FrameFlowLoss_singdr
    Loss_func_depth_cross = StereodepthLoss_singdr
    Loss_func_cycle_cross = CrossCycleLoss
    Loss_func_fusion = FusionLoss_dVRK(args=args, batch_size=args.train_batchsize, height=args.image_size[1],
                                  width=args.image_size[0]).to(device)
    Loss_func_fusion_eval = FusionLoss_dVRK(args=args, batch_size=args.val_batchsize, height=args.image_size[1],
                                       width=args.image_size[0]).to(device)

    params_dict = [{'params': model_depth.parameters(), 'lr': args.lr_depth, 'weight_decay': args.weight_decay},
                   {'params': model_flow.parameters(), 'lr': args.lr_flow, 'weight_decay': args.weight_decay},
                   {'params': model_pose.parameters(), 'lr': args.lr_pose, 'weight_decay': args.weight_decay}]

    optimizer = torch.optim.AdamW(params_dict)
    step_size = int(args.max_epoch * 0.6 * len(train_dataloader))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size,
                                                gamma=0.1)

    # ================train process=============================================
    for epoch in range(args.max_epoch):
        LOGGER.info(f'---------------Training epoch : {epoch}-----------------')
        batch_time = AverageMeter()
        loss_log_depth = AverageMeter()
        loss_log_flow = AverageMeter()
        loss_log_pose = AverageMeter()
        loss_log_depth_cross = AverageMeter()
        loss_log_flow_cross = AverageMeter()
        loss_log_cycle_cross = AverageMeter()
        loss_log = AverageMeter()

        loss_val_log_depth = AverageMeter()
        loss_val_log_flow = AverageMeter()
        loss_val_log_pose = AverageMeter()
        start = time.time()

        model_depth.train()
        model_flow.train()
        model_pose.train()

        for it, batch in enumerate(train_dataloader, 0):
            imgl_sta = batch['imgl_sta'].cuda(device)
            imgl_end = batch['imgl_end'].cuda(device)
            imgr_sta = batch['imgr_sta'].cuda(device)
            imgr_end = batch['imgr_end'].cuda(device)

            imgl_sta_spec = batch['imgl_sta_spec'].cuda(device)
            imgl_end_spec = batch['imgl_end_spec'].cuda(device)
            imgr_sta_spec = batch['imgr_sta_spec'].cuda(device)
            imgr_end_spec = batch['imgr_end_spec'].cuda(device)

            batch_size = len(imgl_sta)

            disp_sta = model_depth(imgl_sta, imgr_sta)
            disp_end = model_depth(imgl_end, imgr_end)

            flow_l = model_flow(imgl_sta, imgl_end)
            flow_r = model_flow(imgr_sta, imgr_end)
            flow_l_inv = model_flow(imgl_end, imgl_sta)

            cam_pose_l, _, _ = model_pose(imgl_sta, imgl_end)

            imgl_sta_recon_depth = apply_disparity(imgr_sta, -1 * disp_sta[2], max_disp=args.max_disp)
            imgl_sta_recon_flow = apply_flow(imgl_end, flow_l_inv[2], max_disp=args.max_disp)

            disp_sta_cross = model_depth(imgl_sta_recon_flow, imgr_sta)
            flow_l_cross = model_flow(imgl_sta_recon_depth, imgl_end)

            with torch.no_grad():
                mask_pred_left,_,_ = model_spec(imgl_sta)
                mask_pred_end,_,_ = model_spec(imgl_end)

            loss_term_depth = Loss_func_depth(imgl_sta, imgr_sta, imgl_sta_spec, imgr_sta_spec, disp_sta[2], mask_pred_left, max_disp=args.max_disp,
                                              recon_right=0)
            loss_depth = args.w_ID * (args.w_ssim * loss_term_depth['loss_ssim_left'] + args.w_l1 * loss_term_depth[
                    'loss_l1_left'] + args.w_grad * loss_term_depth['loss_grad']) + \
                       args.w_OOD * (args.w_ssim * loss_term_depth['loss_ssim_left_spec'] + args.w_l1 * loss_term_depth[
                    'loss_l1_left_spec'] + args.w_grad * loss_term_depth['loss_grad_spec'])

            # loss for flow estimation
            loss_term_flow = Loss_func_flow(imgl_sta, imgl_end, imgl_sta_spec, imgl_end_spec, flow_l[2], mask_pred_end, max_disp=args.max_disp)
            loss_flow = args.w_ID * (args.w_ssim * loss_term_flow['loss_ssim_end'] + args.w_l1 *  loss_term_flow['loss_l1_end'] +
                          args.w_grad * loss_term_flow['loss_grad']) + \
                    args.w_OOD * (args.w_ssim * loss_term_flow['loss_ssim_end_spec'] + args.w_l1 *  loss_term_flow['loss_l1_end_spec'] +
                          args.w_grad * loss_term_flow['loss_grad_spec'])

            # loss for cross-view fusion
            loss_term_fuse = Loss_func_fusion(imgl_sta, imgl_end, disp_sta[2], flow_l[2], cam_pose_l)
            loss_fuse = args.w_fuse_img * loss_term_fuse['loss_fuse_image'] + args.w_fuse_pts * loss_term_fuse[
                'loss_fuse_points']

            # loss for cross cycle constraints via cross-domain fusion
            loss_term_cycle_cross = Loss_func_cycle_cross(imgl_sta,imgl_end,imgr_sta,imgr_end,disp_sta[2],disp_end[2],flow_l[2],flow_r[2],max_disp=args.max_disp)
            loss_cycle_cross = args.w_cycle_img * loss_term_cycle_cross['loss_cross_img'] + args.w_cycle_pts * loss_term_cycle_cross['loss_cross_pts']

            # loss for depth estimation via cross-domain fusion
            loss_term_depth_cross = Loss_func_depth_cross(imgl_sta_recon_flow, imgr_sta, disp_sta_cross[2], max_disp=args.max_disp, recon_right=0)
            loss_depth_cross = args.w_ssim * loss_term_depth_cross['loss_ssim_left'] + \
                               args.w_l1 * loss_term_depth_cross['loss_l1_left'] + \
                               args.w_grad * loss_term_depth_cross['loss_grad']

            # loss for flow estimation via cross-domain fusion
            loss_term_flow_cross = Loss_func_flow_cross(imgl_sta_recon_depth, imgl_end, flow_l_cross[2], max_disp=args.max_disp)
            loss_flow_cross = args.w_ssim * loss_term_flow_cross['loss_ssim_end'] + \
                              args.w_l1 * loss_term_flow_cross['loss_l1_end'] + \
                              args.w_grad * loss_term_flow_cross['loss_grad']

            loss = args.w_tasks[0] * (loss_depth + args.w_cross[0] * loss_depth_cross) + \
                   args.w_tasks[1] * (loss_flow + args.w_cross[1] * loss_flow_cross) + \
                   args.w_tasks[2] * loss_fuse + \
                   args.w_cross[2] * loss_cycle_cross

            if it % args.freq_print_train == 0:
                LOGGER.info(
                    '[Depth Loss]: Loss SSIM Left:{:.5f}, Loss L1 Left:{:.5f}, Loss Grad:{:.5f}'.format(
                        loss_term_depth['loss_ssim_left'].item(), loss_term_depth['loss_l1_left'].item(),
                        loss_term_depth['loss_grad'].item()) +
                    ' [Flow Loss]: Loss SSIM End:{:.5f}, Loss L1 End:{:.5f}, Loss Grad:{:.5f}'.format(
                        loss_term_flow['loss_ssim_end'].item(), loss_term_flow['loss_l1_end'].item(),
                        loss_term_flow['loss_grad'].item()) +
                    ' [Cross Depth Loss]: Loss SSIM Left:{:.5f}, Loss L1 Left:{:.5f}, Loss Grad:{:.5f}'.format(
                        loss_term_depth_cross['loss_ssim_left'].item(), loss_term_depth_cross['loss_l1_left'].item(),
                        loss_term_depth_cross['loss_grad'].item()) +
                    ' [Cross Flow Loss]: Loss SSIM End:{:.5f}, Loss L1 End:{:.5f}, Loss Grad:{:.5f}'.format(
                        loss_term_flow_cross['loss_ssim_end'].item(), loss_term_flow_cross['loss_l1_end'].item(),
                        loss_term_flow_cross['loss_grad'].item()) +
                    ' [Cross Cycle Loss]: Loss Image:{:.5f}, Loss Points:{:.5f} '.format(
                        loss_term_cycle_cross['loss_cross_img'].item(), loss_term_cycle_cross['loss_cross_pts'].item()) +
                    ' [Fusion Loss]: Loss Image:{:.5f}, Loss Points:{:.5f}, Loss Ref:{:.5f}'.format(
                        loss_term_fuse['loss_fuse_image'].item(), loss_term_fuse['loss_fuse_points'].item(),
                        loss_term_fuse['loss_ref'].item()))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            batch_time.update(time.time() - start)
            loss_log.update(loss.item(), batch_size)
            loss_log_depth.update(loss_term_depth['loss_ssim_left'].item(), batch_size)
            loss_log_flow.update(loss_term_flow['loss_ssim_end'].item(), batch_size)
            loss_log_pose.update(loss_term_fuse['loss_fuse_image'].item(), batch_size)
            loss_log_depth_cross.update(loss_term_depth_cross['loss_ssim_left'].item(), batch_size)
            loss_log_flow_cross.update(loss_term_flow_cross['loss_ssim_end'].item(), batch_size)
            loss_log_cycle_cross.update(loss_term_cycle_cross['loss_cross_img'].item(),batch_size)

            if it % args.freq_print_train == 0:
                message = 'Epoch : [{0}][{1}/{2}]  Learning rate  {learning_rate:.7f}\t' \
                          'Batch Time {batch_time.val:.3f}s ({batch_time.ave:.3f})\t' \
                          'Speed {speed:.1f} samples/s \t' \
                          'Loss_train {loss1.val:.5f} ({loss1.ave:.5f}) \t' \
                          'Loss Stereo {loss2.val:.5f}({loss2.ave:.5f}) \t' \
                          'Loss Flow {loss3.val:.5f}({loss3.ave:.5f}) \t' \
                          'Loss Cross Stereo {loss4.val:.5f}({loss4.ave:.5f}) \t' \
                          'Loss Cross Flow {loss5.val:.5f}({loss5.ave:.5f}) \t'\
                          'Loss Cross Cycle {loss6.val:.5f}({loss6.ave:.5f}) \t'\
                          'Loss Pose {loss7.val:.5f}({loss7.ave:.5f}) \t'.format(
                    epoch, it, len(train_dataloader), learning_rate=optimizer.param_groups[0]['lr'],
                    batch_time=batch_time, speed=batch_size / batch_time.val, loss1=loss_log, loss2=loss_log_depth,
                    loss3=loss_log_flow, loss4=loss_log_depth_cross, loss5=loss_log_flow_cross, loss6=loss_log_cycle_cross, loss7=loss_log_pose)
                LOGGER.info(message)
            start = time.time()

        with torch.no_grad():
            model_depth.eval()
            model_flow.eval()
            model_pose.eval()
            for it, batch in enumerate(val_dataloader, 0):
                imgl_sta = batch['imgl_sta'].cuda(device)
                imgl_end = batch['imgl_end'].cuda(device)
                imgr_sta = batch['imgr_sta'].cuda(device)
                imgr_end = batch['imgr_end'].cuda(device)

                imgl_sta_spec = batch['imgl_sta_spec'].cuda(device)
                imgl_end_spec = batch['imgl_end_spec'].cuda(device)
                imgr_sta_spec = batch['imgr_sta_spec'].cuda(device)
                imgr_end_spec = batch['imgr_end_spec'].cuda(device)

                batch_size = len(imgl_sta)

                disp_sta = model_depth(imgl_sta, imgr_sta)
                flow_l = model_flow(imgl_sta, imgl_end)
                cam_pose_l, _, _ = model_pose(imgl_sta, imgl_end)
                mask_pred_left, _, _ = model_spec(imgl_sta)
                mask_pred_end, _, _ = model_spec(imgl_end)

                loss_term_depth = Loss_func_depth(imgl_sta, imgr_sta, imgl_sta_spec, imgr_sta_spec, disp_sta,
                                                  mask_pred_left, max_disp=args.max_disp,
                                                  recon_right=0)
                loss_val_depth = loss_term_depth['loss_ssim_left']
                loss_val_log_depth.update(loss_val_depth, batch_size)

                loss_term_flow = Loss_func_flow(imgl_sta, imgl_end, imgl_sta_spec, imgl_end_spec, flow_l, mask_pred_end, max_disp=args.max_disp)
                loss_val_flow = loss_term_flow['loss_ssim_end']
                loss_val_log_flow.update(loss_val_flow, batch_size)

                loss_term_pose = Loss_func_fusion_eval(imgl_sta, imgl_end, disp_sta, flow_l, cam_pose_l)
                loss_val_pose = loss_term_pose['loss_fuse_image']
                loss_val_log_pose.update(loss_val_pose, batch_size)

        message = 'Evaluation=== Depth Stereo SSIM {loss1.ave:.5f}, Flow End SSIM {loss2.ave:.5f}, Pose Estimation L1 {loss3.ave:.5f} \t'.format(
            loss1=loss_val_log_depth, loss2=loss_val_log_flow, loss3=loss_val_log_pose)
        LOGGER.info(message)

    checkpoint_dir = os.path.join(logdir, 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    LOGGER.info('=> saving checkpoint to {}'.format(checkpoint_dir))
    states = dict()
    states['model_depth_state_dict'] = model_depth.state_dict()
    states['model_flow_state_dict'] = model_flow.state_dict()
    states['model_pose_state_dict'] = model_pose.state_dict()
    states['optimizer_state_dict'] = optimizer.state_dict()
    torch.save(states, os.path.join(checkpoint_dir, 'last.tar'))
    LOGGER.info('Finish Training')




if __name__ == '__main__':
    args = get_args()

    if args.mode == 'train':
        if args.task == 'End_to_End':
            train_end_to_end()


