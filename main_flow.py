import time
from tqdm import tqdm
from utils.utils import AverageMeter, ConsoleLogger, calc_rmse, calc_psnr, calc_ssim
from models.model_flow import FlowNet
from dataset.dataset_dVRK import *
from loss.loss_stereo import *




def get_args():
    parser = argparse.ArgumentParser()

    # =========for hyper parameters===
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--trial', type=str, default='base')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--seed', type=int, default=1)

    # ==========define the task==============
    parser.add_argument('--task', type=str, default='Flow', choices=['Flow'])
    parser.add_argument('--percent', type=float, default=1.0, choices=[1.0, 0.5, 0.3, 0.2])

    # =========for SCARED dataset============
    parser.add_argument('--step', type=int, default=3, help='time step for frames')
    parser.add_argument('--train_img_root', type=str,
                        default='')
    parser.add_argument('--val_img_root', type=str,
                        default='')
    parser.add_argument('--test_img_root', type=str,
                        default='')
    parser.add_argument('--co_transform', type=list, default=[True, False, True, False, False],
                        help='Trans, Rotate, Vflip, Hflip, Swap')

    # =========for training===========
    parser.add_argument('--train_batchsize', type=int, default=16)
    parser.add_argument('--val_batchsize', type=int, default=1)
    parser.add_argument('--max_epoch', default=20, help='max training epoch', type=int)
    parser.add_argument('--lr', default=1e-3, help='learning rate', type=float)
    parser.add_argument('--weight_decay', default=0.0001, help='decay of learning rate', type=float)
    parser.add_argument('--freq_print_train', default=20, help='Printing frequency for training', type=int)
    parser.add_argument('--freq_print_val', default=20, help='Printing frequency for validation', type=int)
    parser.add_argument('--freq_print_test', default=50, help='Printing frequency for test', type=int)
    parser.add_argument('--load_model', type=str, default='')

    # ==========loss function============
    parser.add_argument('--loss_type', type=str, default='stereo', choices=['stereo'])
    parser.add_argument('--w_l1', type=float, default=1.0)
    parser.add_argument('--w_ssim', type=float, default=1.0)
    parser.add_argument('--w_grad', type=float, default=5.0)
    parser.add_argument('--w_pyramid', type=list, default=[0.5, 0.7, 1.0])

    parser.add_argument('--clip_min', type=float, default=0.1)
    parser.add_argument('--clip_max', type=float, default=30.0)

    # ========for model ==============
    parser.add_argument('--model_type', type=str, default='base', choices=['base'])
    parser.add_argument('--max_disp', type=int, default=64)


    parser.add_argument('--image_size', type=list, default=[320, 256])
    parser.add_argument('--save_epoch', type=int, default=1)



    # parse configs
    args = parser.parse_args()

    return args


# ===================================================Training Functions=================================================


def train_flow():
    args = get_args()
    LOGGER = ConsoleLogger('train_flow_' + args.trial, 'train')
    logdir = LOGGER.getLogFolder()
    LOGGER.info(args)
    torch.manual_seed(args.seed)

    # ==================================dataset============================================================
    # Place the dataset and dataloader here
    train_set = dVRK_dataset_flow(args,stage='Train')
    train_dataloader = DataLoader(train_set, batch_size=args.train_batchsize, shuffle=True, num_workers=16,
                                  drop_last=True)
    val_set = dVRK_dataset_flow(args,stage='Val')
    val_dataloader = DataLoader(val_set, batch_size=args.val_batchsize, shuffle=False, num_workers=16,
                                drop_last=False)

    # ==================================model============================================================
    device = torch.device(f'cuda:{args.gpu}')
    model = FlowNet(device=device, maxrange=args.max_disp)
    model = model.cuda(device)

    # ==================================loss function============================================================
    Loss_func = FrameFlowLoss_singdr
    optimizer = torch.optim.AdamW(params=model.parameters(),
                                  lr=args.lr,
                                  weight_decay=args.weight_decay)
    step_size = int(args.max_epoch * 0.4 * len(train_dataloader))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size,
                                                gamma=0.1)

    # ================train process=============================================
    for epoch in range(args.max_epoch):
        LOGGER.info(f'---------------Training epoch : {epoch}-----------------')
        batch_time = AverageMeter()
        loss_log = AverageMeter()
        loss_val_log = AverageMeter()
        start = time.time()

        model.train()
        for it, batch in enumerate(train_dataloader, 0):
            imgl = batch['imgl'].cuda(device)
            imgr = batch['imgr'].cuda(device)
            batch_size = len(imgl)

            flow_pred = model(imgl,imgr)


            # loss function generation

            loss_term_s1 = Loss_func(imgl, imgr, flow_pred[0],max_disp=args.max_disp)
            loss_term_s2 = Loss_func(imgl, imgr, flow_pred[1],max_disp=args.max_disp)
            loss_term_s3 = Loss_func(imgl, imgr, flow_pred[2],max_disp=args.max_disp)
            loss_s1 = args.w_ssim * loss_term_s1['loss_ssim_end'] + \
                      args.w_l1 * loss_term_s1['loss_l1_end'] + \
                      args.w_grad * loss_term_s1['loss_grad']
            loss_s2 = args.w_ssim *  loss_term_s2['loss_ssim_end'] + \
                      args.w_l1 *  loss_term_s2['loss_l1_end'] + \
                      args.w_grad * loss_term_s2['loss_grad']
            loss_s3 = args.w_ssim * loss_term_s3['loss_ssim_end'] + \
                      args.w_l1 *  loss_term_s3['loss_l1_end'] + \
                      args.w_grad * loss_term_s3['loss_grad']
            loss = args.w_pyramid[0] * loss_s1 + args.w_pyramid[1] * loss_s2 + args.w_pyramid[2] * loss_s3
            if it % args.freq_print_train == 0:
                LOGGER.info('Loss SSIM End:{:.5f},  Loss L1 End:{:.5f}, Loss Grad:{:.5f}'.format(
                    loss_term_s3['loss_ssim_end'].item(), loss_term_s3['loss_l1_end'].item(),loss_term_s3['loss_grad'].item()))

            # optimize the model
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm(model.parameters(), max_norm=args.clip_max)
            optimizer.step()
            scheduler.step()
            batch_time.update(time.time() - start)
            loss_log.update(loss.item(), batch_size)
            if it % args.freq_print_train == 0:
                message = 'Epoch : [{0}][{1}/{2}]  Learning rate  {learning_rate:.7f}\t' \
                          'Batch Time {batch_time.val:.3f}s ({batch_time.ave:.3f})\t' \
                          'Speed {speed:.1f} samples/s \t' \
                          'Loss_train {loss1.val:.5f} ({loss1.ave:.5f})\t'.format(
                    epoch, it, len(train_dataloader), learning_rate=optimizer.param_groups[0]['lr'],
                    batch_time=batch_time, speed=batch_size / batch_time.val, loss1=loss_log)
                LOGGER.info(message)
            start = time.time()

        # ================validation process=============================================
        with torch.no_grad():
            model.eval()
            for it, batch in enumerate(val_dataloader, 0):
                imgl = batch['imgl'].cuda(device)
                imgr = batch['imgr'].cuda(device)
                batch_size = len(imgl)


                flow_pred = model(imgl, imgr)
                loss_term = Loss_func(imgl, imgr, flow_pred,max_disp=args.max_disp)

                loss_val = loss_term['loss_ssim_end']
                loss_val_log.update(loss_val, batch_size)


        message = 'Evaluation=== Loss L1 Flow {loss.ave:.5f} \t'.format(loss=loss_val_log)
        LOGGER.info(message)


    checkpoint_dir = os.path.join(logdir, 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    LOGGER.info('=> saving checkpoint to {}'.format(checkpoint_dir))
    states = dict()
    states['model_state_dict'] = model.state_dict()
    states['optimizer_state_dict'] = optimizer.state_dict()
    torch.save(states, os.path.join(checkpoint_dir, 'last.tar'))
    LOGGER.info('Finish Training')





# ===================================================Test Functions=====================================================
# ======================================================================================================================

def _test_flow(checkpoint=''):
    args = get_args()
    if checkpoint:
        args.load_model = checkpoint
    LOGGER = ConsoleLogger('test_flow_sequence', 'test')
    logdir = LOGGER.getLogFolder()
    LOGGER.info(args)




    test_set = dVRK_dataset_flow(args,stage='Test')
    test_dataloader = DataLoader(test_set, batch_size=args.val_batchsize, shuffle=False, num_workers=16,
                                 drop_last=False)



    # ==================================model============================================================
    device = torch.device(f'cuda:{args.gpu}')
    model = FlowNet(device=device, maxrange=args.max_disp)
    model = model.cuda(device)

    if args.load_model:
        model_path = args.load_model
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"No checkpoint found at {model_path}")
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        LOGGER.info(f'---------------Finishing loading models----------------')

    rmse_log = AverageMeter()
    psnr_log = AverageMeter()
    ssim_log = AverageMeter()

    rmse_list = []
    psnr_list = []
    ssim_list = []
    num = 0

    with torch.no_grad():
        model.eval()
        for it, batch in enumerate(tqdm(test_dataloader,desc='Processing'), 0):
            imgl = batch['imgl'].cuda(device)
            imgr = batch['imgr'].cuda(device)
            b,c,h,w = imgl.shape


            flow_pred_forward = model(imgl, imgr)
            forward_flow = flow_pred_forward[:,0:2,:,:].view(b,2,h,w)
            flow_pred_backward = model(imgr, imgl)
            backward_flow = flow_pred_backward[:,0:2,:,:].view(b,2,h,w)


            imgl_recon = apply_flow(imgr,backward_flow,max_disp=args.max_disp)
            imgr_recon = apply_flow(imgl,forward_flow,max_disp=args.max_disp)

            rmse_term = calc_rmse(imgr_recon,imgr)
            psnr_term = calc_psnr(imgr_recon,imgr)
            ssim_term = calc_ssim(imgr_recon,imgr)

            for tdx in range(len(rmse_term)):
                rmse_list.append(rmse_term[tdx].cpu().numpy())
                psnr_list.append(psnr_term[tdx].cpu().numpy())
                ssim_list.append(ssim_term[tdx].cpu().numpy())


            rmse_log.update(torch.mean(rmse_term),b)
            psnr_log.update(torch.mean(psnr_term),b)
            ssim_log.update(torch.mean(ssim_term),b)



    message = 'Test Metrics======== RMSE: {metric1.ave:.4f}, PSNR: {metric2.ave:.4f}, SSIM: {metric3.ave:.4f} '.format(
            metric1=rmse_log,metric2=psnr_log,metric3=ssim_log)
    LOGGER.info(message)


if __name__ == '__main__':
    args = get_args()
    if args.mode == 'train':
        if args.task == 'Flow':
            train_flow()
    if args.mode == 'test':
        if args.task == 'Flow':
            checkpoint = ''
            _test_flow(checkpoint=checkpoint)



