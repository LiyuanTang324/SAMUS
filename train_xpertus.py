"""Train SAMUS on all XpertUS segmentation datasets.

Usage:
    python train_xpertus.py --modelname SAMUS --task XpertUS

Trains on the union of all segmentation datasets' train splits.
Validates per-dataset on val splits each epoch.
Saves weights following UniUSNet conventions (latest + best with symlinks).
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import time
import logging
import sys
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils.config import get_config
from utils.data_us import JointTransform2D
from utils.data_xpertus import XpertUSDataset, XpertUSSingleDataset, _discover_seg_datasets
from utils.loss_functions.sam_loss import get_criterion
from utils.generate_prompts import get_click_prompt
from models.model_dict import get_model
import utils.metrics as metrics
from hausdorff import hausdorff_distance


def validate_per_dataset(model, criterion, opt, args, tf_val):
    """Run validation on each segmentation dataset separately.

    Returns (per_dataset_dices, avg_dice) where per_dataset_dices is a dict.
    """
    model.eval()
    dataset_names = _discover_seg_datasets(opt.data_path, 'val')
    per_dataset_dices = {}

    for ds_name in dataset_names:
        val_dataset = XpertUSSingleDataset(
            data_root=opt.data_path,
            dataset_name=ds_name,
            split='val',
            joint_transform=tf_val,
            img_size=args.encoder_input_size,
        )
        valloader = DataLoader(
            val_dataset, batch_size=opt.batch_size,
            shuffle=False, num_workers=8, pin_memory=True,
        )

        dice_sum = 0.0
        count = 0

        for datapack in valloader:
            imgs = datapack['image'].to(dtype=torch.float32, device=opt.device)
            masks = datapack['low_mask'].to(dtype=torch.float32, device=opt.device)
            label = datapack['label'].to(dtype=torch.float32, device=opt.device)
            pt = get_click_prompt(datapack, opt)

            with torch.no_grad():
                pred = model(imgs, pt)

            if args.modelname == 'MSA' or args.modelname == 'SAM':
                gt = masks.detach().cpu().numpy()
            else:
                gt = label.detach().cpu().numpy()
            gt = gt[:, 0, :, :]

            predict = torch.sigmoid(pred['masks'])
            predict = predict.detach().cpu().numpy()
            seg = predict[:, 0, :, :] > 0.5

            b = seg.shape[0]
            for j in range(b):
                pred_i = np.zeros((1, seg.shape[1], seg.shape[2]))
                pred_i[seg[j:j+1] == 1] = 255
                gt_i = np.zeros((1, gt.shape[1], gt.shape[2]))
                gt_i[gt[j:j+1] == 1] = 255
                dice_i = metrics.dice_coefficient(pred_i, gt_i)
                dice_sum += dice_i
                count += 1

        ds_dice = dice_sum / max(count, 1)
        per_dataset_dices[ds_name] = ds_dice

    avg_dice = np.mean(list(per_dataset_dices.values())) if per_dataset_dices else 0.0
    return per_dataset_dices, avg_dice


def main():
    parser = argparse.ArgumentParser(description='Train SAMUS on XpertUS')
    parser.add_argument('--modelname', default='SAMUS', type=str,
                        help='type of model, e.g., SAM, SAMFull, MedSAM, MSA, SAMed, SAMUS')
    parser.add_argument('-encoder_input_size', type=int, default=256)
    parser.add_argument('-low_image_size', type=int, default=128)
    parser.add_argument('--task', default='XpertUS', help='task or dataset name')
    parser.add_argument('--vit_name', type=str, default='vit_b')
    parser.add_argument('--sam_ckpt', type=str, default='checkpoints/sam_vit_b_01ec64.pth')
    parser.add_argument('--batch_size', type=int, default=8, help='batch_size per gpu')
    parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
    parser.add_argument('--base_lr', type=float, default=0.0005)
    parser.add_argument('--warmup', type=bool, default=False)
    parser.add_argument('--warmup_period', type=int, default=250)
    parser.add_argument('-keep_log', type=bool, default=True)

    args = parser.parse_args()
    opt = get_config(args.task)
    device = torch.device(opt.device)

    os.makedirs(opt.save_path, exist_ok=True)

    logging.basicConfig(
        filename=os.path.join(opt.save_path, 'log.txt'),
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s',
        datefmt='%H:%M:%S',
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    if args.keep_log:
        logtimestr = time.strftime('%m%d%H%M')
        boardpath = opt.tensorboard_path + args.modelname + opt.save_path_code + logtimestr
        os.makedirs(boardpath, exist_ok=True)
        TensorWriter = SummaryWriter(boardpath)

    # ---- reproducibility ----
    seed_value = getattr(opt, 'seed', 42)
    np.random.seed(seed_value)
    random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True

    # ---- model ----
    model = get_model(args.modelname, args=args, opt=opt)
    opt.batch_size = args.batch_size * args.n_gpu

    # ---- data ----
    tf_train = JointTransform2D(
        img_size=args.encoder_input_size, low_img_size=args.low_image_size,
        ori_size=opt.img_size, crop=opt.crop,
        p_flip=0.0, p_rota=0.5, p_scale=0.5, p_gaussn=0.0,
        p_contr=0.5, p_gama=0.5, p_distor=0.0,
        color_jitter_params=None, long_mask=True,
    )
    tf_val = JointTransform2D(
        img_size=args.encoder_input_size, low_img_size=args.low_image_size,
        ori_size=opt.img_size, crop=opt.crop,
        p_flip=0, color_jitter_params=None, long_mask=True,
    )

    train_dataset = XpertUSDataset(
        data_root=opt.data_path, split='train',
        joint_transform=tf_train, img_size=args.encoder_input_size,
    )
    trainloader = DataLoader(
        train_dataset, batch_size=opt.batch_size,
        shuffle=True, num_workers=8, pin_memory=True,
    )

    logging.info(f"Training samples: {len(train_dataset)}  |  "
                 f"Datasets: {train_dataset.dataset_names}")

    model.to(device)
    if opt.pre_trained:
        checkpoint = torch.load(opt.load_path)
        new_state_dict = {}
        for k, v in checkpoint.items():
            if k[:7] == 'module.':
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict)

    if args.n_gpu > 1:
        model = nn.DataParallel(model)

    if args.warmup:
        b_lr = args.base_lr / args.warmup_period
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=b_lr, betas=(0.9, 0.999), weight_decay=0.1,
        )
    else:
        b_lr = args.base_lr
        optimizer = optim.Adam(
            model.parameters(), lr=args.base_lr,
            betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False,
        )

    criterion = get_criterion(modelname=args.modelname, opt=opt)

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info("Total_params: {}".format(pytorch_total_params))

    # ---- training loop ----
    iter_num = 0
    max_iterations = opt.epochs * len(trainloader)
    best_performance = 0.0
    best_epoch = 0
    loss_log = np.zeros(opt.epochs + 1)
    dice_log = np.zeros(opt.epochs + 1)

    for epoch in range(opt.epochs):
        model.train()
        train_losses = 0

        for batch_idx, datapack in enumerate(trainloader):
            imgs = datapack['image'].to(dtype=torch.float32, device=opt.device)
            masks = datapack['low_mask'].to(dtype=torch.float32, device=opt.device)
            bbox = torch.as_tensor(datapack['bbox'], dtype=torch.float32, device=opt.device)
            pt = get_click_prompt(datapack, opt)

            pred = model(imgs, pt, bbox)
            train_loss = criterion(pred, masks)

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            train_losses += train_loss.item()

            if args.warmup and iter_num < args.warmup_period:
                lr_ = args.base_lr * ((iter_num + 1) / args.warmup_period)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
            else:
                if args.warmup:
                    shift_iter = iter_num - args.warmup_period
                    assert shift_iter >= 0, f'Shift iter is {shift_iter}, smaller than zero'
                lr_ = args.base_lr * (1.0 - iter_num / max_iterations) ** 0.9
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_

            iter_num += 1

        avg_train_loss = train_losses / (batch_idx + 1)
        logging.info('epoch [{}/{}], train loss:{:.4f}'.format(epoch, opt.epochs, avg_train_loss))
        if args.keep_log:
            TensorWriter.add_scalar('train_loss', avg_train_loss, epoch)
            TensorWriter.add_scalar('learning_rate', optimizer.state_dict()['param_groups'][0]['lr'], epoch)
            loss_log[epoch] = avg_train_loss

        # ---- save latest (rolling, UniUSNet-style) ----
        save_latest_path = os.path.join(opt.save_path, 'latest_{}.pth'.format(epoch))
        prev_latest = os.path.join(opt.save_path, 'latest_{}.pth'.format(epoch - 1))
        latest_link = os.path.join(opt.save_path, 'latest.pth')
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
        }, save_latest_path)
        if os.path.exists(prev_latest):
            os.remove(prev_latest)
        if os.path.islink(latest_link) or os.path.exists(latest_link):
            os.remove(latest_link)
        os.symlink(os.path.abspath(save_latest_path), latest_link)

        # ---- per-dataset validation ----
        if epoch % opt.eval_freq == 0:
            per_ds_dices, avg_dice = validate_per_dataset(
                model, criterion, opt, args, tf_val,
            )
            for ds_name, ds_dice in per_ds_dices.items():
                logging.info('  val {} dice: {:.4f}'.format(ds_name, ds_dice))
                if args.keep_log:
                    TensorWriter.add_scalar('val_dice/{}'.format(ds_name), ds_dice, epoch)

            logging.info('epoch [{}/{}], val avg dice:{:.4f}'.format(epoch, opt.epochs, avg_dice))
            if args.keep_log:
                TensorWriter.add_scalar('val_dice_avg', avg_dice, epoch)
                dice_log[epoch] = avg_dice

            # ---- save best (rolling, UniUSNet-style) ----
            if avg_dice > best_performance:
                old_best = os.path.join(
                    opt.save_path,
                    'best_model_{}_{}.pth'.format(best_epoch, round(best_performance, 4)),
                )
                if os.path.exists(old_best):
                    os.remove(old_best)
                best_link = os.path.join(opt.save_path, 'best_model.pth')
                if os.path.islink(best_link) or os.path.exists(best_link):
                    os.remove(best_link)

                best_epoch = epoch
                best_performance = avg_dice
                save_best_path = os.path.join(
                    opt.save_path,
                    'best_model_{}_{}.pth'.format(epoch, round(best_performance, 4)),
                )
                torch.save(model.state_dict(), save_best_path)
                os.symlink(os.path.abspath(save_best_path), best_link)
                logging.info('New best model saved: epoch {} dice {:.4f}'.format(epoch, avg_dice))

            logging.info('Best so far: epoch {} dice {:.4f}'.format(best_epoch, best_performance))

    if args.keep_log:
        TensorWriter.close()
        log_dir = opt.tensorboard_path + args.modelname + opt.save_path_code + logtimestr
        with open(os.path.join(log_dir, 'trainloss.txt'), 'w') as f:
            for v in loss_log:
                f.write(str(v) + '\n')
        with open(os.path.join(log_dir, 'dice.txt'), 'w') as f:
            for v in dice_log:
                f.write(str(v) + '\n')

    logging.info("Training finished!")


if __name__ == '__main__':
    main()
