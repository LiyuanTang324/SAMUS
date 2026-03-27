"""Test SAMUS per-dataset on XpertUS segmentation datasets.

Usage:
    python test_xpertus.py --modelname SAMUS --task XpertUS

Iterates over every segmentation dataset that has a test.txt,
evaluates Dice per dataset, and writes results to result.csv
(same format as UniUSNet).
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import argparse
import csv
import time
import logging
import sys
import torch
import numpy as np
import random
from torch.utils.data import DataLoader

from utils.config import get_config
from utils.data_us import JointTransform2D
from utils.data_xpertus import XpertUSSingleDataset, _discover_seg_datasets
from utils.loss_functions.sam_loss import get_criterion
from utils.generate_prompts import get_click_prompt
from models.model_dict import get_model
import utils.metrics as metrics
from thop import profile


def inference(args, model, opt, tf_val):
    CSV_HEADER = [
        "dataset", "task_type", "prompt",
        "DSC", "IoU", "HD95",
        "AUC", "Macro_F1", "Sens@Spec90", "Sens@Spec95",
        "FPS", "time",
    ]
    os.makedirs(opt.result_path, exist_ok=True)
    csv_path = os.path.join(opt.result_path, 'result.csv')
    if not os.path.exists(csv_path):
        with open(csv_path, 'w', newline='') as f:
            csv.writer(f).writerow(CSV_HEADER)

    log_path = os.path.join(opt.result_path, 'test_result.txt')
    logging.basicConfig(
        filename=log_path, level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S',
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info('checkpoint: {}'.format(opt.load_path))

    seg_datasets = _discover_seg_datasets(opt.data_path, 'test')
    logging.info('Found {} segmentation test sets: {}'.format(
        len(seg_datasets), seg_datasets))

    model.eval()
    all_results = {}
    torch.cuda.reset_peak_memory_stats()

    for ds_name in seg_datasets:
        test_dataset = XpertUSSingleDataset(
            data_root=opt.data_path,
            dataset_name=ds_name,
            split='test',
            joint_transform=tf_val,
            img_size=args.encoder_input_size,
        )
        testloader = DataLoader(
            test_dataset, batch_size=opt.batch_size,
            shuffle=False, num_workers=8, pin_memory=True,
        )
        logging.info('{}: {} test samples'.format(ds_name, len(test_dataset)))

        max_samples = opt.batch_size * (len(testloader) + 1)
        dices = np.zeros(max_samples)
        hds = np.zeros(max_samples)
        ious = np.zeros(max_samples)
        eval_number = 0
        ds_total_time = 0.0
        ds_total_samples = 0

        for datapack in testloader:
            imgs = datapack['image'].to(dtype=torch.float32, device=opt.device)
            masks = datapack['low_mask'].to(dtype=torch.float32, device=opt.device)
            label = datapack['label'].to(dtype=torch.float32, device=opt.device)
            pt = get_click_prompt(datapack, opt)

            torch.cuda.synchronize()
            t_batch_start = time.time()
            with torch.no_grad():
                pred = model(imgs, pt)
            torch.cuda.synchronize()
            ds_total_time += time.time() - t_batch_start
            ds_total_samples += imgs.shape[0]

            if args.modelname in ('MSA', 'SAM'):
                gt = masks.detach().cpu().numpy()
            else:
                gt = label.detach().cpu().numpy()
            gt = gt[:, 0, :, :]

            predict = torch.sigmoid(pred['masks'])
            predict = predict.detach().cpu().numpy()
            seg = predict[:, 0, :, :] > 0.5

            b, h, w = seg.shape
            for j in range(b):
                pred_i = np.zeros((1, h, w))
                pred_i[seg[j:j+1] == 1] = 255
                gt_i = np.zeros((1, h, w))
                gt_i[gt[j:j+1] == 1] = 255

                dices[eval_number] = metrics.dice_coefficient(pred_i, gt_i)
                ious[eval_number] = metrics.iou_coefficient(pred_i, gt_i)
                hds[eval_number] = metrics.hausdorff_95(pred_i[0], gt_i[0])
                eval_number += 1

        dices = dices[:eval_number]
        hds = hds[:eval_number]
        ious = ious[:eval_number]

        mean_dice = np.mean(dices) * 100
        std_dice = np.std(dices) * 100
        mean_hd = np.mean(hds)
        std_hd = np.std(hds)
        mean_iou = np.mean(ious) * 100
        std_iou = np.std(ious) * 100

        logging.info('--- {} ---'.format(ds_name))
        logging.info('  DSC:  {:.2f} +/- {:.2f}'.format(mean_dice, std_dice))
        logging.info('  HD95: {:.2f} +/- {:.2f}'.format(mean_hd, std_hd))
        logging.info('  IoU:  {:.2f} +/- {:.2f}'.format(mean_iou, std_iou))
        if ds_total_time > 0:
            ds_fps = ds_total_samples / ds_total_time
            logging.info('  Inference FPS: {:.2f} ({} samples in {:.2f}s)'.format(
                ds_fps, ds_total_samples, ds_total_time))

        all_results[ds_name] = mean_dice / 100.0

        seg_fps_str = f"{ds_total_samples / ds_total_time:.2f}" if ds_total_time > 0 else ""
        with open(csv_path, 'a', newline='') as f:
            csv.writer(f).writerow([
                ds_name, "segmentation", False,
                f"{mean_dice / 100.0:.4f}", f"{mean_iou / 100.0:.4f}", f"{mean_hd:.4f}",
                "", "", "", "",
                seg_fps_str,
                time.strftime("%Y-%m-%d %H:%M:%S"),
            ])

    vram_peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    logging.info('VRAM Peak: {:.2f} MB'.format(vram_peak_mb))

    logging.info('========== Summary ==========')
    for ds_name, dice in all_results.items():
        logging.info('  {:15s}  Dice: {:.4f}'.format(ds_name, dice))
    if all_results:
        logging.info('  {:15s}  Dice: {:.4f}'.format(
            'AVERAGE', np.mean(list(all_results.values()))))


def main():
    parser = argparse.ArgumentParser(description='Test SAMUS on XpertUS')
    parser.add_argument('--modelname', default='SAMUS', type=str)
    parser.add_argument('-encoder_input_size', type=int, default=256)
    parser.add_argument('-low_image_size', type=int, default=128)
    parser.add_argument('--task', default='XpertUS', help='task or dataset name')
    parser.add_argument('--vit_name', type=str, default='vit_b')
    parser.add_argument('--sam_ckpt', type=str, default='checkpoints/sam_vit_b_01ec64.pth')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--n_gpu', type=int, default=1)
    parser.add_argument('--base_lr', type=float, default=0.0001)

    args = parser.parse_args()
    opt = get_config(args.task)
    opt.mode = "val"
    opt.modelname = args.modelname
    device = torch.device(opt.device)

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
    opt.batch_size = args.batch_size * args.n_gpu
    model = get_model(args.modelname, args=args, opt=opt)
    model.to(device)

    checkpoint = torch.load(opt.load_path)
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    new_state_dict = {}
    for k, v in state_dict.items():
        if k[:7] == 'module.':
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict)

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total_params: {}".format(pytorch_total_params))

    dummy_input = torch.randn(1, 1, args.encoder_input_size, args.encoder_input_size).to(device)
    dummy_points = (torch.tensor([[[1, 2]]]).float().to(device), torch.tensor([[1]]).float().to(device))
    flops, params = profile(model, inputs=(dummy_input, dummy_points))
    print('GFLOPs: {:.4f}  params: {:.0f}'.format(flops / 1e9, params))

    model.eval()
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input, dummy_points)
        torch.cuda.synchronize()
        t_start = time.time()
        fps_iters = 100
        for _ in range(fps_iters):
            _ = model(dummy_input, dummy_points)
        torch.cuda.synchronize()
        t_end = time.time()
    fps = fps_iters / (t_end - t_start)
    print('FPS: {:.2f}'.format(fps))

    tf_val = JointTransform2D(
        img_size=args.encoder_input_size, low_img_size=args.low_image_size,
        ori_size=opt.img_size, crop=opt.crop,
        p_flip=0, color_jitter_params=None, long_mask=True,
    )

    inference(args, model, opt, tf_val)


if __name__ == '__main__':
    main()
