import numpy as np
import torch
from scipy.ndimage import distance_transform_edt, binary_erosion, generate_binary_structure


def dice_coefficient(pred, gt, smooth=1e-5):
    """ computational formula：
        dice = 2TP/(FP + 2TP + FN)
    """
    N = gt.shape[0]
    pred[pred >= 1] = 1
    gt[gt >= 1] = 1
    pred_flat = pred.reshape(N, -1)
    gt_flat = gt.reshape(N, -1)
    intersection = (pred_flat * gt_flat).sum(1)
    unionset = pred_flat.sum(1) + gt_flat.sum(1)
    dice = (2 * intersection + smooth) / (unionset + smooth)
    return dice.sum() / N


def iou_coefficient(pred, gt, smooth=1e-5):
    """Intersection over Union (Jaccard Index).

    iou = TP / (TP + FP + FN)
    """
    N = gt.shape[0]
    pred[pred >= 1] = 1
    gt[gt >= 1] = 1
    pred_flat = pred.reshape(N, -1)
    gt_flat = gt.reshape(N, -1)
    TP = (pred_flat * gt_flat).sum(1)
    FP = pred_flat.sum(1) - TP
    FN = gt_flat.sum(1) - TP
    iou = (TP + smooth) / (TP + FP + FN + smooth)
    return iou.sum() / N


def hausdorff_95(pred, gt):
    """95th Percentile Hausdorff Distance (HD95).

    Surface-based HD95 between two binary 2D masks.
    pred and gt: 2D numpy arrays with values in {0, 1} or {0, 255}.
    Returns Euclidean HD95 in pixels.
    """
    pred = (pred > 0).astype(bool)
    gt = (gt > 0).astype(bool)

    if not pred.any() and not gt.any():
        return 0.0
    if not pred.any() or not gt.any():
        return np.sqrt(pred.shape[0] ** 2 + pred.shape[1] ** 2)

    conn = generate_binary_structure(pred.ndim, 1)

    pred_border = pred ^ binary_erosion(pred, structure=conn, border_value=0)
    gt_border = gt ^ binary_erosion(gt, structure=conn, border_value=0)

    if not pred_border.any():
        pred_border = pred
    if not gt_border.any():
        gt_border = gt

    dt_gt = distance_transform_edt(~gt_border)
    dt_pred = distance_transform_edt(~pred_border)

    sds_pred_to_gt = dt_gt[pred_border]
    sds_gt_to_pred = dt_pred[gt_border]

    all_surface_distances = np.concatenate([sds_pred_to_gt, sds_gt_to_pred])
    return np.percentile(all_surface_distances, 95)


def get_matrix(pred, gt, smooth=1e-5):
    N = gt.shape[0]
    pred[pred >= 1] = 1
    gt[gt >= 1] = 1
    pred_flat = pred.reshape(N, -1)
    gt_flat = gt.reshape(N, -1)
    TP = (pred_flat * gt_flat).sum(1)
    FN = gt_flat.sum(1) - TP
    pred_flat_no = (pred_flat + 1) % 2
    gt_flat_no = (gt_flat + 1) % 2
    TN = (pred_flat_no * gt_flat_no).sum(1)
    FP = pred_flat.sum(1) - TP
    return TP, FP, TN, FN