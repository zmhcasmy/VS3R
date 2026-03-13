import math

import cv2
import numpy as np


def f_measure(foreground_mask, gt_mask, void_pixels=None, bound_th=0.008):
    """
    Compute mean,recall and decay from per-frame evaluation.
    Calculates precision/recall for boundaries between foreground_mask and
    gt_mask using morphological operators to speed it up.

    Arguments:
        foreground_mask (ndarray): binary segmentation image.
        gt_mask         (ndarray): binary annotated image.
        void_pixels     (ndarray): optional mask with void pixels

    Returns:
        F (float): boundaries F-measure
    """
    assert np.atleast_3d(foreground_mask).shape[2] == 1
    if void_pixels is not None:
        void_pixels = void_pixels.astype(bool)
    else:
        void_pixels = np.zeros_like(foreground_mask).astype(bool)

    bound_pix = bound_th if bound_th >= 1 else \
        np.ceil(bound_th * np.linalg.norm(foreground_mask.shape))

    # Get the pixel boundaries of both masks
    fg_boundary = _seg2bmap(foreground_mask * np.logical_not(void_pixels))
    gt_boundary = _seg2bmap(gt_mask * np.logical_not(void_pixels))

    from skimage.morphology import disk

    # fg_dil = binary_dilation(fg_boundary, disk(bound_pix))
    fg_dil = cv2.dilate(fg_boundary.astype(np.uint8),
                        disk(bound_pix).astype(np.uint8))
    # gt_dil = binary_dilation(gt_boundary, disk(bound_pix))
    gt_dil = cv2.dilate(gt_boundary.astype(np.uint8),
                        disk(bound_pix).astype(np.uint8))

    # Get the intersection
    gt_match = gt_boundary * fg_dil
    fg_match = fg_boundary * gt_dil

    # Area of the intersection
    n_fg = np.sum(fg_boundary)
    n_gt = np.sum(gt_boundary)

    # % Compute precision and recall
    if n_fg == 0 and n_gt > 0:
        precision = 1
        recall = 0
    elif n_fg > 0 and n_gt == 0:
        precision = 0
        recall = 1
    elif n_fg == 0 and n_gt == 0:
        precision = 1
        recall = 1
    else:
        precision = np.sum(fg_match) / float(n_fg)
        recall = np.sum(gt_match) / float(n_gt)

    # Compute F measure
    if precision + recall == 0:
        F = 0
    else:
        F = 2 * precision * recall / (precision + recall)

    return F


def _seg2bmap(seg, width=None, height=None):
    """
    From a segmentation, compute a binary boundary map with 1 pixel wide
    boundaries.  The boundary pixels are offset by 1/2 pixel towards the
    origin from the actual segment boundary.
    Arguments:
        seg     : Segments labeled from 1..k.
        width	  :	Width of desired bmap  <= seg.shape[1]
        height  :	Height of desired bmap <= seg.shape[0]
    Returns:
        bmap (ndarray):	Binary boundary map.
     David Martin <dmartin@eecs.berkeley.edu>
     January 2003
    """

    seg = seg.astype(bool)
    seg[seg > 0] = 1

    assert np.atleast_3d(seg).shape[2] == 1

    width = seg.shape[1] if width is None else width
    height = seg.shape[0] if height is None else height

    h, w = seg.shape[:2]

    ar1 = float(width) / float(height)
    ar2 = float(w) / float(h)

    assert not (
        width > w | height > h | abs(ar1 - ar2) > 0.01
    ), "Can" "t convert %dx%d seg to %dx%d bmap." % (w, h, width, height)

    e = np.zeros_like(seg)
    s = np.zeros_like(seg)
    se = np.zeros_like(seg)

    e[:, :-1] = seg[:, 1:]
    s[:-1, :] = seg[1:, :]
    se[:-1, :-1] = seg[1:, 1:]

    b = seg ^ e | seg ^ s | seg ^ se
    b[-1, :] = seg[-1, :] ^ e[-1, :]
    b[:, -1] = seg[:, -1] ^ s[:, -1]
    b[-1, -1] = 0

    if w == width and h == height:
        bmap = b
    else:
        bmap = np.zeros((height, width))
        for x in range(w):
            for y in range(h):
                if b[y, x]:
                    j = 1 + math.floor((y - 1) + height / h)
                    i = 1 + math.floor((x - 1) + width / h)
                    bmap[j, i] = 1

    return bmap


def eval_iou(gt_masks, res_masks):
    """
    gt_masks: (N, H, W)
    res_masks: (N, H, W)
    """
    inters = np.logical_and(gt_masks, res_masks)
    outers = np.logical_or(gt_masks, res_masks)

    inters = np.sum(inters, axis=(1, 2))
    outers = np.sum(outers, axis=(1, 2))

    ious = inters / outers
    ious[np.isclose(outers, 0)] = 0
    return ious


def eval_tversky(gt_masks, res_masks):
    """
    gt_masks: (N, H, W)
    res_masks: (N, H, W)
    """
    tp = np.logical_and(gt_masks, res_masks)
    fn = np.logical_and(gt_masks, np.logical_not(res_masks))
    fp = np.logical_and(np.logical_not(gt_masks), res_masks)
    tp = tp.sum(axis=(1, 2))
    fn = fn.sum(axis=(1, 2))
    fp = fp.sum(axis=(1, 2))
    tversky = tp / (tp + 2 * fn + 2 * fp)
    return tversky


def eval_boundary(gt_masks, res_masks):
    """
    gt_masks: (N, H, W)
    res_masks: (N, H, W)
    """
    if gt_masks.shape != res_masks.shape or gt_masks.ndim != 3:
        raise ValueError(
            f"Expected (N,H,W) masks with same shape, got {gt_masks.shape=} {res_masks.shape=}")

    N = gt_masks.shape[0]
    f_res = np.zeros(N, dtype=np.float32)

    for i in range(N):
        f_res[i] = f_measure(
            foreground_mask=res_masks[i],
            gt_mask=gt_masks[i],
            void_pixels=None,
            bound_th=0.008
        )

    return f_res


def eval_statistics(metrics_res):
    """
    metrics_res: (N, )
    """
    M = np.nanmean(metrics_res)
    R = np.nanmean(metrics_res > 0.5)
    N_bins = 4
    ids = np.round(np.linspace(1, len(metrics_res), N_bins + 1) + 1e-10) - 1
    ids = ids.astype(np.uint8)

    D_bins = [metrics_res[ids[i]:ids[i + 1] + 1] for i in range(0, 4)]

    D = np.nanmean(D_bins[0]) - np.nanmean(D_bins[3])
    return M, R, D
