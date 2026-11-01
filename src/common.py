"""Shared helpers: IO, skin mask, shape scoring, metrics."""
import os, glob
import cv2
import numpy as np

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def list_images(folder):
    return [f for f in sorted(glob.glob(os.path.join(folder, "**", "*"), recursive=True))
            if f.lower().endswith(IMG_EXT)]


def read_img(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"cannot read {path}")
    return img


def stem(path):
    return os.path.splitext(os.path.basename(path))[0]


def skin_mask(bgr):
    """Classic YCrCb skin range (hands appear in most images)."""
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    cr, cb = ycc[..., 1], ycc[..., 2]
    return (cr >= 133) & (cr <= 173) & (cb >= 77) & (cb <= 127)


def fill_holes(mask):
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros(mask.shape, dtype=np.uint8)
    cv2.drawContours(out, cnts, -1, 1, thickness=-1)
    return out


def ellipse_fit_iou(comp, cnt):
    """Returns (IoU between component and its fitted ellipse, ellipse). (0, None) if cannot fit."""
    if len(cnt) < 5:
        return 0.0, None
    el = cv2.fitEllipse(cnt)
    em = np.zeros(comp.shape, dtype=np.uint8)
    cv2.ellipse(em, el, 1, -1)
    inter = np.logical_and(em, comp).sum()
    union = np.logical_or(em, comp).sum()
    return float(inter) / max(1, union), el


def mask_iou(a, b):
    a, b = a.astype(bool), b.astype(bool)
    u = np.logical_or(a, b).sum()
    return 1.0 if u == 0 else float(np.logical_and(a, b).sum()) / u


# ---------------------------------------------------------------- metrics
def boundary_f1(pred, gt, tol_frac=0.01):
    """Boundary F1 (tolerance = tol_frac * image diagonal) and HD95 in pixels."""
    pred, gt = pred.astype(np.uint8), gt.astype(np.uint8)
    h, w = gt.shape
    diag = float(np.hypot(h, w))
    tol = max(1.0, tol_frac * diag)
    k = np.ones((3, 3), np.uint8)
    pb = pred - cv2.erode(pred, k)
    gb = gt - cv2.erode(gt, k)
    if pb.sum() == 0 and gb.sum() == 0:
        return 1.0, 0.0
    if pb.sum() == 0 or gb.sum() == 0:
        return 0.0, diag
    dt_g = cv2.distanceTransform((1 - gb).astype(np.uint8), cv2.DIST_L2, 3)
    dt_p = cv2.distanceTransform((1 - pb).astype(np.uint8), cv2.DIST_L2, 3)
    prec = (dt_g[pb > 0] <= tol).mean()
    rec = (dt_p[gb > 0] <= tol).mean()
    f1 = 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
    hd95 = max(np.percentile(dt_g[pb > 0], 95), np.percentile(dt_p[gb > 0], 95))
    return float(f1), float(hd95)


def seg_metrics(pred, gt):
    p, g = pred.astype(bool), gt.astype(bool)
    tp = np.logical_and(p, g).sum()
    fp = np.logical_and(p, ~g).sum()
    fn = np.logical_and(~p, g).sum()
    iou = 1.0 if (tp + fp + fn) == 0 else tp / (tp + fp + fn)
    dice = 1.0 if (tp + fp + fn) == 0 else 2 * tp / (2 * tp + fp + fn)
    bf1, hd95 = boundary_f1(p, g)
    return dict(iou=float(iou), dice=float(dice),
                precision=float(tp / max(1, tp + fp)), recall=float(tp / max(1, tp + fn)),
                boundary_f1=bf1, hd95=hd95,
                gt_area_frac=float(g.mean()), pred_area_frac=float(p.mean()))


def overlay(bgr, mask, color=(0, 0, 255), alpha=0.5):
    out = bgr.copy()
    m = mask.astype(bool)
    out[m] = (alpha * np.array(color) + (1 - alpha) * out[m]).astype(np.uint8)
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnts, -1, (255, 255, 255), 1)
    return out
