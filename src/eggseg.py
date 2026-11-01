"""Classical (no deep learning) egg segmentation.

Idea: generate MANY candidate regions from cheap thresholding on several colour channels
(both polarities, optional k-means), then SELECT the ones that look like eggs using a
shape/edge/colour score:
    - how well the region matches its own best-fit ellipse (eggs are ellipses),
    - convexity (solidity),
    - strength of image gradient along its boundary,
    - plausible axis ratio,
    - penalties for touching the image border and for being skin coloured (hands).
Optionally refine with GrabCut and regularise with an ellipse.
Everything is controlled by `Cfg`, so every experiment in the log is one Cfg.
"""
from dataclasses import dataclass, asdict, replace
import cv2
import numpy as np
from common import skin_mask, fill_holes, ellipse_fit_iou, mask_iou


@dataclass(frozen=True)
class Cfg:
    max_side: int = 512            # processing resolution
    channels: tuple = ("gray",)    # any of gray,L,a,b,sat,val,bgdist
    polarities: tuple = ("bright",)  # bright / dark (/ mid when otsu_classes=3)
    blur: int = 0                  # gaussian kernel (odd); 0 = off
    morph_open: int = 0            # kernel size; 0 = off
    morph_close: int = 0
    otsu_classes: int = 2          # 2 = classic Otsu, 3 = multi-Otsu (bright / mid / dark bands)
    thresholds: tuple = ()         # extra fixed thresholds (MSER-like sweep) on each channel, 'bright' side
    sweep_q: tuple = ()            # quantile sweep: thresholds at these quantiles of each channel (scale-free MSER-like candidates)
    ignore: tuple = ()             # fractional boxes (x0,y0,x1,y1) to blank out (logos, captions)
    use_kmeans: bool = False
    kmeans_k: int = 3
    selection: str = "largest"     # "largest" | "score"
    skin_penalty: float = 0.0
    border_penalty: float = 0.2
    min_area: float = 0.01         # fraction of image
    max_area: float = 0.75
    min_score: float = 0.55
    max_eggs: int = 1
    nms_iou: float = 0.4
    grabcut: bool = False
    gc_iters: int = 3
    ellipse: bool = False          # replace/blend mask by fitted ellipse when they agree
    force_one: bool = True         # always return the best candidate even if below min_score
    max_border_frac: float = 1.0   # reject candidates with more than this fraction of the contour on the image border (1.0 = off)
    center_w: float = 0.0          # weight of a "near the image centre" prior in the score (0 = off)
    contrast_w: float = 0.0        # weight of inside-vs-surrounding-ring Lab colour contrast in the score (0 = off)

    def to_dict(self):
        return asdict(self)


# ------------------------------------------------------------------ features
def _channel_maps(bgr, names):
    maps = {}
    need_lab = any(n in names for n in ("L", "a", "b", "bgdist"))
    need_hsv = any(n in names for n in ("sat", "val"))
    if "gray" in names:
        maps["gray"] = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if need_lab:
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        maps["L"], maps["a"], maps["b"] = lab[..., 0], lab[..., 1], lab[..., 2]
        if "bgdist" in names:
            # distance to the median colour of a thin border ring = "how different from background"
            h, w = lab.shape[:2]
            t = max(2, int(0.04 * min(h, w)))
            ring = np.concatenate([lab[:t].reshape(-1, 3), lab[-t:].reshape(-1, 3),
                                   lab[:, :t].reshape(-1, 3), lab[:, -t:].reshape(-1, 3)])
            med = np.median(ring, axis=0)
            d = np.linalg.norm(lab.astype(np.float32) - med.astype(np.float32), axis=2)
            maps["bgdist"] = cv2.normalize(d, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if need_hsv:
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        maps["sat"], maps["val"] = hsv[..., 1], hsv[..., 2]
    return {k: v for k, v in maps.items() if k in names}


def multi_otsu3(img8):
    """Two thresholds (t1<t2) maximising between-class variance of 3 classes (exhaustive, vectorised)."""
    p = np.bincount(img8.ravel(), minlength=256).astype(np.float64)
    p /= p.sum()
    w = np.cumsum(p)
    mu = np.cumsum(p * np.arange(256))
    mt = mu[-1]
    t1, t2 = np.meshgrid(np.arange(256), np.arange(256), indexing="ij")
    w0, m0 = w[t1], mu[t1]
    w1, m1 = w[t2] - w[t1], mu[t2] - mu[t1]
    w2, m2 = 1 - w[t2], mt - mu[t2]
    with np.errstate(divide="ignore", invalid="ignore"):
        v = m0 ** 2 / w0 + m1 ** 2 / w1 + m2 ** 2 / w2
    v[(t2 <= t1) | (w0 < 1e-3) | (w1 < 1e-3) | (w2 < 1e-3)] = -1
    i = np.unravel_index(np.argmax(v), v.shape)
    return int(i[0]), int(i[1])


def _binary_candidates(bgr, cfg):
    """Yield (name, uint8 0/1 mask) candidates."""
    for name, m in _channel_maps(bgr, cfg.channels).items():
        for t in cfg.thresholds:
            yield f"{name}>{t}", (m > t).astype(np.uint8)
        for q in cfg.sweep_q:                      # adaptive sweep: a pale egg shows up at SOME quantile even if Otsu misses it
            t = float(np.quantile(m, q))
            if "bright" in cfg.polarities:
                yield f"{name}>q{q}", (m > t).astype(np.uint8)
            if "dark" in cfg.polarities:
                yield f"{name}<=q{q}", (m <= t).astype(np.uint8)
        if cfg.otsu_classes == 3:
            t1, t2 = multi_otsu3(m)
            bands = {"bright": [m > t2, m > t1], "dark": [m <= t1, m <= t2], "mid": [(m > t1) & (m <= t2)]}
            for pol in cfg.polarities:
                for j, b in enumerate(bands.get(pol, [])):
                    yield f"{name}:{pol}{j}", b.astype(np.uint8)
        else:
            _, th = cv2.threshold(m, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            for pol in cfg.polarities:
                yield f"{name}:{pol}", (th > 0 if pol == "bright" else th == 0).astype(np.uint8)
    if cfg.use_kmeans:
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 15, 1.0)
        cv2.setRNGSeed(0)
        _, lab_lbl, _ = cv2.kmeans(lab, cfg.kmeans_k, None, crit, 2, cv2.KMEANS_PP_CENTERS)
        lab_lbl = lab_lbl.reshape(bgr.shape[:2])
        for c in range(cfg.kmeans_k):
            yield f"km{c}", (lab_lbl == c).astype(np.uint8)


def _clean(mask, cfg):
    h, w = mask.shape
    for x0, y0, x1, y1 in cfg.ignore:
        mask[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)] = 0
    if cfg.morph_open > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.morph_open,) * 2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    if cfg.morph_close > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.morph_close,) * 2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    return mask


# ------------------------------------------------------------------ scoring
def _score_component(comp, cnt, grad, skin, lab, cfg):
    h, w = comp.shape
    area = float(comp.sum())
    af = area / (h * w)
    if af < cfg.min_area or af > cfg.max_area or len(cnt) < 5:
        return 0.0
    hull = cv2.convexHull(cnt)
    solidity = area / max(1.0, cv2.contourArea(hull))
    eiou, el = ellipse_fit_iou(comp, cnt)
    if el is None:
        return 0.0
    ax = sorted(el[1])
    ratio = ax[0] / max(1e-6, ax[1])
    ratio_ok = float(np.clip((ratio - 0.4) / 0.2, 0, 1))          # eggs: minor/major ~0.65-0.9
    edge = np.zeros(comp.shape, np.uint8)
    cv2.drawContours(edge, [cnt], -1, 1, 1)
    g = float(np.clip(grad[edge > 0].mean() / (np.percentile(grad, 95) + 1e-6), 0, 1))
    x, y, bw, bh = cv2.boundingRect(cnt)
    pts = cnt.reshape(-1, 2)
    on_border = np.mean((pts[:, 0] <= 1) | (pts[:, 1] <= 1) | (pts[:, 0] >= w - 2) | (pts[:, 1] >= h - 2))
    if on_border > cfg.max_border_frac:
        return 0.0
    skin_frac = float(skin[comp > 0].mean())
    # rescale so that 'any convex blob' does not score high: real eggs have ellipse-IoU >~0.9, rectangles ~0.65
    e_term = float(np.clip((eiou - 0.6) / 0.35, 0, 1))
    s_term = float(np.clip((solidity - 0.8) / 0.2, 0, 1))
    s = 0.50 * e_term + 0.15 * s_term + 0.20 * g + 0.15 * ratio_ok
    wsum = 1.0                       # optional extra terms; with weights = 0 the score is exactly as before
    if cfg.center_w > 0:
        M = cv2.moments(comp, binaryImage=True)
        d = np.hypot(M["m10"] / M["m00"] / w - 0.5, M["m01"] / M["m00"] / h - 0.5) / 0.7071
        s += cfg.center_w * (1.0 - d); wsum += cfg.center_w
    if cfg.contrast_w > 0:
        k = max(3, int(0.03 * min(h, w)))
        ring = cv2.dilate(comp, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))) - comp
        ct = 0.0
        if ring.sum() > 0:
            ct = float(np.clip(np.linalg.norm(lab[comp > 0].mean(0) - lab[ring > 0].mean(0)) / 30.0, 0, 1))
        s += cfg.contrast_w * ct; wsum += cfg.contrast_w
    s /= wsum
    s -= cfg.border_penalty * on_border + cfg.skin_penalty * skin_frac
    return float(max(0.0, s))


def _extract(cands, bgr, cfg):
    gray = cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    grad = cv2.magnitude(gx, gy)
    skin = skin_mask(bgr)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    out = []
    for name, m in cands:
        m = _clean(m, cfg)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        H, W = m.shape
        for cnt in cnts:
            if cfg.selection == "score":
                # speed/memory: skip regions the scorer would reject anyway (a mask is allocated per region, and threshold
                # sweeps on textured images create thousands of tiny ones). 20% margin keeps results identical.
                ca = cv2.contourArea(cnt)
                if ca < 0.8 * cfg.min_area * H * W or ca > 1.25 * cfg.max_area * H * W:
                    continue
            comp = np.zeros(m.shape, np.uint8)
            cv2.drawContours(comp, [cnt], -1, 1, -1)
            if cfg.selection == "largest":
                sc = float(comp.sum())
            else:
                sc = _score_component(comp, cnt, grad, skin, lab, cfg)
            out.append((sc, name, comp))
    return out


# ------------------------------------------------------------------ refinement
def _grabcut(bgr, comp, iters):
    k = max(3, int(0.03 * min(comp.shape)))
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    sure_fg = cv2.erode(comp, ker, iterations=1)
    probable = cv2.dilate(comp, ker, iterations=1)
    gc = np.full(comp.shape, cv2.GC_BGD, np.uint8)
    gc[probable > 0] = cv2.GC_PR_BGD
    gc[comp > 0] = cv2.GC_PR_FGD
    gc[sure_fg > 0] = cv2.GC_FGD
    if sure_fg.sum() < 20:
        return comp
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(bgr, gc, None, bgd, fgd, iters, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return comp
    res = ((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)).astype(np.uint8)
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(res)
    if n <= 1:
        return comp
    res = (lbl == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])).astype(np.uint8)
    res = fill_holes(res)
    return res if mask_iou(res, comp) > 0.6 else comp      # reject GrabCut runaway


def _ellipse_regularise(comp):
    cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return comp
    cnt = max(cnts, key=cv2.contourArea)
    if len(cnt) < 5:
        return comp
    em = np.zeros(comp.shape, np.uint8)
    cv2.ellipse(em, cv2.fitEllipse(cnt), 1, -1)
    return em if mask_iou(em, comp) > 0.85 else comp


# ------------------------------------------------------------------ public API
def segment_instances(bgr, cfg=Cfg()):
    """Returns list of full-resolution uint8 (0/1) instance masks, best first."""
    H, W = bgr.shape[:2]
    s = min(1.0, cfg.max_side / max(H, W))
    small = cv2.resize(bgr, (int(round(W * s)), int(round(H * s))), interpolation=cv2.INTER_AREA) if s < 1 else bgr
    if cfg.blur > 1:
        small = cv2.GaussianBlur(small, (cfg.blur | 1,) * 2, 0)

    scored = _extract(_binary_candidates(small, cfg), small, cfg)
    scored.sort(key=lambda t: -t[0])
    chosen = []
    for sc, name, comp in scored:
        if len(chosen) >= cfg.max_eggs:
            break
        if sc <= 0:
            break
        if cfg.selection == "score" and sc < cfg.min_score and not (cfg.force_one and not chosen):
            break
        if any(mask_iou(comp, c) > cfg.nms_iou for _, c in chosen):
            continue
        if cfg.selection == "largest" and comp.sum() / comp.size > cfg.max_area:
            continue                        # baseline guard: skip whole-image blobs
        chosen.append((sc, comp))

    out = []
    for sc, comp in chosen:
        if cfg.grabcut:
            comp = _grabcut(small, comp, cfg.gc_iters)
        if cfg.ellipse:
            comp = _ellipse_regularise(comp)
        full = cv2.resize(comp.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR) > 0.5
        out.append(full.astype(np.uint8))
    return out


def segment(bgr, cfg=Cfg()):
    """Union mask (uint8 0/1) at original resolution."""
    inst = segment_instances(bgr, cfg)
    if not inst:
        return np.zeros(bgr.shape[:2], np.uint8)
    return np.clip(np.sum(inst, axis=0), 0, 1).astype(np.uint8)
