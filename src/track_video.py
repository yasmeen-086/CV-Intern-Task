"""Task 4: segment + track individual eggs in a video.

Per frame : classical segmentation (final config, up to --max_eggs instances)
            + optional watershed split of merged / touching eggs.
Association: Hungarian matching, cost = (1 - IoU of velocity-shifted track mask vs detection) + centroid distance term.
Track life : confirmed after --min_hits, dropped after --max_missed missed frames (coasts on constant-velocity prediction).
Outputs    : annotated mp4 (coloured mask + persistent ID), tracks.csv (frame,id,cx,cy,area), summary.
"""
import argparse, json, os, subprocess, csv
import cv2, numpy as np
from scipy.optimize import linear_sum_assignment
from dataclasses import replace
from eggseg import Cfg, segment_instances
from evaluate import load_cfg

PALETTE = [(66, 135, 245), (60, 200, 90), (245, 170, 50), (200, 60, 200), (50, 200, 230), (240, 90, 90),
           (150, 150, 255), (120, 220, 160), (255, 200, 120), (180, 120, 60)]


def split_touching(mask, area_ratio=1.6, ref_area=None):
    """Split merged eggs. Blobs much bigger than the typical egg are cut by
    (1) marker-based watershed on the distance transform with an adaptive seed threshold, else
    (2) k-means on pixel coordinates with k = round(area / typical egg area)."""
    if ref_area is None or mask.sum() < area_ratio * ref_area:
        return [mask]
    k = int(round(mask.sum() / ref_area))
    if k < 2:
        return [mask]
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    for frac in np.arange(0.9, 0.45, -0.05):             # highest threshold that yields >=2 seeds
        seeds = (dist >= frac * dist.max()).astype(np.uint8)
        n, markers = cv2.connectedComponents(seeds)
        if n - 1 >= 2:
            markers = markers + 1
            markers[(mask > 0) & (seeds == 0)] = 0        # unknown region
            markers[mask == 0] = 1                        # background label
            ws = cv2.watershed(cv2.cvtColor(mask * 255, cv2.COLOR_GRAY2BGR), markers.astype(np.int32))
            parts = [((ws == l) & (mask > 0)).astype(np.uint8) for l in range(2, n + 1)]
            parts = [p for p in parts if p.sum() > 0.25 * ref_area]
            if len(parts) >= 2:
                return parts
    ys, xs = np.nonzero(mask)                              # fallback
    pts = np.float32(np.stack([xs, ys], 1))
    cv2.setRNGSeed(0)
    _, lbl, _ = cv2.kmeans(pts, k, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5), 3, cv2.KMEANS_PP_CENTERS)
    parts = []
    for c in range(k):
        p = np.zeros_like(mask); sel = lbl.ravel() == c
        p[ys[sel], xs[sel]] = 1
        parts.append(p)
    return parts


class Track:
    _next = 1
    def __init__(self, mask, c):
        self.id = Track._next; Track._next += 1
        self.mask, self.c, self.v = mask, c, np.zeros(2)
        self.hits, self.missed, self.age = 1, 0, 1
        self.area = float(mask.sum())

    def predict(self):
        return self.c + self.v

    def shifted_mask(self):
        dx, dy = self.v
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        return cv2.warpAffine(self.mask, M, self.mask.shape[::-1], flags=cv2.INTER_NEAREST)

    def update(self, mask, c):
        self.v = 0.6 * self.v + 0.4 * (c - self.c)
        self.mask, self.c = mask, c
        self.area = 0.7 * self.area + 0.3 * float(mask.sum())
        self.hits += 1; self.missed = 0; self.age += 1


def centroid(m):
    ys, xs = np.nonzero(m)
    return np.array([xs.mean(), ys.mean()])


def iou(a, b):
    u = np.logical_or(a, b).sum()
    return 0.0 if u == 0 else np.logical_and(a, b).sum() / u


def associate(tracks, dets, diag, w_dist=0.5, gate=0.9):
    if not tracks or not dets:
        return [], list(range(len(tracks))), list(range(len(dets)))
    C = np.zeros((len(tracks), len(dets)))
    for i, t in enumerate(tracks):
        sm = t.shifted_mask(); pc = t.predict()
        for j, (dm, dc) in enumerate(dets):
            C[i, j] = (1 - iou(sm, dm)) + w_dist * np.linalg.norm(pc - dc) / (0.25 * diag)
    r, c = linear_sum_assignment(C)
    matches = [(i, j) for i, j in zip(r, c) if C[i, j] < gate]
    mt = {i for i, _ in matches}; md = {j for _, j in matches}
    return matches, [i for i in range(len(tracks)) if i not in mt], [j for j in range(len(dets)) if j not in md]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=None, help="YouTube URL (downloaded with yt-dlp)")
    ap.add_argument("--video", default="data/video.mp4")
    ap.add_argument("--cfg", default="results/final_config.json")
    ap.add_argument("--out", default="results/tracking")
    ap.add_argument("--max_eggs", type=int, default=12)
    ap.add_argument("--min_score", type=float, default=None, help="override cfg.min_score for the video")
    ap.add_argument("--max_area", type=float, default=0.3, help="max egg area as fraction of frame (rejects trays/hands/background)")
    ap.add_argument("--proc_side", type=int, default=640, help="resize frames so the long side is this (speed)")
    ap.add_argument("--min_hits", type=int, default=3)
    ap.add_argument("--max_missed", type=int, default=15)
    ap.add_argument("--assoc_gate", type=float, default=0.9,
                    help="maximum association cost; increase to retain IDs through longer occlusions")
    ap.add_argument("--no_split", action="store_true")
    ap.add_argument("--max_frames", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.url and not os.path.exists(a.video):
        os.makedirs(os.path.dirname(a.video) or ".", exist_ok=True)
        subprocess.check_call(["yt-dlp", "-f", "mp4/bestvideo[height<=720]+bestaudio/best", "-o", a.video, a.url])
    cfg = load_cfg(a.cfg)
    cfg = replace(cfg, max_eggs=a.max_eggs, selection="score", force_one=False, max_area=a.max_area)
    if a.min_score is not None:
        cfg = replace(cfg, min_score=a.min_score)

    cap = cv2.VideoCapture(a.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    W0, H0 = int(cap.get(3)), int(cap.get(4))
    s = min(1.0, a.proc_side / max(W0, H0)); W, H = int(W0 * s), int(H0 * s)
    vw = cv2.VideoWriter(os.path.join(a.out, "tracked.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    rows, tracks, ref_area, fidx = [], [], None, 0
    diag = float(np.hypot(W, H))
    while True:
        ok, frame = cap.read()
        if not ok or (a.max_frames and fidx >= a.max_frames): break
        frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA) if s < 1 else frame
        insts = segment_instances(frame, cfg)
        dets = []
        for m in insts:
            parts = [m] if a.no_split else split_touching(m, ref_area=ref_area)
            dets += [(p, centroid(p)) for p in parts if p.sum() > 0]
        if dets:                                         # running median of single-egg area -> split threshold
            ar = np.median([d[0].sum() for d in dets])
            ref_area = ar if ref_area is None else 0.95 * ref_area + 0.05 * ar
        matches, lost, new = associate(tracks, dets, diag, gate=a.assoc_gate)
        for i, j in matches: tracks[i].update(*dets[j])
        for i in lost:
            t = tracks[i]; t.missed += 1; t.age += 1; t.c = t.predict()
        for j in new: tracks.append(Track(*dets[j]))
        tracks = [t for t in tracks if t.missed <= a.max_missed and not (t.hits < a.min_hits and t.missed > 3)]

        vis = frame.copy()
        for t in tracks:
            if t.hits < a.min_hits or t.missed > 0: continue
            col = PALETTE[(t.id - 1) % len(PALETTE)]
            vis[t.mask > 0] = (0.5 * np.array(col) + 0.5 * vis[t.mask > 0]).astype(np.uint8)
            cnts, _ = cv2.findContours(t.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, cnts, -1, col, 2)
            cv2.putText(vis, f"egg {t.id}", (int(t.c[0]) - 25, int(t.c[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            rows.append((fidx, t.id, round(float(t.c[0]), 1), round(float(t.c[1]), 1), int(t.mask.sum())))
        cv2.putText(vis, f"frame {fidx}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        vw.write(vis); fidx += 1
        if fidx % 50 == 0: print("frame", fidx)
    vw.release(); cap.release()
    with open(os.path.join(a.out, "tracks.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["frame", "id", "cx", "cy", "area"]); w.writerows(rows)
    ids = sorted({r[1] for r in rows})
    lens = {i: sum(1 for r in rows if r[1] == i) for i in ids}
    print(f"{fidx} frames, {len(ids)} distinct track IDs; longest tracks: {sorted(lens.values(), reverse=True)[:10]}")


if __name__ == "__main__":
    main()
