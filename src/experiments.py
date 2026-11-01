"""Task 2: reproducible experiment log. Runs ONLY on the dev split.

Trials are cumulative: each trial = best-so-far config + one change. A change is KEPT only if dev mean IoU
improves by >= --min_gain (default 0.005); otherwise it is DISCARDED (and shows up as an unsuccessful attempt).
Add your own trials in TRIALS - the more honest iterations you try, the better the log.
"""
import argparse, json, os, time, itertools, hashlib
from concurrent.futures import ProcessPoolExecutor
import cv2, numpy as np, pandas as pd
from dataclasses import replace
from common import read_img, stem, seg_metrics
from eggseg import Cfg, segment

ALL_CH = ("gray", "L", "a", "b", "sat", "val", "bgdist")

# (id, short name, changes, why we tried it)
TRIALS = [
    ("T0", "Baseline: gray Otsu, largest bright blob", {},
     "Simplest possible classical pipeline - gives a floor to beat."),
    ("T1", "Gaussian blur k=5", dict(blur=5),
     "Sensor noise / egg-shell speckles create ragged Otsu masks; smoothing should help."),
    ("T2", "Morph open(5)+close(7)", dict(morph_open=5, morph_close=7),
     "Remove thin bridges to fingers and fill speckle holes."),
    ("T3", "[enabler] Shape-score selection, both polarities", dict(polarities=("bright", "dark"), selection="score"),
     "Choosing the blob by size is fragile (hand/background can be larger). Choose by 'egg-likeness' instead "
     "(ellipse-fit IoU, convexity, edge strength). Alone it cannot help if no candidate is egg-shaped, so it is an enabling step."),
    ("T4", "3-class multi-Otsu (bright/mid/dark bands)", dict(otsu_classes=3, polarities=("bright", "dark", "mid")),
     "Egg on a hand on a background is a 3-level scene; a 2-class Otsu merges egg+hand so the egg never becomes a candidate."),
    ("T5", "Multi-channel candidates (L,a,b,sat,val,bgdist)", dict(channels=ALL_CH),
     "A single gray channel loses colour contrast (white egg vs pinkish skin differ in saturation / a*)."),
    ("T6", "Skin penalty 0.5", dict(skin_penalty=0.5),
     "Hands are the main distractor; penalise candidates that are mostly skin coloured."),
    ("T7", "+ k-means (k=3) candidates", dict(use_kmeans=True),
     "Otsu splits on one global threshold; clustering in Lab may isolate the egg when the background is multi-modal."),
    ("T8", "GrabCut refinement", dict(grabcut=True),
     "Thresholds give blobby edges; GrabCut snaps the contour to colour edges, mimicking SAM boundaries."),
    ("T9", "Ellipse regularisation", dict(ellipse=True),
     "Eggs are ellipses; replace the mask by the fitted ellipse when they already agree (>0.85 IoU) to remove bumps."),
    ("T10", "Allow up to 3 eggs per image", dict(max_eggs=3),
     "Some images contain several eggs; max_eggs=1 would cap recall there."),
    ("T11", "Quantile threshold sweep (q = .2 .35 .5 .65 .8 .9)", dict(sweep_q=(0.2, 0.35, 0.5, 0.65, 0.8, 0.9)),
     "Dev failure analysis (round 1): all IoU=0 images returned one wrong region and the real egg scored below min_score. "
     "Pale eggs on pale/textured cloth are not isolated by a single Otsu threshold, so they never become candidates. "
     "Thresholding each channel at several quantiles (MSER-like) lets a pale egg appear as a candidate at some level."),
    ("T12", "Hard border rejection (>15% of contour on image border)", dict(max_border_frac=0.15),
     "The wrong regions in the worst cases sat on image borders/corners (carpet corner, bowl edge). Eggs here are photographed inside the frame, so reject border-clipped regions."),
    ("T13", "Centre prior (weight 0.5)", dict(center_w=0.5),
     "Most photos are object-centred; background slivers have centroids far from the centre. Adds a soft preference for central regions."),
    ("T14", "Inside/outside colour-contrast term (weight 0.5)", dict(contrast_w=0.5),
     "Pale eggs on pale cloth have weak gradients (low edge score). Contrast against a thin surrounding ring might be more robust than a gradient on a soft edge."),
]
ENABLERS = {"T3"}       # kept if not worse than best - ENABLER_TOL (documented in the log)
ENABLER_TOL = 0.01
GRID = dict(blur=[3, 5, 9], morph_close=[5, 9, 15], min_score=[0.5, 0.65, 0.8], max_eggs=[1, 3])


def _work(args):
    img_path, gt_path, cfg = args
    t = time.time()
    pred = segment(read_img(img_path), cfg)
    dt = time.time() - t
    gt = cv2.imread(gt_path, 0) > 127
    if pred.shape != gt.shape:
        pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
    m = seg_metrics(pred, gt)
    m["time"] = dt
    return m


CACHE_PATH = "results/exp_cache.json"
CACHE, CACHE_TAG = {}, ""


def _key(cfg):
    return CACHE_TAG + json.dumps(cfg.to_dict(), sort_keys=True, default=list)


def evaluate(items, cfg, pool):
    """Evaluate one config on the dev items. Results are checkpointed to results/exp_cache.json after every
    config, so if the machine crashes/restarts, re-running the script resumes instead of starting over."""
    k = _key(cfg)
    if k in CACHE:
        print("   (cached)", end="")
        return CACHE[k]
    rows = list(pool.map(_work, [(i, g, cfg) for i, g in items], chunksize=2))
    df = pd.DataFrame(rows)
    r = dict(miou=float(df.iou.mean()), median_iou=float(df.iou.median()), dice=float(df.dice.mean()), bf1=float(df.boundary_f1.mean()),
             frac_iou_gt_0_8=float((df.iou > 0.8).mean()), sec_per_img=float(df.time.mean()))
    CACHE[k] = r
    os.makedirs("results", exist_ok=True)
    json.dump(CACHE, open(CACHE_PATH + ".tmp", "w")); os.replace(CACHE_PATH + ".tmp", CACHE_PATH)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--gt", default="results/sam_gt/masks")
    ap.add_argument("--split", default="results/split.json")
    ap.add_argument("--bad", default="results/sam_gt/bad_images.txt")
    ap.add_argument("--workers", type=int, default=max(1, min(3, (os.cpu_count() or 2) // 2)),
                    help="parallel processes (default: gentle - at most 3). Lower it if your laptop gets hot or restarts")
    ap.add_argument("--fresh", action="store_true", help="ignore the checkpoint cache and recompute everything")
    ap.add_argument("--min_gain", type=float, default=0.005)
    ap.add_argument("--skip_grid", action="store_true")
    a = ap.parse_args()

    sp = json.load(open(a.split))
    bad = set(open(a.bad).read().split()) if os.path.exists(a.bad) else set()
    items = []
    for rel in sp["dev"]:                                  # DEV ONLY
        s = stem(rel)
        g = os.path.join(a.gt, s + ".png")
        if s in bad or not os.path.exists(g): continue
        items.append((os.path.join(a.images, rel), g))
    print(f"dev images used: {len(items)}")
    global CACHE, CACHE_TAG
    CACHE_TAG = f"{len(items)}|{hashlib.md5(chr(10).join(i for i, _ in items).encode()).hexdigest()[:10]}|"      # cache is only valid for this exact dev set
    CACHE = {} if a.fresh or not os.path.exists(CACHE_PATH) else json.load(open(CACHE_PATH))
    print(f"checkpoint: {len([k for k in CACHE if k.startswith(CACHE_TAG)])} configs already computed" if CACHE else "checkpoint: none (fresh start)")

    log, best_cfg, best = [], Cfg(), None
    with ProcessPoolExecutor(a.workers) as pool:
        for tid, name, change, why in TRIALS:
            cand = replace(best_cfg, **change)
            r = evaluate(items, cand, pool)
            if best is None:
                kept, best = True, r
            else:
                kept = r["miou"] >= best["miou"] + a.min_gain
                if not kept and tid in ENABLERS and r["miou"] >= best["miou"] - ENABLER_TOL:
                    kept, why = True, why + " [kept as enabler: dev mIoU within tolerance of best]"
                if kept: best = r
            if kept: best_cfg = cand
            log.append(dict(trial=tid, change=name, params=json.dumps(change, default=list), why=why,
                            **{k: round(v, 4) for k, v in r.items()}, decision="KEPT" if kept else "DISCARDED"))
            print(f"{tid} {name:55s} mIoU={r['miou']:.4f} -> {'KEPT' if kept else 'DISCARDED'}")

        # second pass: a change discarded early may only help once enabling changes (e.g. score selection) are in
        for tid, name, change, why in TRIALS:
            if any(l["trial"] == tid and l["decision"] == "DISCARDED" for l in log) and tid != "T0":
                cand = replace(best_cfg, **change)
                if cand == best_cfg: continue
                r = evaluate(items, cand, pool)
                kept = r["miou"] >= best["miou"] + a.min_gain
                if kept: best_cfg, best = cand, r
                log.append(dict(trial=tid + " (retry)", change=name, params=json.dumps(change, default=list),
                                why="Re-test on top of the final config of pass 1: " + why,
                                **{k: round(v, 4) for k, v in r.items()}, decision="KEPT" if kept else "DISCARDED"))
                print(f"{tid} retry {name:48s} mIoU={r['miou']:.4f} -> {'KEPT' if kept else 'DISCARDED'}")

        if not a.skip_grid:                                # small dev-only hyper-parameter search
            keys = list(GRID)
            res = []
            for vals in itertools.product(*GRID.values()):
                ch = dict(zip(keys, vals))
                r = evaluate(items, replace(best_cfg, **ch), pool)
                res.append((r["miou"], ch, r))
            res.sort(key=lambda t: -t[0])
            top = res[0]
            kept = top[0] >= best["miou"] + a.min_gain
            if kept: best_cfg, best = replace(best_cfg, **top[1]), top[2]
            log.append(dict(trial="T15", change=f"Grid search over {len(res)} combos (dev only)",
                            params=json.dumps(top[1]), why="Tune the remaining hyper-parameters jointly.",
                            **{k: round(v, 4) for k, v in top[2].items()}, decision="KEPT" if kept else "DISCARDED"))
            pd.DataFrame([dict(miou=round(m, 4), **c) for m, c, _ in res]).to_csv("results/grid_search.csv", index=False)
            print("T15 grid best", top[1], round(top[0], 4), "KEPT" if kept else "DISCARDED")

    os.makedirs("results", exist_ok=True)
    df = pd.DataFrame(log)
    df.to_csv("results/experiment_log.csv", index=False)
    json.dump(best_cfg.to_dict(), open("results/final_config.json", "w"), indent=1, default=list)
    print("final cfg saved. Final dev mIoU:", round(best["miou"], 4))


if __name__ == "__main__":
    main()
