"""Task 3: run final config on the EVAL split once, compute metrics vs SAM pseudo-GT, make plots."""
import argparse, json, os
import cv2, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import read_img, stem, seg_metrics, overlay
from eggseg import Cfg, segment, segment_instances


def load_cfg(path):
    d = json.load(open(path))
    for k, val in list(d.items()):                       # JSON lists -> tuples (also nested, e.g. ignore boxes)
        if isinstance(val, list):
            d[k] = tuple(tuple(x) if isinstance(x, list) else x for x in val)
    known = Cfg.__dataclass_fields__
    return Cfg(**{k: val for k, val in d.items() if k in known})   # tolerate configs saved by older versions


def classify_failure(r):
    if r.iou >= 0.8: return "ok"
    if r.iou < 0.1: return "wrong object / miss"
    if r.recall >= 0.85 and r.precision < 0.8: return "over-segmentation (leaks into hand/bg)"
    if r.precision >= 0.85 and r.recall < 0.8: return "under-segmentation (part of egg missed)"
    return "mixed boundary error"


def error_map(pred, gt, bgr):
    out = bgr.copy()
    fp, fn = (pred & ~gt), (~pred & gt)
    out[fp] = (0.4 * out[fp] + 0.6 * np.array([0, 0, 255])).astype(np.uint8)   # red = false positive
    out[fn] = (0.4 * out[fn] + 0.6 * np.array([255, 0, 0])).astype(np.uint8)   # blue = false negative
    return out


def panel(rows, df, imgs, gts, preds, title, path):
    n = len(rows)
    fig, ax = plt.subplots(n, 4, figsize=(12, 3 * n))
    ax = np.atleast_2d(ax)
    for r, (i, row) in enumerate(rows.iterrows()):
        bgr, g, p = imgs[row.image], gts[row.image], preds[row.image]
        for c, (im, t) in enumerate([(bgr, "image"), (overlay(bgr, g, (0, 255, 0)), "SAM GT"),
                                      (overlay(bgr, p, (0, 0, 255)), "ours"), (error_map(p, g, bgr), "error: red=FP blue=FN")]):
            ax[r, c].imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)); ax[r, c].axis("off")
            ax[r, c].set_title(f"{t}" + (f"  IoU={row.iou:.2f}" if c == 2 else ""), fontsize=8)
    fig.suptitle(title); fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--gt", default="results/sam_gt/masks")
    ap.add_argument("--split", default="results/split.json")
    ap.add_argument("--cfg", default="results/final_config.json")
    ap.add_argument("--bad", default="results/sam_gt/bad_images.txt")
    ap.add_argument("--which", default="eval", choices=["eval", "dev"])
    ap.add_argument("--out", default="results/eval")
    ap.add_argument("--k", type=int, default=6)
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "pred_masks"), exist_ok=True)
    cfg = load_cfg(a.cfg)
    bad = set(open(a.bad).read().split()) if os.path.exists(a.bad) else set()
    rels = json.load(open(a.split))[a.which]

    rows, imgs, gts, preds = [], {}, {}, {}
    for rel in rels:
        s = stem(rel)
        gp = os.path.join(a.gt, s + ".png")
        if s in bad or not os.path.exists(gp): continue
        bgr = read_img(os.path.join(a.images, rel))
        gt = cv2.imread(gp, 0) > 127
        inst = segment_instances(bgr, cfg)
        pred = (np.clip(np.sum(inst, 0), 0, 1) > 0) if inst else np.zeros(gt.shape, bool)
        cv2.imwrite(os.path.join(a.out, "pred_masks", s + ".png"), pred.astype(np.uint8) * 255)
        m = seg_metrics(pred, gt)
        gl = cv2.imread(os.path.join(a.gt, "..", "labels", s + ".png"), 0)
        n_gt = int(gl.max()) if gl is not None else -1
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        m.update(image=s, n_pred=len(inst), n_gt=n_gt,
                 egg_sat=float(hsv[..., 1][gt].mean()) if gt.any() else np.nan,
                 egg_val=float(hsv[..., 2][gt].mean()) if gt.any() else np.nan)
        rows.append(m); imgs[s], gts[s], preds[s] = bgr, gt, pred
    df = pd.DataFrame(rows)
    df["failure"] = df.apply(classify_failure, axis=1)
    df.to_csv(os.path.join(a.out, "per_image_metrics.csv"), index=False)

    cols = ["iou", "dice", "precision", "recall", "boundary_f1", "hd95"]
    summ = df[cols].agg(["mean", "std", "median", "min"]).T.round(4)
    summ["p10"] = df[cols].quantile(0.1).round(4)
    summ.index.name = "metric"
    summ.to_csv(os.path.join(a.out, "summary.csv"))
    fail = df.failure.value_counts().rename_axis("failure mode").reset_index(name="count")
    fail.to_csv(os.path.join(a.out, "failure_modes.csv"), index=False)
    print(summ, "\n", fail, f"\ncount agreement: {(df.n_pred == df.n_gt).mean():.3f}  | IoU>0.8: {(df.iou > 0.8).mean():.3f}")

    # ---- plots
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    ax[0, 0].hist(df.iou, bins=20, color="C0"); ax[0, 0].set_title("IoU distribution"); ax[0, 0].set_xlabel("IoU")
    ax[0, 1].boxplot([df[c] for c in ["iou", "dice", "precision", "recall", "boundary_f1"]])
    ax[0, 1].set_xticklabels(["IoU", "Dice", "Prec", "Rec", "BF1"]); ax[0, 1].set_title("Metric spread")
    sc = ax[0, 2].scatter(df.recall, df.precision, c=df.iou, cmap="viridis", s=18)
    ax[0, 2].set_xlabel("recall"); ax[0, 2].set_ylabel("precision"); ax[0, 2].set_title("Precision vs recall (colour=IoU)")
    plt.colorbar(sc, ax=ax[0, 2])
    ax[1, 0].scatter(df.gt_area_frac, df.iou, s=14); ax[1, 0].set_xlabel("egg area / image area"); ax[1, 0].set_ylabel("IoU")
    ax[1, 0].set_title("Does egg size matter?")
    ax[1, 1].scatter(df.egg_sat, df.iou, s=14, c=df.egg_val, cmap="copper_r"); ax[1, 1].set_xlabel("mean egg saturation (HSV-S)")
    ax[1, 1].set_ylabel("IoU"); ax[1, 1].set_title("Egg colour vs IoU (colour = brightness)")
    ax[1, 2].barh(fail["failure mode"], fail["count"]); ax[1, 2].set_title("Failure modes (IoU<0.8)")
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "quantitative_plots.png"), dpi=110); plt.close(fig)

    worst = df.sort_values("iou").head(a.k); best = df.sort_values("iou").tail(3)
    panel(worst, df, imgs, gts, preds, "Worst segmentations", os.path.join(a.out, "worst_cases.png"))
    panel(best, df, imgs, gts, preds, "Best segmentations", os.path.join(a.out, "best_cases.png"))


if __name__ == "__main__":
    main()
