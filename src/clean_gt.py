"""Post-filter SAM pseudo-GT: drop instances clipped by the image border (corner/edge slivers that
passed the ellipse filter) and tiny specks, but only when another instance remains in that image.
Rewrites results/sam_gt/{masks,labels,overlays} in place and prints every drop so you can check it."""
import argparse, os
import cv2, numpy as np
from common import list_images, stem, read_img, overlay


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--gt", default="results/sam_gt")
    ap.add_argument("--border_frac", type=float, default=0.15, help="drop if > this fraction of the contour lies on the image border")
    ap.add_argument("--min_area", type=float, default=0.004, help="drop if smaller than this fraction of the image")
    a = ap.parse_args()
    n_drop = 0
    for p in list_images(a.images):
        s = stem(p)
        lab = cv2.imread(os.path.join(a.gt, "labels", s + ".png"), 0)
        if lab is None or lab.max() == 0:
            continue
        h, w = lab.shape
        ids = [i for i in range(1, int(lab.max()) + 1) if (lab == i).any()]
        ids.sort(key=lambda i: (lab == i).sum())              # smallest first
        drop = []
        for i in ids:
            m = (lab == i).astype(np.uint8)
            cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            pts = np.concatenate([c.reshape(-1, 2) for c in cnts])
            on_b = np.mean((pts[:, 0] <= 1) | (pts[:, 1] <= 1) | (pts[:, 0] >= w - 2) | (pts[:, 1] >= h - 2))
            if (on_b > a.border_frac or m.mean() < a.min_area) and len(ids) - len(drop) > 1:
                drop.append(i)
                print(f"drop instance {i} of {len(ids)} in {s}  (border={on_b:.2f}, area={m.mean():.3f})")
        if not drop:
            continue
        n_drop += len(drop)
        keep = [i for i in ids if i not in drop]
        new = np.zeros_like(lab)
        for k, i in enumerate(keep, 1):
            new[lab == i] = k
        union = (new > 0).astype(np.uint8)
        cv2.imwrite(os.path.join(a.gt, "labels", s + ".png"), new)
        cv2.imwrite(os.path.join(a.gt, "masks", s + ".png"), union * 255)
        cv2.imwrite(os.path.join(a.gt, "overlays", s + ".jpg"), overlay(read_img(p), union))
    print(f"dropped {n_drop} instances in total")


if __name__ == "__main__":
    main()
