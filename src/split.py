"""4:1 dev/eval split with leakage control.

Leakage sources in Roboflow exports:
  1. the SAME source photo exported several times as augmentations (name pattern  <name>_jpg.rf.<hash>.jpg)
  2. near-duplicate photos (burst shots of the same egg/hand) with different file names.
We therefore (a) group by the source name, (b) merge groups whose images have perceptual (dHash)
Hamming distance <= --hamming, and (c) split at GROUP level so a group never straddles dev/eval.
"""
import argparse, json, os, random, re
import cv2
import numpy as np
from common import list_images, read_img


def dhash(bgr, size=8):
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (size + 1, size), interpolation=cv2.INTER_AREA)
    return (g[:, 1:] > g[:, :-1]).flatten()


def source_key(path):
    name = os.path.basename(path)
    name = re.split(r"\.rf\.", name)[0]            # roboflow augmentation suffix
    name = re.sub(r"_(jpg|jpeg|png)$", "", name, flags=re.I)
    return name


class DSU:
    def __init__(self, n): self.p = list(range(n))
    def f(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]; x = self.p[x]
        return x
    def u(self, a, b): self.p[self.f(a)] = self.f(b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", default="results/split.json")
    ap.add_argument("--ratio", type=float, default=0.8, help="dev fraction (4:1 => 0.8)")
    ap.add_argument("--hamming", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    files = list_images(a.images)
    n = len(files)
    hashes = np.stack([dhash(read_img(f)) for f in files])
    dsu = DSU(n)
    keys = {}
    for i, f in enumerate(files):                      # (a) same source name
        k = source_key(f)
        if k in keys: dsu.u(i, keys[k])
        else: keys[k] = i
    for i in range(n):                                 # (b) near duplicates
        d = (hashes[i] != hashes[i + 1:]).sum(1)
        for j in np.where(d <= a.hamming)[0]:
            dsu.u(i, i + 1 + j)
    groups = {}
    for i in range(n):
        groups.setdefault(dsu.f(i), []).append(i)
    glist = list(groups.values())
    biggest = max(len(g) for g in glist)
    if biggest > 0.1 * n:
        print(f"WARNING: largest leakage group has {biggest}/{n} images (transitive near-duplicate chaining?). "
              f"Lower --hamming (try 4-6) or inspect the images.")
    random.Random(a.seed).shuffle(glist)
    glist.sort(key=len, reverse=True)                  # big groups first -> easier to hit exact ratio
    target_dev = round(a.ratio * n)
    dev, ev = [], []
    for g in glist:                                    # fill dev up to target, rest -> eval
        (dev if len(dev) + len(g) <= target_dev else ev).extend(g)
    # sanity: minimum cross-split hash distance
    dh, eh = hashes[dev], hashes[ev]
    min_cross = int(min((dh != e).sum(1).min() for e in eh)) if len(ev) and len(dev) else -1
    res = dict(dev=[os.path.relpath(files[i], a.images) for i in sorted(dev)],
               eval=[os.path.relpath(files[i], a.images) for i in sorted(ev)],
               n_groups=len(glist), min_cross_split_hamming=min_cross, seed=a.seed,
               hamming_threshold=a.hamming, images_root=a.images)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"{n} images -> dev {len(dev)} / eval {len(ev)} ({len(dev)/max(1,len(ev)):.2f}:1), "
          f"{len(glist)} groups, min cross-split dHash distance = {min_cross} (threshold {a.hamming})")


if __name__ == "__main__":
    main()
