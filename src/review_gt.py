"""Labelled contact sheets for QA of the SAM pseudo-ground-truth.
Each thumbnail shows the image number (from the file name) and the number of GT instances (xN).
Border colour: RED = suspicious (empty mask or mask covers >45% of the image), YELLOW = more than one instance, none = single instance.
Writes results/sam_gt/review/sheet_XX.jpg, review_index.csv and auto_flag.txt (the RED ones, as candidates for bad_images.txt)."""
import argparse, os, re, csv
import cv2, numpy as np
from common import list_images, stem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--gt", default="results/sam_gt")
    ap.add_argument("--per_sheet", type=int, default=24)
    a = ap.parse_args()
    out = os.path.join(a.gt, "review"); os.makedirs(out, exist_ok=True)
    rows, thumbs = [], []
    for k, p in enumerate(list_images(a.images)):
        s = stem(p)
        ov = cv2.imread(os.path.join(a.gt, "overlays", s + ".jpg"))
        lab = cv2.imread(os.path.join(a.gt, "labels", s + ".png"), 0)
        if ov is None or lab is None:
            continue
        n = int(lab.max()); area = float((lab > 0).mean())
        flag = "red" if (n == 0 or area > 0.45) else ("yellow" if n > 1 else "")
        m = re.search(r"_(\d+)_(?:jpg|jpeg|png)", s)
        short = m.group(1) if m else str(k)
        t = cv2.resize(ov, (200, 200))
        if flag:
            cv2.rectangle(t, (0, 0), (199, 199), (0, 0, 255) if flag == "red" else (0, 220, 255), 4)
        cv2.rectangle(t, (0, 0), (92, 24), (0, 0, 0), -1)
        cv2.putText(t, f"#{short} x{n}", (3, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        thumbs.append(t)
        rows.append(dict(short_id=short, stem=s, n_instances=n, gt_area_frac=round(area, 4), flag=flag))
    for i in range(0, len(thumbs), a.per_sheet):
        chunk = thumbs[i:i + a.per_sheet]
        chunk += [np.zeros_like(thumbs[0])] * (-len(chunk) % 6)
        cv2.imwrite(os.path.join(out, f"sheet_{i // a.per_sheet:02d}.jpg"),
                    np.vstack([np.hstack(chunk[j:j + 6]) for j in range(0, len(chunk), 6)]))
    with open(os.path.join(out, "review_index.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    red = [r["stem"] for r in rows if r["flag"] == "red"]
    open(os.path.join(a.gt, "auto_flag.txt"), "w").write("\n".join(red) + ("\n" if red else ""))
    print(f"{len(rows)} images | red (suspicious): {len(red)} | yellow (multi-instance): {sum(r['flag']=='yellow' for r in rows)}")
    print("sheets in", out)


if __name__ == "__main__":
    main()
