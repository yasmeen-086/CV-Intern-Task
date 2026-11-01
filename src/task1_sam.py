"""Task 1: pseudo-ground-truth with SAM.

Mode A (preferred, if the Roboflow export has boxes): --boxes_coco _annotations.coco.json  -> SAM box prompt per egg.
Mode B (default): SAM automatic mask generator, then keep masks that look like eggs
        (SAM's own quality scores + ellipse fit + convexity + area range, skin-coloured masks rejected).
Outputs per image: <out>/masks/<stem>.png (uint8 union 0/255), <out>/labels/<stem>.png (instance ids),
                   <out>/overlays/<stem>.jpg, and a contact sheet for FAST MANUAL QA.
"""
import argparse, json, os
import cv2, numpy as np
from tqdm import tqdm
from common import list_images, read_img, stem, skin_mask, ellipse_fit_iou, mask_iou, overlay


def load_sam(ckpt, model_type, device):
    import torch
    from segment_anything import sam_model_registry
    sam = sam_model_registry[model_type](checkpoint=ckpt).to(device)
    return sam


def egg_like(m, bgr, min_a=0.01, max_a=0.75):
    mu = m.astype(np.uint8)
    af = mu.mean()
    if af < min_a or af > max_a: return 0.0
    cnts, _ = cv2.findContours(mu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(cnts, key=cv2.contourArea)
    eiou, _ = ellipse_fit_iou(mu, cnt)
    sol = mu.sum() / max(1.0, cv2.contourArea(cv2.convexHull(cnt)))
    skin = skin_mask(bgr)[m].mean()
    if skin > 0.5: return 0.0
    return float(eiou * sol)


def pick_auto(masks, bgr, max_eggs=6, thr=0.85):
    scored = []
    for md in masks:
        s = egg_like(md["segmentation"], bgr)
        if s > 0:
            scored.append((s * md["predicted_iou"] * md["stability_score"], s, md["segmentation"]))
    scored.sort(key=lambda t: -t[0])
    keep = []
    for q, s, m in scored:
        if len(keep) >= max_eggs: break
        if s < thr and keep: break              # always keep the best, others must be clearly egg-like
        if any(mask_iou(m, k) > 0.5 for k in keep): continue
        keep.append(m)
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", default="results/sam_gt")
    ap.add_argument("--ckpt", required=True, help="e.g. sam_vit_h_4b8939.pth")
    ap.add_argument("--model_type", default="vit_h", choices=["vit_h", "vit_l", "vit_b"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--boxes_coco", default=None)
    ap.add_argument("--max_eggs", type=int, default=6)
    a = ap.parse_args()
    from segment_anything import SamPredictor, SamAutomaticMaskGenerator
    sam = load_sam(a.ckpt, a.model_type, a.device)
    pred = SamPredictor(sam)
    amg = SamAutomaticMaskGenerator(sam, points_per_side=24, pred_iou_thresh=0.86,
                                    stability_score_thresh=0.90, min_mask_region_area=200)
    for d in ("masks", "labels", "overlays"):
        os.makedirs(os.path.join(a.out, d), exist_ok=True)

    boxes = {}
    if a.boxes_coco:
        coco = json.load(open(a.boxes_coco))
        idf = {im["id"]: im["file_name"] for im in coco["images"]}
        for an in coco["annotations"]:
            x, y, w, h = an["bbox"]
            boxes.setdefault(idf[an["image_id"]], []).append([x, y, x + w, y + h])

    thumbs = []
    for p in tqdm(list_images(a.images)):
        bgr = read_img(p); rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        fn = os.path.basename(p)
        if fn in boxes:
            pred.set_image(rgb); inst = []
            for b in boxes[fn]:
                m, sc, _ = pred.predict(box=np.array(b), multimask_output=False)
                inst.append(m[0])
        else:
            inst = pick_auto(amg.generate(rgb), bgr, a.max_eggs)
        lab = np.zeros(bgr.shape[:2], np.uint8)
        for i, m in enumerate(inst, 1): lab[m] = i
        union = (lab > 0).astype(np.uint8)
        s = stem(p)
        cv2.imwrite(os.path.join(a.out, "masks", s + ".png"), union * 255)
        cv2.imwrite(os.path.join(a.out, "labels", s + ".png"), lab)
        ov = overlay(bgr, union)
        cv2.imwrite(os.path.join(a.out, "overlays", s + ".jpg"), ov)
        thumbs.append(cv2.resize(ov, (160, 160)))
    # contact sheets of 8x6 for quick manual QA of the pseudo-GT
    per = 48
    for k in range(0, len(thumbs), per):
        chunk = thumbs[k:k + per]
        rows = [np.hstack(chunk[i:i + 8] + [np.zeros_like(chunk[0])] * (8 - len(chunk[i:i + 8]))) for i in range(0, len(chunk), 8)]
        cv2.imwrite(os.path.join(a.out, f"contact_sheet_{k // per:02d}.jpg"), np.vstack(rows))
    print("done. REVIEW the contact sheets; list any bad images in results/sam_gt/bad_images.txt (they are excluded later).")


if __name__ == "__main__":
    main()
