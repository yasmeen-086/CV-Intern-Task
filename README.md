# Egg Segmentation and Individual Egg Tracking

Pipeline: **SAM pseudo-GT -> leakage-aware 4:1 split -> classical segmenter + logged experiments -> evaluation vs SAM -> video tracking.**

The dataset and SAM checkpoint files are not included in this repository. Download the dataset and checkpoint separately, then follow the setup instructions below. SAM masks are model-generated reference masks, not manually verified ground truth.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# SAM checkpoint (vit_h ~2.4GB best quality; vit_b ~375MB is fine on CPU/small GPU)
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth      # or sam_vit_b_01ec64.pth
```
Download the Roboflow dataset (images only needed) and put all images under `data/images/` 
If you exported **COCO format**, keep `_annotations.coco.json` too (used for SAM box prompts).
**Run every command from the repo root.**

## Run
```bash
# Task 1 - SAM masks (+ contact sheets for manual QA in results/sam_gt/)
python src/task1_sam.py --images data/images --ckpt sam_vit_h_4b8939.pth --model_type vit_h \
       [--boxes_coco data/_annotations.coco.json] [--device cpu]
#   -> open results/sam_gt/contact_sheet_*.jpg, write bad image stems (one per line) to results/sam_gt/bad_images.txt

# Task 2 - split + experiments (DEV ONLY)
python src/split.py --images data/images            # results/split.json
python src/experiments.py --images data/images      # results/experiment_log.csv, final_config.json, grid_search.csv
#   add --skip_grid for a quick run; --workers N to control parallelism

# Task 3 - evaluate once on the EVAL split
python src/evaluate.py --images data/images         # results/eval/*  (metrics csv, plots, worst/best panels)
python src/evaluate.py --images data/images --which dev --out results/eval_dev   # dev numbers for the overfitting check

# Task 4 - video
python src/track_video.py --video data/video.mp4 --cfg results_after_cleaning_v1/final_config.json \
       --out results/tracking --max_eggs 12
# You may use --url "<YouTube URL>" instead of --video to download the input.
#   tune if needed: --min_score 0.5 --max_area 0.15 --proc_side 960 --no_split

```

## Files
| file | purpose |
|---|---|
| `src/task1_sam.py` | SAM segmentation (box prompts if annotations, else auto-mask + egg-likeness filter) |
| `src/split.py` | group-level 4:1 split; groups = same Roboflow source name + dHash near-duplicates |
| `src/eggseg.py` | the classical algorithm; every option lives in `Cfg` |
| `src/experiments.py` | trial list -> experiment log (kept/discarded automatically), dev-only grid search |
| `src/evaluate.py` | IoU/Dice/precision/recall/boundary-F1/HD95, plots, failure taxonomy, worst-case panels |
| `src/track_video.py` | per-frame segmentation + watershed splitting + Hungarian IoU tracker |
| `Egg_Segmentation_and_Tracking_Report.docx` | internship assignment report |
| `results_after_cleaning_v1/` | latest cleaned-reference experiment, evaluation, split, and tracking artifacts |

## Notes
- SAM masks are *pseudo*-ground-truth: metrics measure agreement with SAM, not with reality.
- AI-tool usage is disclosed in the report.
- A sample tracking output is included at `results_after_cleaning_v1/tracking_final2/tracked.mp4`.
- The required public Google Drive video link has not been added yet; add it here after uploading the video to Drive.
