# Egg Segmentation and Individual Egg Tracking

Internship assignment project covering SAM-based segmentation, classical image processing, segmentation evaluation, and video tracking.

**Workflow:** inspect the dataset → generate SAM reference masks → split images into development and evaluation sets → build and tune a classical segmenter → evaluate predictions against SAM masks → segment and track eggs in video.

SAM masks are model-generated reference masks. They are not manually verified ground truth, so the evaluation measures agreement with SAM rather than definitive segmentation accuracy.

## Features

- Generate egg masks using the pretrained Segment Anything Model (SAM).
- Review SAM outputs using contact sheets and filter suspicious masks.
- Split images into development and evaluation sets while grouping likely duplicates.
- Compare classical segmentation experiments using development-set mIoU.
- Evaluate segmentation using IoU, Dice, precision, recall, Boundary F1, HD95, and egg-count agreement.
- Segment video frames and associate detections into tracks with persistent IDs.

## Repository contents

```text
src/
  task1_sam.py       SAM-based mask generation
  review_gt.py       Contact sheets and flags for SAM mask review
  clean_gt.py        Filters selected SAM mask regions
  split.py           Group-aware development/evaluation split
  eggseg.py          Classical image-processing segmentation
  experiments.py     Development-set experiments and parameter search
  evaluate.py        Segmentation metrics, failure categories, and plots
  track_video.py     Video segmentation and egg tracking
results_after_cleaning_v1/
  final_config.json  Saved segmentation configuration
  split.json         Saved image split
  experiment_log.csv Experiment history
  grid_search.csv    Grid-search results
  eval/              Evaluation results and visualizations
  eval_dev/          Development-set results and visualizations
  sam_gt/            SAM review artifacts
  tracking_final2/   Sample tracking output and track data
Egg_Segmentation_and_Tracking_Report.docx
requirements.txt
```

## Dataset and model files

The dataset, input video, and SAM checkpoint weights are not included in this repository. Download the dataset and put its images under `data/egg/`. The scripts search recursively within that folder.

Download a SAM checkpoint into the repository root. The larger ViT-H checkpoint is shown below; ViT-B is also supported.

```bash
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

If your dataset is exported in COCO format, keep its `_annotations.coco.json` file with the dataset. The COCO file can provide box prompts to SAM.

## Installation

Run these commands from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Main dependencies include NumPy, OpenCV, SciPy, Matplotlib, pandas, tqdm, PyTorch, Segment Anything, and yt-dlp. SAM checkpoint size and runtime depend on the model and hardware. Use `--device cpu` for CPU inference; it will generally be slower than a supported GPU.

## Task 1: Segmentation using SAM

Generate SAM masks and overlays using automatic mask generation:

```bash
python src/task1_sam.py \
  --images data/egg \
  --ckpt sam_vit_h_4b8939.pth \
  --model_type vit_h \
  --device cpu
```

If you have a COCO annotations file and want to use its bounding boxes as SAM prompts, run:

```bash
python src/task1_sam.py \
  --images data/egg \
  --ckpt sam_vit_h_4b8939.pth \
  --model_type vit_h \
  --device cpu \
  --boxes_coco data/egg/_annotations.coco.json
```

The script writes masks, instance labels, overlays, and contact sheets under `results/sam_gt/`. Review the contact sheets. The optional review script creates additional sheets and flags suspicious cases:

```bash
python src/review_gt.py --images data/egg --gt results/sam_gt
```

If you manually identify bad image stems, list them one per line in `results/sam_gt/bad_images.txt`. The split, experiment, and evaluation scripts can exclude those images.

## Task 2: Classical image processing and experiments

Create a leakage-aware 4:1 development/evaluation split:

```bash
python src/split.py \
  --images data/egg \
  --out results/split.json \
  --ratio 0.8 \
  --hamming 5 \
  --seed 42
```

The split groups images with matching Roboflow source names and merges near-duplicates using difference-hash distance before assigning groups to splits.

Run the development-set experiments and parameter grid search:

```bash
python src/experiments.py --images data/egg --workers 2
```

The command used for the longer run on macOS was:

```bash
caffeinate -i python src/experiments.py --images data/egg --workers 2
```

`caffeinate -i` keeps a Mac awake during the run. The experiment script uses the split and SAM masks under `results/` by default. It writes the experiment log, selected configuration, and grid-search results to `results/`.

To skip the grid search for a quicker run:

```bash
python src/experiments.py --images data/egg --workers 2 --skip_grid
```

## Task 3: Evaluation and analysis

Run evaluation on the held-out evaluation split:

```bash
python src/evaluate.py --images data/egg
```

This is the evaluation command used for the reported evaluation-set results.

Run development-set evaluation separately:

```bash
python src/evaluate.py --images data/egg --which dev --out results/eval_dev
```

This is the development-set evaluation command used for the reported development results.

The evaluation script reports IoU, Dice, precision, recall, Boundary F1, HD95, egg-count agreement, and failure categories. It also writes per-image metrics, plots, and best/worst example panels. Metrics compare the classical predictions with the SAM-generated reference masks.

Do not use the evaluation split to tune parameters. Use the development split for experiments, then run evaluation on the held-out split.

## Task 4: Video tracking

Put the assignment video at `data/video.mp4`, then run:

```bash
python src/track_video.py \
  --video data/video.mp4 \
  --cfg results_after_cleaning_v1/final_config.json \
  --out results/tracking \
  --max_eggs 12
```

To let the script download a YouTube video, provide its URL:

```bash
python src/track_video.py \
  --url "<YouTube URL>" \
  --video data/video.mp4 \
  --cfg results_after_cleaning_v1/final_config.json \
  --out results/tracking \
  --max_eggs 12
```

The tracker segments each frame, optionally splits touching regions, then associates detections using Hungarian matching with mask IoU and centroid distance. It uses velocity-based prediction and retains tracks through a configurable number of missed frames. The default `--max_missed` is 15; this can help preserve IDs through short occlusions, but it does not guarantee correct identity through every occlusion.

Useful video options include `--min_score`, `--max_area`, `--proc_side`, `--max_missed`, `--assoc_gate`, and `--no_split`. Run `python src/track_video.py --help` to see all available options.



o: 
