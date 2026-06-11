# Bumblebee Photobooth ArUco Detection Handoff

This file summarizes the current chat and technical state so the project can be moved to a PC for training a model to find/rectify the ArUco paper.

## Project Locations

- Local workspace: `/home/wlab/Desktop/bumblebee_photobooth`
- Raspberry Pi project copy: `/home/pi/Desktop`
- SSH target used during diagnostics: `pi@cc.local`
- Main capture script: `bb_pp_resize.py`
- Active ArUco config: `aruco_config.json`
- Active ArUco params: `aruco_params_current.json`

## User Problem

The ArUco scanner is inconsistent:

- Some codes phase in and out of detection while stationary.
- Some clearly visible codes are never detected.
- Some frames show incorrect IDs.
- The current test bee should decode as ArUco ID `67`.
- Codes with many black bits are especially problematic.

## Implemented Script/Config Changes

### Manual ID Entry

Added a manual ArUco fallback to `bb_pp_resize.py`:

- Press `m` in live preview or capture review.
- Type the numeric ArUco ID.
- `Backspace` edits, `Enter` saves, `Esc` cancels.
- Manual ID is used in the image label and `_arucoid-...jpg` filename.
- Config flag: `"manual_entry_enabled": true`.

### Dictionary Change

Confirmed locally that:

- `CUSTOM_4X4_4000_FROM_DICT_4X4_1000.npz`
- first 1000 entries exactly match OpenCV `DICT_4X4_1000`
- marker size is `4`
- `maxCorrectionBits` is `0`

Changed active config to use the built-in OpenCV dictionary:

```json
"dictionary": {
  "mode": "default",
  "marker_size_bits": 4,
  "predefined_dict_count": 1000
}
```

The active config now points at:

```json
"active_params_file": "aruco_params_current.json"
```

### Detection Parameter Changes

Current stable params include:

```json
{
  "adaptiveThreshConstant": 7,
  "adaptiveThreshWinSizeMin": 5,
  "adaptiveThreshWinSizeMax": 41,
  "adaptiveThreshWinSizeStep": 4,
  "cornerRefinementMaxIterations": 30,
  "cornerRefinementMinAccuracy": 0.1,
  "cornerRefinementWinSize": 5,
  "errorCorrectionRate": 0.0,
  "markerBorderBits": 1,
  "maxErroneousBitsInBorderRate": 0.3,
  "maxMarkerPerimeterRate": 4.0,
  "minMarkerPerimeterRate": 0.03,
  "minOtsuStdDev": 1.0,
  "perspectiveRemoveIgnoredMarginPerCell": 0.25,
  "perspectiveRemovePixelPerCell": 8,
  "polygonalApproxAccuracyRate": 0.03
}
```

Other active settings:

```json
"scan_roi_ratio": 0.35,
"detection_upscale_factor": 2.0,
"preprocess_modes": ["raw", "clahe", "clahe_sharpen"],
"use_detector_params": true
```

### Tuning Diagnostics Added

In `t` tuning mode:

- Magenta outlines show rejected ArUco candidates.
- Green outlines show decoded IDs.
- The panel shows rejected candidate count and detection upscale factor.

Interpretation:

- Magenta around the tag but no green ID means OpenCV found a square but failed to decode.
- No magenta around the tag means contour/quiet-zone/lighting/ROI problem.

## Raspberry Pi Diagnostics

The Pi was accessible over SSH at `pi@cc.local`.

Pi-side facts:

- OpenCV version: `4.6.0`
- `cv2.aruco` is available.
- Picamera2 import works.
- Camera detected as Sony IMX477 / Raspberry Pi HQ camera.
- Project scripts/configs are in `/home/pi/Desktop`.

### Live Diagnostic For Expected ID 67

Ran a temporary script capturing 80 preview frames.

Results:

```text
dictionary=DICT_4X4_1000
params_path=/home/pi/Desktop/aruco_params_current.json
detector_params_enabled=True
scan_roi_ratio=0.35
detection_upscale_factor=2.0
preprocess_modes=["raw", "clahe", "clahe_sharpen"]
frames=80
frames_with_any_id=8
frames_with_expected_67=0
id_counts={106: 3, 110: 1, 145: 1, 147: 1, 653: 1, 816: 1}
rejected_min_mean_max=26/87.86/108
area_ratio_min_mean_max=0.000012/0.001087/0.008420
```

Saved images on Pi:

```text
/home/pi/Desktop/aruco_diagnostics/aruco_diag_last_20250626_090919.jpg
/home/pi/Desktop/aruco_diagnostics/aruco_diag_best_rejected_20250626_090919.jpg
```

### Full-Resolution Still Diagnostic

Captured at `4056x3040`.

Results:

```text
dictionary=DICT_4X4_1000
still_ids=none
still_rejected=134
raw=/home/pi/Desktop/aruco_diagnostics/aruco_still_raw_20250626_091026.jpg
overlay=/home/pi/Desktop/aruco_diagnostics/aruco_still_overlay_20250626_091026.jpg
roi=/home/pi/Desktop/aruco_diagnostics/aruco_still_roi_20250626_091026.jpg
```

Visual diagnosis from the still ROI:

- The paper is visible and in focus.
- The black ArUco border sits directly on black bee hair.
- The black border and black cells are mottled and blend with the bee body.
- There is not a clean white quiet zone around the marker.
- This is especially bad for black-heavy 4x4 codes such as ID `67`.

An offline parameter grid against the saved still produced more wrong IDs, such as `190`, `322`, `816`, and `987`, but did not produce ID `67`. The grid was stopped early after it became clear that more permissive settings mainly increased false positives.

## Current Diagnosis

The main failure is physical/visual, not just detector parameters.

OpenCV ArUco expects a square marker with a black border and inner binary matrix. The black border supports fast marker detection. When the black border sits on black bee hair, the marker boundary disappears, and OpenCV may only see many rejected candidate fragments.

For ID `67`, the expected marker has relatively few white cells, making it difficult to recover from a lost black border or a mottled black region.

Best physical fix:

- Print/apply the marker on a matte white backing.
- Preserve a white quiet zone around the whole ArUco square.
- Minimum quiet zone: about one cell width on all sides.
- Better quiet zone: two cell widths.
- Keep the black ArUco border intact inside the white quiet zone.

## Model Training Recommendation

Yes, a learned model can help, but do not train it to directly classify hundreds of IDs from the whole bee photo.

Recommended pipeline:

1. Train a small model to find the code paper/corners.
2. Use the predicted corners to perspective-warp the code into a square.
3. Decode the 4x4 bits algorithmically.
4. Compare decoded bits against allowed IDs, probably IDs `0-300`.
5. Use manual `m` entry if confidence is low.

### Best Model Type

Use a YOLO pose/keypoint model:

- Class: `aruco_paper`
- Keypoints: 4
  - top-left
  - top-right
  - bottom-right
  - bottom-left

Why pose/keypoint is preferred:

- The model learns where the paper corners are even when the black ArUco border blends into bee hair.
- Once corners are known, classical geometry and bit sampling can decode the ID.
- This needs far less data than training one classifier per ID.

Alternative:

- YOLO segmentation model for the paper outline.
- Use the mask/contour to estimate a quadrilateral.

### Dataset Size

A few hundred real photos may be enough for paper/corner localization if they cover:

- black bees and yellow bees
- good/bad focus
- glare
- tilted labels
- different code IDs
- black-heavy and white-heavy patterns
- varying scale and location

It is probably not enough for end-to-end classification across hundreds of IDs without synthetic augmentation.

### Annotation Plan

For each training image:

- Label one object: the code paper.
- Mark the four paper/marker corners in consistent order.
- Include images where ArUco fails, not just easy successes.
- Split roughly:
  - 80% train
  - 10% validation
  - 10% test

### Training Command Shape

Using Ultralytics YOLO pose, the command will look roughly like:

```bash
yolo pose train model=yolo11n-pose.pt data=aruco_paper_pose.yaml epochs=100 imgsz=640
```

Or with current Ultralytics model naming:

```bash
yolo pose train model=yolo26n-pose.pt data=aruco_paper_pose.yaml epochs=100 imgsz=640
```

Exact model name depends on the installed Ultralytics version on the training PC.

### Deployment Concept

At runtime on the Pi:

1. Capture frame.
2. Run small YOLO model to predict four corners.
3. Warp patch to canonical square.
4. Sample 6x6 grid:
   - outer one-cell black border
   - inner 4x4 data matrix
5. Decode against `DICT_4X4_1000`, restricted to first ~300 IDs.
6. Return ID only if confidence/Hamming distance is safe.

## Useful Source Links

OpenCV ArUco docs:

- https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html

Ultralytics YOLO pose docs:

- https://docs.ultralytics.com/tasks/pose/

Ultralytics pose dataset format:

- https://docs.ultralytics.com/datasets/pose/

Ultralytics training docs:

- https://docs.ultralytics.com/modes/train/

Ultralytics segmentation docs:

- https://docs.ultralytics.com/tasks/segment/

## Practical Next Steps On The PC

1. Copy project images and diagnostics from the Pi/this laptop.
2. Build a dataset folder:

```text
aruco_paper_pose/
  images/
    train/
    val/
    test/
  labels/
    train/
    val/
    test/
  aruco_paper_pose.yaml
```

3. Annotate the paper corners.
4. Train YOLO pose.
5. Validate on held-out images, especially failed ID `67` style images.
6. Add a decoder script that warps from four corners and samples bits.
7. Integrate into `bb_pp_resize.py` as a fallback before manual entry.

