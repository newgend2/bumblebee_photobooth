# ArUco Paper Segmentation Dataset

This is the preferred v2 dataset for bent or slightly curved ArUco tags. The
model learns the printed marker outline only; IDs are still decoded
algorithmically after rectification.

## Label Target

Create one polygon object per recoverable marker:

- Class: `aruco_paper`
- Points: 4-8 ordered points around the printed ArUco boundary
- Use 4 points for flat tags
- Add points only where a marker edge visibly bends
- Do not include cut paper, inconsistent white slivers, bee body, or hair

Leave severe folds or occlusions unlabeled unless the printed grid is still
plausibly recoverable.

## Split

Keep near-duplicate shots from the same bee/session in the same split when
possible:

- `images/train`, `labels/train`
- `images/val`, `labels/val`
- `images/test`, `labels/test`

YOLO segmentation labels contain one row per marker:

```text
0 x1 y1 x2 y2 x3 y3 x4 y4 ...
```

Coordinates are normalized to image width and height.

## Easiest Workflow

1. Crop raw bee photos to the same center ROI used by the scanner:

   ```bash
   python3 tools/make_aruco_crops.py raw_data \
     --out aruco_paper_seg/images/to_annotate
   ```

2. Annotate the ROI crops in Label Studio with `label_studio_aruco_polygon.xml`.

3. Export the normal Label Studio JSON.

4. Convert the export to YOLO segmentation labels:

   ```bash
   python3 tools/label_studio_to_yolo_seg.py export.json \
     --images aruco_paper_seg/images/to_annotate \
     --labels-out label_studio_yolo_seg_labels \
     --write-empty
   ```

5. Split the image/label pairs:

   ```bash
   python3 tools/split_yolo_dataset.py \
     --images aruco_paper_seg/images/to_annotate \
     --labels label_studio_yolo_seg_labels \
     --out aruco_paper_seg \
     --allow-empty
   ```

6. Train:

   ```bash
   python3 tools/train_aruco_segmentation.py
   ```

7. Copy the best checkpoint to:

   ```text
   models/aruco_paper_seg.pt
   ```
