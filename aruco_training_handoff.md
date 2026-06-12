# ArUco Bee Tag Segmentation Handoff

The current fallback path uses YOLO segmentation, not pose. The model predicts
the printed ArUco marker outline; the ID is still decoded algorithmically after
rectification.

## Runtime Flow

1. OpenCV ArUco detection runs first on the configured center scan ROI.
2. If OpenCV finds a valid marker, that ID is used.
3. If OpenCV fails, YOLO segmentation predicts the `aruco_paper` outline.
4. The predicted outline is reduced to a best-fit quadrilateral.
5. The quadrilateral is perspective-warped to a canonical square.
6. The decoder samples the ArUco cells and compares them to generated reference
   markers for the configured dictionary and ID range.
7. If the strict bit checks fail, the scanner falls back to manual entry.

The fallback does not train an ID classifier and does not run OpenCV's ArUco
detector on the predicted cutout. It uses the model only to locate and rectify
the marker.

## Annotation Rules

- Label only the printed ArUco marker boundary.
- Use 4 points for flat tags.
- Use 5-8 points only where a marker edge visibly bends.
- Keep points ordered around the perimeter.
- Do not include cut-paper slivers, bee body, or hair.
- Leave severe folds or unrecoverable occlusions unlabeled.

## Key Files

- `bb_pp_resize.py`: camera capture and OpenCV/segmentation/manual detection flow.
- `aruco_segmentation_fallback.py`: YOLO segmentation fallback.
- `aruco_bit_decoder.py`: strict warp-and-bit decoder shared by fallback tools.
- `label_studio_aruco_polygon.xml`: Label Studio polygon config.
- `tools/label_studio_to_yolo_seg.py`: Label Studio JSON to YOLO segmentation labels.
- `tools/train_aruco_segmentation.py`: segmentation training helper.
- `tools/predict_aruco_segmentation.py`: offline prediction/debug helper.
- `aruco_paper_seg.yaml`: Ultralytics segmentation dataset YAML.

## Train And Test

```bash
python3 tools/label_studio_to_yolo_seg.py export.json \
  --images aruco_paper_seg/images/to_annotate \
  --labels-out label_studio_yolo_seg_labels \
  --write-empty

python3 tools/split_yolo_dataset.py \
  --images aruco_paper_seg/images/to_annotate \
  --labels label_studio_yolo_seg_labels \
  --out aruco_paper_seg \
  --allow-empty

python3 tools/train_aruco_segmentation.py

python3 tools/predict_aruco_segmentation.py aruco_paper_seg/images/test \
  --model models/aruco_paper_seg.pt \
  --full-frame \
  --debug \
  --overlays seg_debug_overlays \
  --failure-overlays
```
