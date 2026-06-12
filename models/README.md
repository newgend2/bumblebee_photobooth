# Models

Place the trained YOLO segmentation model here as:

```text
models/aruco_paper_seg.pt
```

Segmentation is used because it can learn slightly bent printed outlines.

`bb_pp_resize.py` uses the learned model as an optional fallback after OpenCV
ArUco detection fails. If the model file or Ultralytics package is missing, the
scanner keeps running with OpenCV and manual ID entry.
