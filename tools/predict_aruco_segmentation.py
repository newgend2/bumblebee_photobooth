#!/usr/bin/env python3
"""Run the trained segmentation fallback and decoder on still images."""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR))

from aruco_segmentation_fallback import ArucoSegmentationFallback, outline_to_quad


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+", help="Image files or directories")
    parser.add_argument("--config", default="aruco_config.json")
    parser.add_argument("--model", default=None)
    parser.add_argument("--overlays", default=None)
    parser.add_argument(
        "--full-frame",
        action="store_true",
        help="Disable scan-ROI cropping for model inference.",
    )
    parser.add_argument(
        "--piecewise",
        action="store_true",
        help="Enable experimental piecewise rectification after quad decode fails.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=None,
        help="Override the segmentation confidence threshold.",
    )
    parser.add_argument(
        "--allowed-id-max",
        type=int,
        default=None,
        help="Override the maximum ArUco ID considered by the bit decoder.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print whether failures are no-mask, no-quad, or decode-rejected.",
    )
    parser.add_argument(
        "--failure-overlays",
        action="store_true",
        help="With --overlays, also save overlays for failed decodes when a mask exists.",
    )
    return parser.parse_args()


def iter_images(paths):
    for text in paths:
        path = Path(text)
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path
        elif path.is_dir():
            for child in sorted(path.rglob("*")):
                if child.is_file() and child.suffix.lower() in IMAGE_EXTENSIONS:
                    yield child


def read_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def load_predefined_dictionary(config):
    dictionary_config = config.get("dictionary", {})
    if dictionary_config.get("mode", "default") != "default":
        raise SystemExit(
            "predict_aruco_segmentation.py currently expects a default dictionary."
        )

    marker_size_bits = int(dictionary_config.get("marker_size_bits", 4))
    predefined_count = int(dictionary_config.get("predefined_dict_count", 1000))
    dict_name = f"DICT_{marker_size_bits}X{marker_size_bits}_{predefined_count}"
    if not hasattr(cv2.aruco, dict_name):
        raise SystemExit(f"OpenCV does not provide {dict_name}")

    dict_id = getattr(cv2.aruco, dict_name)
    if hasattr(cv2.aruco, "getPredefinedDictionary"):
        return cv2.aruco.getPredefinedDictionary(dict_id)
    return cv2.aruco.Dictionary_get(dict_id)


def draw_overlay(image, result):
    out = image.copy()
    outline = result.outline.reshape(-1, 2).astype(int)
    corners = result.corners.reshape(4, 2).astype(int)
    cv2.polylines(out, [outline], isClosed=True, color=(0, 180, 255), thickness=2)
    cv2.polylines(out, [corners], isClosed=True, color=(255, 255, 0), thickness=2)
    label = (
        f"ID {result.marker_id} seg "
        f"conf={result.confidence:.2f} "
        f"{result.rectification} "
        f"data={result.data_bit_errors} border={result.border_bit_errors}"
    )
    x = max(0, int(corners[:, 0].min()))
    y = max(24, int(corners[:, 1].min()) - 8)
    cv2.putText(
        out,
        label,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return out


def draw_diagnostic_overlay(image, outline=None, corners=None, label="nocode"):
    out = image.copy()
    if outline is not None:
        outline = outline.reshape(-1, 2).astype(int)
        cv2.polylines(out, [outline], isClosed=True, color=(0, 180, 255), thickness=2)
    if corners is not None:
        corners = corners.reshape(4, 2).astype(int)
        cv2.polylines(out, [corners], isClosed=True, color=(255, 255, 0), thickness=2)

    cv2.putText(
        out,
        label,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 180, 255),
        2,
        cv2.LINE_AA,
    )
    return out


def model_frame_for_fallback(fallback, image, scanner_settings, color_code):
    detect_frame, x_offset, y_offset = fallback._detect_frame_and_offset(
        image,
        scanner_settings,
    )
    if color_code == cv2.COLOR_RGB2GRAY and detect_frame.ndim == 3:
        model_frame = cv2.cvtColor(detect_frame, cv2.COLOR_RGB2BGR)
    else:
        model_frame = detect_frame
    return detect_frame, model_frame, x_offset, y_offset


def diagnose_segmentation(fallback, image, scanner_settings, color_code):
    detect_frame, model_frame, x_offset, y_offset = model_frame_for_fallback(
        fallback,
        image,
        scanner_settings,
        color_code,
    )
    results = fallback.model.predict(
        source=model_frame,
        imgsz=int(fallback.config["imgsz"]),
        conf=float(fallback.config["confidence_threshold"]),
        verbose=False,
    )
    mask_count = 0
    box_count = 0
    for result in results:
        if result.boxes is not None:
            box_count += len(result.boxes)
        if result.masks is not None and result.masks.xy is not None:
            mask_count += len(result.masks.xy)

    best = fallback._best_outline(results, detect_frame.shape)
    if best is None:
        return {
            "status": "no-mask",
            "box_count": box_count,
            "mask_count": mask_count,
        }

    confidence, area, outline = best
    offset = np.array([x_offset, y_offset], dtype=np.float32)
    outline = outline + offset
    corners = outline_to_quad(outline, fallback.config)
    if corners is None:
        return {
            "status": "no-quad",
            "confidence": confidence,
            "area": area,
            "box_count": box_count,
            "mask_count": mask_count,
            "outline": outline,
        }

    decoded = fallback.decoder.decode(image, corners, color_code=color_code)
    if decoded is None:
        return {
            "status": "decode-rejected",
            "confidence": confidence,
            "area": area,
            "box_count": box_count,
            "mask_count": mask_count,
            "outline": outline,
            "corners": corners,
        }

    marker_id, data_errors, border_errors, margin = decoded
    return {
        "status": "decoded",
        "confidence": confidence,
        "area": area,
        "box_count": box_count,
        "mask_count": mask_count,
        "outline": outline,
        "corners": corners,
        "marker_id": marker_id,
        "data_errors": data_errors,
        "border_errors": border_errors,
        "margin": margin,
    }


def diagnostic_text(diagnostic):
    parts = [diagnostic["status"]]
    if "confidence" in diagnostic:
        parts.append(f"conf={diagnostic['confidence']:.3f}")
    if "mask_count" in diagnostic:
        parts.append(f"masks={diagnostic['mask_count']}")
    if "box_count" in diagnostic:
        parts.append(f"boxes={diagnostic['box_count']}")
    return "\t".join(parts)


def main():
    args = parse_args()
    config = read_json(args.config)
    segment_config = dict(config.get("yolo_segmentation_fallback", {}))
    segment_config["enabled"] = True
    if args.model is not None:
        segment_config["model_path"] = args.model
    if args.full_frame:
        segment_config["use_scan_roi"] = False
    if args.piecewise:
        segment_config["piecewise_rectification_enabled"] = True
    if args.conf is not None:
        segment_config["confidence_threshold"] = args.conf
    if args.allowed_id_max is not None:
        segment_config["allowed_id_max"] = args.allowed_id_max

    aruco_dict = load_predefined_dictionary(config)
    fallback = ArucoSegmentationFallback(segment_config, aruco_dict, REPO_DIR)
    if not fallback.load():
        raise SystemExit(f"Could not load segmentation fallback: {fallback.status}")

    overlay_dir = Path(args.overlays) if args.overlays else None
    if overlay_dir is not None:
        overlay_dir.mkdir(parents=True, exist_ok=True)

    scanner_settings = {"config": config}
    for image_path in iter_images(args.images):
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"{image_path}\tnoread")
            continue

        result = fallback.detect(
            image,
            scanner_settings=scanner_settings,
            color_code=cv2.COLOR_BGR2GRAY,
        )
        if result is None:
            diagnostic = None
            if args.debug or (overlay_dir is not None and args.failure_overlays):
                diagnostic = diagnose_segmentation(
                    fallback,
                    image,
                    scanner_settings,
                    cv2.COLOR_BGR2GRAY,
                )
            if args.debug and diagnostic is not None:
                print(f"{image_path}\tnocode\t{diagnostic_text(diagnostic)}")
            else:
                print(f"{image_path}\tnocode")
            if (
                overlay_dir is not None
                and args.failure_overlays
                and diagnostic is not None
                and "outline" in diagnostic
            ):
                overlay = draw_diagnostic_overlay(
                    image,
                    outline=diagnostic.get("outline"),
                    corners=diagnostic.get("corners"),
                    label=diagnostic_text(diagnostic).replace("\t", " "),
                )
                cv2.imwrite(str(overlay_dir / image_path.name), overlay)
            continue

        print(
            f"{image_path}\tID={result.marker_id}\t"
            f"conf={result.confidence:.3f}\t"
            f"rectification={result.rectification}\t"
            f"data_errors={result.data_bit_errors}\t"
            f"border_errors={result.border_bit_errors}\t"
            f"margin={result.hamming_margin}"
        )

        if overlay_dir is not None:
            overlay = draw_overlay(image, result)
            cv2.imwrite(str(overlay_dir / image_path.name), overlay)


if __name__ == "__main__":
    main()
