#!/usr/bin/env python3
import copy
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from picamera2 import Picamera2


DESKTOP_DIR = Path("/home/pi/Desktop")
LOCATIONS_FILE = Path(__file__).with_name("locations.txt")
CUSTOM_DICT_NPZ = Path(__file__).with_name("CUSTOM_4X4_4000_FROM_DICT_4X4_1000.npz")
ARUCO_CONFIG_FILE = Path(__file__).with_name("aruco_config.json")
TIMEZONE = "America/Los_Angeles"
TIME_INPUT_FORMAT = "%Y-%m-%d %H-%M-%S"

_CUSTOM_DICT_BYTES_OWNER = None
_DETECTOR_PARAMS_SUPPORTED = None
NO_SCAN_VALUE = "noscan"
TEXT_BACKGROUND_ALPHA = 0.28

DEFAULT_DETECTOR_PARAM_VALUES = {
    "markerBorderBits": 1,
    "minMarkerPerimeterRate": 0.03,
    "maxMarkerPerimeterRate": 4.0,
    "adaptiveThreshWinSizeMin": 5,
    "adaptiveThreshWinSizeMax": 41,
    "adaptiveThreshWinSizeStep": 4,
    "adaptiveThreshConstant": 7,
    "polygonalApproxAccuracyRate": 0.03,
    "perspectiveRemovePixelPerCell": 8,
    "perspectiveRemoveIgnoredMarginPerCell": 0.25,
    "cornerRefinementWinSize": 5,
    "cornerRefinementMaxIterations": 30,
    "cornerRefinementMinAccuracy": 0.1,
    "errorCorrectionRate": 0.0,
    "maxErroneousBitsInBorderRate": 0.3,
    "minOtsuStdDev": 1.0,
}

DEFAULT_ARUCO_CONFIG = {
    "dictionary": {
        "mode": "default",
        "custom_npz": "CUSTOM_4X4_4000_FROM_DICT_4X4_1000.npz",
        "marker_size_bits": 4,
        "predefined_dict_count": 1000,
    },
    "scanning_enabled": True,
    "manual_entry_enabled": True,
    "scan_roi_ratio": 0.35,
    "active_params_file": "aruco_params_current.json",
    "legacy_params_file": "aruco_params_legacy.json",
    "saved_params_file": "aruco_params_current.json",
}

DEFAULT_ARUCO_PARAMS = {
    "name": "stable_4x4_1000",
    "description": (
        "Stable tuned ArUco settings for the built-in DICT_4X4_1000 dictionary."
    ),
    "use_detector_params": True,
    "filters": {
        "min_marker_area_ratio": 0.0,
        "max_marker_area_ratio": 1.0,
    },
    "detection_upscale_factor": 2.0,
    "preprocess_modes": ["raw", "clahe", "clahe_sharpen"],
    "detector_params": DEFAULT_DETECTOR_PARAM_VALUES,
}

TUNABLE_SETTING_SPECS = [
    {
        "label": "ROI ratio",
        "path": ("config", "scan_roi_ratio"),
        "min": 0.05,
        "max": 1.0,
        "step": 0.01,
        "kind": "float",
        "fmt": "{:.3f}",
    },
    {
        "label": "Min marker area",
        "path": ("params", "filters", "min_marker_area_ratio"),
        "min": 0.0,
        "max": 0.20,
        "step": 0.0001,
        "kind": "float",
        "fmt": "{:.5f}",
    },
    {
        "label": "Max marker area",
        "path": ("params", "filters", "max_marker_area_ratio"),
        "min": 0.0001,
        "max": 1.0,
        "step": 0.001,
        "kind": "float",
        "fmt": "{:.4f}",
    },
    {
        "label": "Adaptive constant",
        "path": ("params", "detector_params", "adaptiveThreshConstant"),
        "min": 0,
        "max": 25,
        "step": 1,
        "kind": "int",
    },
    {
        "label": "Threshold win min",
        "path": ("params", "detector_params", "adaptiveThreshWinSizeMin"),
        "min": 3,
        "max": 51,
        "step": 2,
        "kind": "int",
    },
    {
        "label": "Threshold win max",
        "path": ("params", "detector_params", "adaptiveThreshWinSizeMax"),
        "min": 3,
        "max": 101,
        "step": 2,
        "kind": "int",
    },
    {
        "label": "Threshold win step",
        "path": ("params", "detector_params", "adaptiveThreshWinSizeStep"),
        "min": 1,
        "max": 20,
        "step": 1,
        "kind": "int",
    },
    {
        "label": "Polygon accuracy",
        "path": ("params", "detector_params", "polygonalApproxAccuracyRate"),
        "min": 0.01,
        "max": 1.0,
        "step": 0.01,
        "kind": "float",
        "fmt": "{:.3f}",
    },
    {
        "label": "Perspective px/cell",
        "path": ("params", "detector_params", "perspectiveRemovePixelPerCell"),
        "min": 1,
        "max": 20,
        "step": 1,
        "kind": "int",
    },
    {
        "label": "Perspective margin",
        "path": ("params", "detector_params", "perspectiveRemoveIgnoredMarginPerCell"),
        "min": 0.0,
        "max": 0.50,
        "step": 0.01,
        "kind": "float",
        "fmt": "{:.3f}",
    },
    {
        "label": "Min Otsu std",
        "path": ("params", "detector_params", "minOtsuStdDev"),
        "min": 0.0,
        "max": 10.0,
        "step": 0.5,
        "kind": "float",
        "fmt": "{:.1f}",
    },
    {
        "label": "Error correction",
        "path": ("params", "detector_params", "errorCorrectionRate"),
        "min": 0.0,
        "max": 1.0,
        "step": 0.05,
        "kind": "float",
        "fmt": "{:.2f}",
    },
    {
        "label": "Border error rate",
        "path": ("params", "detector_params", "maxErroneousBitsInBorderRate"),
        "min": 0.0,
        "max": 1.0,
        "step": 0.05,
        "kind": "float",
        "fmt": "{:.2f}",
    },
    {
        "label": "Corner refine win",
        "path": ("params", "detector_params", "cornerRefinementWinSize"),
        "min": 1,
        "max": 15,
        "step": 1,
        "kind": "int",
    },
    {
        "label": "Detect upscale",
        "path": ("params", "detection_upscale_factor"),
        "min": 1.0,
        "max": 4.0,
        "step": 0.5,
        "kind": "float",
        "fmt": "{:.1f}",
    },
]

BEE_FILENAME_RE = re.compile(
    r"^bee-(?P<bee>\d+)_angle-(?P<angle>\d+)"
    r"_date-(?P<date>\d{4}-\d{2}-\d{2})"
    r"_time-(?P<time>\d{2}-\d{2}-\d{2})"
    r"_arucoid-(?P<aruco>nocode|noscan|\d+)\.jpg$",
    re.IGNORECASE,
)


def get_predefined_dict_id(marker_size_bits, dict_count):
    dict_name = f"DICT_{marker_size_bits}X{marker_size_bits}_{dict_count}"
    if not hasattr(cv2.aruco, dict_name):
        raise ValueError(f"OpenCV does not provide {dict_name}")
    return getattr(cv2.aruco, dict_name), dict_name


def load_predefined_dictionary(marker_size_bits, dict_count):
    dict_id, dict_name = get_predefined_dict_id(marker_size_bits, dict_count)
    if hasattr(cv2.aruco, "getPredefinedDictionary"):
        aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
    else:
        aruco_dict = cv2.aruco.Dictionary_get(dict_id)
    return aruco_dict, dict_name


def make_custom_dictionary(bytes_list, marker_size, max_correction_bits):
    predefined_count = DEFAULT_ARUCO_CONFIG["dictionary"]["predefined_dict_count"]
    dict_name = f"DICT_{marker_size}X{marker_size}_{predefined_count}"
    if hasattr(cv2.aruco, dict_name):
        dict_id = getattr(cv2.aruco, dict_name)
        if hasattr(cv2.aruco, "getPredefinedDictionary"):
            aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
        else:
            aruco_dict = cv2.aruco.Dictionary_get(dict_id)
    elif hasattr(cv2.aruco, "extendDictionary"):
        aruco_dict = cv2.aruco.extendDictionary(1, marker_size)
    else:
        aruco_dict = cv2.aruco.Dictionary(bytes_list, marker_size, max_correction_bits)

    aruco_dict.bytesList = bytes_list
    try:
        aruco_dict.maxCorrectionBits = max_correction_bits
    except AttributeError:
        pass
    return aruco_dict


def read_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, data):
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def repo_path(path_text):
    return Path(__file__).with_name(path_text)


def deep_merge(default, loaded):
    merged = copy.deepcopy(default)
    for key, value in loaded.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def ensure_aruco_config_files():
    legacy_path = repo_path(DEFAULT_ARUCO_CONFIG["legacy_params_file"])
    saved_path = repo_path(DEFAULT_ARUCO_CONFIG["saved_params_file"])

    if not ARUCO_CONFIG_FILE.exists():
        write_json(ARUCO_CONFIG_FILE, DEFAULT_ARUCO_CONFIG)

    if not legacy_path.exists():
        write_json(legacy_path, DEFAULT_ARUCO_PARAMS)

    if not saved_path.exists():
        saved_params = copy.deepcopy(DEFAULT_ARUCO_PARAMS)
        saved_params["name"] = "current_tuned"
        saved_params["description"] = "Editable tuned ArUco settings."
        write_json(saved_path, saved_params)


def load_aruco_settings():
    ensure_aruco_config_files()
    config = deep_merge(DEFAULT_ARUCO_CONFIG, read_json(ARUCO_CONFIG_FILE))
    config.pop("scan_schedule", None)
    params_path = repo_path(config["active_params_file"])

    if not params_path.exists():
        print(f"[WARN] Missing active ArUco params file: {params_path}")
        params_path = repo_path(config["legacy_params_file"])
        config["active_params_file"] = config["legacy_params_file"]

    params = deep_merge(DEFAULT_ARUCO_PARAMS, read_json(params_path))
    return {
        "config": config,
        "params": params,
        "params_path": params_path,
    }


def save_aruco_settings(scanner_settings, use_saved_params_file=True):
    config = scanner_settings["config"]
    params = scanner_settings["params"]

    if use_saved_params_file:
        params_file = config["saved_params_file"]
        params["name"] = "current_tuned"
        config["active_params_file"] = params_file
    else:
        params_file = config["active_params_file"]

    params_path = repo_path(params_file)
    write_json(params_path, params)
    write_json(ARUCO_CONFIG_FILE, config)
    scanner_settings["params_path"] = params_path


def load_legacy_aruco_settings(scanner_settings):
    legacy_file = scanner_settings["config"]["legacy_params_file"]
    legacy_path = repo_path(legacy_file)
    scanner_settings["params"] = deep_merge(DEFAULT_ARUCO_PARAMS, read_json(legacy_path))
    scanner_settings["config"]["active_params_file"] = legacy_file
    scanner_settings["params_path"] = legacy_path
    write_json(ARUCO_CONFIG_FILE, scanner_settings["config"])


def load_custom_dictionary(npz_path):
    global _CUSTOM_DICT_BYTES_OWNER

    with np.load(str(npz_path)) as data:
        bytes_list = np.ascontiguousarray(data["bytesList"], dtype=np.uint8).copy()
        marker_size = int(data["markerSize"][0])
        max_correction_bits = int(data["maxCorrectionBits"][0])

    # Some OpenCV builds keep a native view of bytesList. Keep the backing array alive.
    _CUSTOM_DICT_BYTES_OWNER = bytes_list
    aruco_dict = make_custom_dictionary(
        _CUSTOM_DICT_BYTES_OWNER,
        marker_size,
        max_correction_bits,
    )
    return aruco_dict, npz_path.stem


def load_dictionary(aruco_config):
    dictionary_config = aruco_config.get("dictionary", {})
    dict_mode = dictionary_config.get("mode", "custom")
    marker_size_bits = int(dictionary_config.get("marker_size_bits", 4))
    predefined_count = int(dictionary_config.get("predefined_dict_count", 1000))

    if dict_mode == "default":
        return load_predefined_dictionary(marker_size_bits, predefined_count)
    if dict_mode == "custom":
        custom_npz = dictionary_config.get("custom_npz", CUSTOM_DICT_NPZ.name)
        return load_custom_dictionary(repo_path(custom_npz))
    raise ValueError("dictionary mode must be 'default' or 'custom'")


def detector_params_supported():
    global _DETECTOR_PARAMS_SUPPORTED

    if _DETECTOR_PARAMS_SUPPORTED is not None:
        return _DETECTOR_PARAMS_SUPPORTED

    code = (
        "import cv2\n"
        "p = cv2.aruco.DetectorParameters_create() "
        "if hasattr(cv2.aruco, 'DetectorParameters_create') "
        "else cv2.aruco.DetectorParameters()\n"
        "print('ok')\n"
    )
    result = subprocess.run(
        [shutil.which("python3") or "python3", "-c", code],
        text=True,
        capture_output=True,
    )
    _DETECTOR_PARAMS_SUPPORTED = result.returncode == 0
    return _DETECTOR_PARAMS_SUPPORTED


def make_detector_params_from_values(values):
    if hasattr(cv2.aruco, "DetectorParameters_create"):
        p = cv2.aruco.DetectorParameters_create()
    else:
        p = cv2.aruco.DetectorParameters()

    p.cornerRefinementMethod = getattr(cv2.aruco, "CORNER_REFINE_SUBPIX", 1)

    for k, v in values.items():
        if not hasattr(p, k):
            print(f"[WARN] DetectorParameters has no attribute '{k}', skipping")
            continue
        setattr(p, k, v)

    return p


def make_detector_params_for_settings(scanner_settings):
    if not scanner_settings["params"].get("use_detector_params", False):
        return None

    if not detector_params_supported():
        print("[WARN] OpenCV detector params failed smoke test; using defaults.")
        scanner_settings["params"]["use_detector_params"] = False
        return None

    values = scanner_settings["params"].get("detector_params", {})
    values = deep_merge(DEFAULT_DETECTOR_PARAM_VALUES, values)
    scanner_settings["params"]["detector_params"] = values
    return make_detector_params_from_values(values)


def detection_upscale_factor(scanner_settings):
    factor = float(scanner_settings["params"].get("detection_upscale_factor", 1.0))
    return max(1.0, min(4.0, factor))


def detection_preprocess_modes(scanner_settings):
    configured = scanner_settings["params"].get("preprocess_modes", ["raw"])
    if isinstance(configured, str):
        configured = [configured]

    modes = []
    for mode in configured:
        mode = str(mode).strip().lower()
        if mode and mode not in modes:
            modes.append(mode)

    if "raw" not in modes:
        modes.insert(0, "raw")
    return modes or ["raw"]


def preprocess_detection_gray(gray, mode):
    if mode == "raw":
        return gray
    if mode == "equalize":
        return cv2.equalizeHist(gray)
    if mode == "clahe":
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        return clahe.apply(gray)
    if mode == "sharpen":
        blurred = cv2.GaussianBlur(gray, (0, 0), 1.0)
        return cv2.addWeighted(gray, 1.8, blurred, -0.8, 0)
    if mode == "clahe_sharpen":
        enhanced = preprocess_detection_gray(gray, "clahe")
        return preprocess_detection_gray(enhanced, "sharpen")
    return gray


def scale_detection_gray(gray, scale):
    if scale <= 1.0:
        return gray
    height, width = gray.shape[:2]
    scaled_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return cv2.resize(gray, scaled_size, interpolation=cv2.INTER_CUBIC)


def unscale_corners(corners, scale):
    if corners is None:
        return []
    if scale <= 1.0:
        return list(corners)
    return [marker_corners / scale for marker_corners in corners]


def detect_markers_in_gray(gray, aruco_dict, detector_params):
    if detector_params is None:
        return cv2.aruco.detectMarkers(gray, aruco_dict)
    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)
        return detector.detectMarkers(gray)
    return cv2.aruco.detectMarkers(gray, aruco_dict, parameters=detector_params)


def normalize_detection_result(corners, ids_raw):
    if ids_raw is None or not len(ids_raw):
        return [], []

    pairs = sorted(zip(ids_raw.flatten().tolist(), corners), key=lambda x: x[0])
    ids = [p[0] for p in pairs]
    corners = [p[1] for p in pairs]
    return corners, ids


def scan_roi_rect(frame_shape, scanner_settings):
    h, w = frame_shape[:2]
    ratio = float(scanner_settings["config"].get("scan_roi_ratio", 1 / 7))
    ratio = max(0.01, min(1.0, ratio))
    side = max(8, int(min(w, h) * ratio))
    x1 = max(0, (w - side) // 2)
    y1 = max(0, (h - side) // 2)
    x2 = min(w, x1 + side)
    y2 = min(h, y1 + side)
    return x1, y1, x2, y2


def shift_corners(corners, dx, dy):
    shifted = []
    offset = np.array([dx, dy], dtype=np.float32)
    for marker_corners in corners:
        shifted.append(marker_corners + offset)
    return shifted


def marker_area_ratio(marker_corners, frame_shape):
    h, w = frame_shape[:2]
    area = cv2.contourArea(marker_corners.reshape((4, 2)).astype(np.float32))
    return area / max(1, w * h)


def apply_marker_area_filters(corners, ids, scanner_settings, frame_shape):
    filters = scanner_settings["params"].get("filters", {})
    min_area = float(filters.get("min_marker_area_ratio", 0.0))
    max_area = float(filters.get("max_marker_area_ratio", 1.0))

    if not ids:
        return corners, ids

    kept = []
    kept_ids = []
    for marker_id, marker_corners in zip(ids, corners):
        area_ratio = marker_area_ratio(marker_corners, frame_shape)
        if min_area <= area_ratio <= max_area:
            kept.append(marker_corners)
            kept_ids.append(marker_id)

    return kept, kept_ids


def run_detection(
    frame,
    aruco_dict,
    detector_params,
    scanner_settings,
    color_code=cv2.COLOR_RGB2GRAY,
):
    x1, y1, x2, y2 = scan_roi_rect(frame.shape, scanner_settings)
    detection_frame = frame[y1:y2, x1:x2]
    gray = cv2.cvtColor(detection_frame, color_code)
    scale = detection_upscale_factor(scanner_settings)
    best_rejected = []

    for mode in detection_preprocess_modes(scanner_settings):
        prepared_gray = preprocess_detection_gray(gray, mode)
        detect_gray = scale_detection_gray(prepared_gray, scale)
        corners, ids_raw, rejected = detect_markers_in_gray(
            detect_gray,
            aruco_dict,
            detector_params,
        )

        corners = unscale_corners(corners, scale)
        rejected = unscale_corners(rejected, scale)
        if len(rejected) > len(best_rejected):
            best_rejected = rejected

        corners, ids = normalize_detection_result(corners, ids_raw)
        if not ids:
            continue

        corners = shift_corners(corners, x1, y1)
        corners, ids = apply_marker_area_filters(
            corners,
            ids,
            scanner_settings,
            frame.shape,
        )
        if ids:
            rejected = shift_corners(rejected, x1, y1)
            return corners, ids, rejected

    best_rejected = shift_corners(best_rejected, x1, y1)
    return [], [], best_rejected


def first_aruco_id(ids):
    return ids[0] if ids else None


def draw_translucent_rect(img, top_left, bottom_right, color, alpha):
    alpha = max(0.0, min(1.0, alpha))
    if alpha <= 0.0:
        return img

    x1, y1 = top_left
    x2, y2 = bottom_right
    h, w = img.shape[:2]
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(0, min(w - 1, x2))
    y2 = max(0, min(h - 1, y2))
    if x2 <= x1 or y2 <= y1:
        return img

    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, cv2.FILLED)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, dst=img)
    return img


def flip_preview_for_monitor(img):
    return cv2.flip(img, -1)


def flip_corners_for_monitor(corners, frame_shape):
    h, w = frame_shape[:2]
    offset = np.array([w - 1, h - 1], dtype=np.float32)
    scale = np.array([-1, -1], dtype=np.float32)
    return [marker_corners * scale + offset for marker_corners in corners]


def draw_aruco_overlay(frame, corners, ids):
    overlay = frame.copy()
    if not ids:
        return overlay

    ids_np = np.array(ids, dtype=np.int32).reshape(-1, 1)
    cv2.aruco.drawDetectedMarkers(overlay, corners, ids_np)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.5, overlay.shape[1] / 3000)
    thickness = max(1, int(font_scale * 2))

    for marker_id, marker_corners in zip(ids, corners):
        pts = marker_corners.reshape((4, 2)).astype(int)
        label = f"ID {marker_id}"
        lx = int(np.min(pts[:, 0]))
        ly = max(int(np.min(pts[:, 1])) - 8, 20)

        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        draw_translucent_rect(
            overlay,
            (lx - 2, ly - th - baseline),
            (lx + tw + 2, ly + baseline),
            (0, 0, 0),
            TEXT_BACKGROUND_ALPHA,
        )
        cv2.putText(
            overlay,
            label,
            (lx, ly),
            font,
            font_scale,
            (0, 255, 0),
            thickness,
        )

    return overlay


def draw_rejected_overlay(frame, rejected):
    overlay = frame.copy()
    if not rejected:
        return overlay

    for marker_corners in rejected:
        pts = marker_corners.reshape((4, 2)).astype(int)
        cv2.polylines(
            overlay,
            [pts],
            isClosed=True,
            color=(255, 0, 255),
            thickness=1,
            lineType=cv2.LINE_AA,
        )

    return overlay


def draw_scan_roi_overlay(img, scanner_settings, alpha=0.16):
    x1, y1, x2, y2 = scan_roi_rect(img.shape, scanner_settings)
    out = img.copy()
    overlay = out.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), cv2.FILLED)
    out = cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0)
    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 210, 0), 2)
    return out


def draw_area_filter_guides(img, scanner_settings):
    filters = scanner_settings["params"].get("filters", {})
    min_area = float(filters.get("min_marker_area_ratio", 0.0))
    max_area = float(filters.get("max_marker_area_ratio", 1.0))
    h, w = img.shape[:2]
    frame_area = w * h
    cx, cy = w // 2, h // 2

    out = img.copy()
    for area_ratio, color, label in (
        (max_area, (255, 180, 0), "max"),
        (min_area, (0, 0, 255), "min"),
    ):
        if area_ratio <= 0 or area_ratio >= 1:
            continue
        side = int(np.sqrt(area_ratio * frame_area))
        side = max(2, min(side, min(w, h)))
        x1 = max(0, cx - side // 2)
        y1 = max(0, cy - side // 2)
        x2 = min(w - 1, x1 + side)
        y2 = min(h - 1, y1 + side)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 1)
        cv2.putText(
            out,
            label,
            (x1, max(15, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

    return out


def draw_text_box(
    img,
    text_lines,
    bottom=False,
    position="top_right",
    scale=0.7,
    thickness=2,
    background_alpha=TEXT_BACKGROUND_ALPHA,
):
    disp = img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    line_gap = 10
    pad = 12

    sizes = [cv2.getTextSize(line, font, scale, thickness)[0] for line in text_lines]
    max_w = max((s[0] for s in sizes), default=0)
    total_h = sum(s[1] for s in sizes) + line_gap * (len(text_lines) - 1)

    h, w = disp.shape[:2]
    box_w = max_w + 2 * pad
    box_h = total_h + 2 * pad

    if bottom:
        x1 = max((w - box_w) // 2, 10)
        y1 = max(h - box_h - 20, 10)
    elif position == "top_left":
        x1 = 10
        y1 = 10
    else:
        x1 = max(w - box_w - 10, 10)
        y1 = 10

    x2 = min(x1 + box_w, w - 10)
    y2 = min(y1 + box_h, h - 10)

    background_alpha = max(0.0, min(1.0, background_alpha))
    draw_translucent_rect(disp, (x1, y1), (x2, y2), (0, 0, 0), background_alpha)

    y = y1 + pad
    for line, (tw, th) in zip(text_lines, sizes):
        y += th
        cv2.putText(
            disp, line, (x1 + pad, y),
            font, scale, (255, 255, 255), thickness, cv2.LINE_AA
        )
        y += line_gap

    return disp


def fit_image_for_display(img, max_size):
    max_w, max_h = max_size
    scale = min(max_w / img.shape[1], max_h / img.shape[0], 1.0)
    new_w = max(1, int(img.shape[1] * scale))
    new_h = max(1, int(img.shape[0] * scale))
    return cv2.resize(img, (new_w, new_h))


def get_nested_value(root, path):
    current = root
    for key in path:
        current = current[key]
    return current


def set_nested_value(root, path, value):
    current = root
    for key in path[:-1]:
        current = current.setdefault(key, {})
    current[path[-1]] = value


def format_tunable_value(value, spec):
    if "fmt" in spec:
        return spec["fmt"].format(value)
    return str(value)


def tunable_value(scanner_settings, spec):
    scope = spec["path"][0]
    path = spec["path"][1:]
    return get_nested_value(scanner_settings[scope], path)


def set_tunable_value(scanner_settings, spec, value):
    scope = spec["path"][0]
    path = spec["path"][1:]
    set_nested_value(scanner_settings[scope], path, value)


def adjust_tunable_setting(scanner_settings, spec, direction):
    value = tunable_value(scanner_settings, spec)
    new_value = value + spec["step"] * direction
    new_value = max(spec["min"], min(spec["max"], new_value))

    if spec["kind"] == "int":
        new_value = int(round(new_value))
    else:
        new_value = round(float(new_value), 6)

    set_tunable_value(scanner_settings, spec, new_value)


def tuning_bar(value, spec, width=14):
    span = spec["max"] - spec["min"]
    if span <= 0:
        filled = 0
    else:
        filled = int(round(((value - spec["min"]) / span) * width))
    filled = max(0, min(width, filled))
    return "[" + ("#" * filled) + ("-" * (width - filled)) + "]"


def tuning_panel_lines(scanner_settings, selected_idx, ids, rejected_count, message):
    params_enabled = scanner_settings["params"].get("use_detector_params", False)
    active_file = scanner_settings["config"].get("active_params_file", "unknown")
    upscale = detection_upscale_factor(scanner_settings)
    lines = [
        "ARUCO TUNING MODE",
        f"IDs: {', '.join(str(i) for i in ids) if ids else 'none'}",
        f"Rejected candidates: {rejected_count}",
        f"Detect upscale: {upscale:.1f}x",
        f"OpenCV detector params: {'on' if params_enabled else 'off'}",
        f"Active: {active_file}",
    ]

    start = max(0, selected_idx - 3)
    end = min(len(TUNABLE_SETTING_SPECS), start + 7)
    start = max(0, end - 7)

    for idx in range(start, end):
        spec = TUNABLE_SETTING_SPECS[idx]
        value = tunable_value(scanner_settings, spec)
        prefix = ">" if idx == selected_idx else " "
        lines.append(
            f"{prefix} {spec['label']}: {format_tunable_value(value, spec)} "
            f"{tuning_bar(value, spec)}"
        )

    lines.extend([
        "+/- adjust  n/p select  u toggle detector",
        "s save tuned  l legacy  t/esc exit",
    ])
    if message:
        lines.append(message)

    return lines


def draw_tuning_panel(img, scanner_settings, selected_idx, ids, rejected_count, message):
    lines = tuning_panel_lines(
        scanner_settings,
        selected_idx,
        ids,
        rejected_count,
        message,
    )
    return draw_text_box(
        img,
        lines,
        position="top_left",
        scale=0.47,
        thickness=1,
        background_alpha=TEXT_BACKGROUND_ALPHA,
    )


def refresh_detector_params(scanner_settings):
    return make_detector_params_for_settings(scanner_settings)


def run_aruco_tuning_mode(win, picam2, aruco_dict, scanner_settings, detector_params):
    selected_idx = 0
    message = "Tuning live preview."

    while True:
        frame = picam2.capture_array("main")
        corners, ids, rejected = run_detection(
            frame,
            aruco_dict,
            detector_params,
            scanner_settings,
        )

        display = flip_preview_for_monitor(frame)
        display = draw_scan_roi_overlay(display, scanner_settings, alpha=0.22)
        display = draw_area_filter_guides(display, scanner_settings)
        display_corners = flip_corners_for_monitor(corners, frame.shape)
        display_rejected = flip_corners_for_monitor(rejected, frame.shape)
        display = draw_rejected_overlay(display, display_rejected)
        display = draw_aruco_overlay(display, display_corners, ids)
        display = draw_tuning_panel(
            display,
            scanner_settings,
            selected_idx,
            ids,
            len(rejected),
            message,
        )

        cv2.imshow(win, display)
        key = cv2.waitKey(1) & 0xFF

        if key in (27, ord('t'), ord('q')):
            return detector_params
        if key in (ord('n'), ord(']')):
            selected_idx = (selected_idx + 1) % len(TUNABLE_SETTING_SPECS)
            message = ""
        elif key in (ord('p'), ord('[')):
            selected_idx = (selected_idx - 1) % len(TUNABLE_SETTING_SPECS)
            message = ""
        elif key in (ord('+'), ord('=')):
            adjust_tunable_setting(
                scanner_settings,
                TUNABLE_SETTING_SPECS[selected_idx],
                1,
            )
            detector_params = refresh_detector_params(scanner_settings)
            message = "Adjusted up."
        elif key == ord('-'):
            adjust_tunable_setting(
                scanner_settings,
                TUNABLE_SETTING_SPECS[selected_idx],
                -1,
            )
            detector_params = refresh_detector_params(scanner_settings)
            message = "Adjusted down."
        elif key == ord('u'):
            currently_enabled = scanner_settings["params"].get("use_detector_params", False)
            scanner_settings["params"]["use_detector_params"] = not currently_enabled
            detector_params = refresh_detector_params(scanner_settings)
            if detector_params is None:
                scanner_settings["params"]["use_detector_params"] = False
                message = "Detector params unavailable; using OpenCV defaults."
            else:
                message = "Detector params enabled."
        elif key == ord('s'):
            save_aruco_settings(scanner_settings, use_saved_params_file=True)
            detector_params = refresh_detector_params(scanner_settings)
            message = "Saved tuned settings as default."
        elif key == ord('l'):
            load_legacy_aruco_settings(scanner_settings)
            detector_params = refresh_detector_params(scanner_settings)
            message = "Loaded legacy settings."


def load_locations():
    if not LOCATIONS_FILE.exists():
        raise FileNotFoundError(f"Missing locations file: {LOCATIONS_FILE}")

    locations = [
        line.strip()
        for line in LOCATIONS_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not locations:
        raise ValueError(f"No locations found in {LOCATIONS_FILE}")

    return locations


def choose_location():
    locations = load_locations()

    print("Available locations:")
    for i, location in enumerate(locations, start=1):
        print(f"  {i}. {location}")

    while True:
        try:
            choice = input("Select location by number or exact name: ").strip()
        except EOFError as exc:
            raise SystemExit("Could not read a location selection from stdin.") from exc

        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(locations):
                return locations[idx - 1]

        for location in locations:
            if choice.lower() == location.lower():
                return location

        print("Please choose one of the listed locations.")


def command_exists(command):
    return shutil.which(command) is not None


def run_command(command, allow_failure=False, retries=0, retry_delay=1.0):
    for attempt in range(retries + 1):
        try:
            result = subprocess.run(
                command,
                check=True,
                text=True,
                capture_output=True,
            )
            return result
        except FileNotFoundError:
            if allow_failure:
                return None
            raise
        except subprocess.CalledProcessError as exc:
            message = exc.stderr.strip() or exc.stdout.strip() or str(exc)
            if (
                "previous request is not finished" in message.lower()
                and attempt < retries
            ):
                time.sleep(retry_delay)
                continue
            if allow_failure:
                return exc
            raise RuntimeError(f"{' '.join(command)} failed: {message}") from exc

    return None


def choose_pi_rtc_device():
    fallback_device = None
    rtc_root = Path("/sys/class/rtc")

    for rtc_path in sorted(rtc_root.glob("rtc*")):
        rtc_name = ""
        try:
            rtc_name = (rtc_path / "name").read_text(encoding="utf-8").strip()
        except OSError:
            pass

        try:
            rtc_device_path = str((rtc_path / "device").resolve())
        except OSError:
            rtc_device_path = ""

        rtc_device = Path("/dev") / rtc_path.name
        if fallback_device is None and rtc_device.exists():
            fallback_device = rtc_device

        haystack = f"{rtc_name} {rtc_device_path}".lower()
        if any(token in haystack for token in ("rpi", "rp1", "raspberry")):
            return str(rtc_device)

    if fallback_device is not None:
        return str(fallback_device)

    for candidate in (Path("/dev/rtc"), Path("/dev/rtc0"), Path("/dev/rtc1")):
        if candidate.exists():
            return str(candidate)

    return None


def set_timezone_if_available():
    if command_exists("timedatectl"):
        run_command(
            ["timedatectl", "set-timezone", TIMEZONE],
            allow_failure=True,
            retries=5,
        )


def set_ntp_enabled(enabled, allow_failure=False):
    if command_exists("timedatectl"):
        value = "true" if enabled else "false"
        run_command(
            ["timedatectl", "set-ntp", value],
            allow_failure=allow_failure,
            retries=5,
        )


def timedatectl_set_time(date_arg):
    if not command_exists("timedatectl"):
        raise RuntimeError("timedatectl is not available; cannot set system time.")

    run_command(["timedatectl", "set-time", date_arg], retries=5)


def system_date_text():
    result = run_command(
        ["date", "+%Y-%m-%d %H-%M-%S %Z"],
        allow_failure=True,
    )
    if result is None or result.returncode != 0:
        return datetime.now().strftime(TIME_INPUT_FORMAT)
    return result.stdout.strip()


def write_system_time_to_rtc():
    rtc_device = choose_pi_rtc_device()
    if rtc_device is None:
        print("No RTC device found. System time was set, but RTC was not updated.")
        return

    print(f"Writing time to RTC device {rtc_device}...")
    run_command(["hwclock", f"--rtc={rtc_device}", "--systohc", "--utc"])


def set_system_time_from_datetime(user_time):
    if os.geteuid() != 0:
        raise RuntimeError(
            "Setting system time requires root. Start the script with sudo to set time."
        )

    time_text = user_time.strftime(TIME_INPUT_FORMAT)
    date_arg = user_time.strftime("%Y-%m-%d %H:%M:%S")

    set_timezone_if_available()
    set_ntp_enabled(False)

    print(f"Setting system time to {time_text} Pacific time...")
    timedatectl_set_time(date_arg)
    print(f"System time after timedatectl: {system_date_text()}")
    write_system_time_to_rtc()
    print("NTP was left disabled so it does not overwrite the manually entered time.")

    print(f"Current system time: {system_date_text()}")


def prompt_int_field(label, min_value, max_value, display_min=None, display_max=None, width=None):
    shown_min = min_value if display_min is None else display_min
    shown_max = max_value if display_max is None else display_max

    while True:
        try:
            prompt = (
                f"{label} ({shown_min:0{width or 1}d}-"
                f"{shown_max:0{width or 1}d}): "
            )
            value = input(prompt).strip()
        except EOFError as exc:
            raise SystemExit("Could not read time input from stdin.") from exc

        if not value.isdigit():
            print("Please enter digits only.")
            continue

        number = int(value)
        if min_value <= number <= max_value:
            return number

        print(f"Please enter a value from {shown_min} to {shown_max}.")


def prompt_datetime_fields():
    while True:
        print("Enter the current local Pacific time.")
        year = prompt_int_field("Year", 2020, 2099, width=4)
        month = prompt_int_field("Month", 1, 12, display_min=1, display_max=12, width=2)
        day = prompt_int_field("Day", 1, 31, display_min=1, display_max=31, width=2)
        hour = prompt_int_field("Hour", 0, 23, display_min=0, display_max=24, width=2)
        minute = prompt_int_field("Minute", 0, 59, display_min=0, display_max=60, width=2)
        second = prompt_int_field("Second", 0, 59, display_min=0, display_max=60, width=2)

        try:
            return datetime(year, month, day, hour, minute, second)
        except ValueError as exc:
            print(f"Invalid date/time: {exc}")
            print("Please enter the date and time again.")


def prompt_to_set_time():
    set_timezone_if_available()
    print(f"Current system time: {datetime.now().strftime(TIME_INPUT_FORMAT)}")
    print(f"Timezone should be {TIMEZONE}.")

    while True:
        try:
            value = input(
                "Set the clock manually? Enter y to set time, or press Enter to keep current time: "
            ).strip()
        except EOFError as exc:
            raise SystemExit("Could not read time input from stdin.") from exc

        if not value:
            return

        if value.lower() not in ("y", "yes"):
            print("Please enter y to set the time, or press Enter to keep current time.")
            continue

        user_time = prompt_datetime_fields()
        try:
            set_system_time_from_datetime(user_time)
        except RuntimeError as exc:
            raise SystemExit(str(exc)) from exc

        return


def make_save_dir(location, captured_at):
    save_dir = DESKTOP_DIR / location / captured_at.strftime("%Y-%m-%d")
    save_dir.mkdir(parents=True, exist_ok=True)
    return save_dir


def get_next_bee_number(save_dir):
    highest = 0
    if save_dir.exists():
        for path in save_dir.iterdir():
            if not path.is_file():
                continue
            match = BEE_FILENAME_RE.match(path.name)
            if match:
                highest = max(highest, int(match.group(1)))
    return highest + 1


def aruco_id_for_filename(aruco_id):
    if aruco_id == NO_SCAN_VALUE:
        return NO_SCAN_VALUE
    return str(aruco_id) if aruco_id is not None else "nocode"


def aruco_id_for_display(aruco_id):
    if aruco_id == NO_SCAN_VALUE:
        return NO_SCAN_VALUE
    return str(aruco_id) if aruco_id is not None else "none"


def effective_aruco_id(scanned_aruco_id, manual_aruco_id):
    return manual_aruco_id if manual_aruco_id is not None else scanned_aruco_id


def aruco_status_line(scanned_aruco_id, manual_aruco_id=None):
    if manual_aruco_id is not None:
        return f"Aruco ID: {aruco_id_for_display(manual_aruco_id)} (manual)"
    return f"Aruco ID: {aruco_id_for_display(scanned_aruco_id)}"


def aruco_scanning_enabled(scanner_settings):
    return bool(scanner_settings["config"].get("scanning_enabled", True))


def manual_aruco_entry_enabled(scanner_settings):
    return bool(scanner_settings["config"].get("manual_entry_enabled", True))


def set_aruco_scanning_enabled(scanner_settings, enabled):
    scanner_settings["config"]["scanning_enabled"] = bool(enabled)
    write_json(ARUCO_CONFIG_FILE, scanner_settings["config"])


def should_scan_for_angle(scanner_settings, angle_num, same_bee_mode, ventral_first_mode):
    if not aruco_scanning_enabled(scanner_settings):
        return False

    if ventral_first_mode:
        return angle_num == 2

    return not same_bee_mode


def scheduled_no_scan_value(scanner_settings, angle_num, ventral_first_mode):
    if not aruco_scanning_enabled(scanner_settings):
        return NO_SCAN_VALUE

    if ventral_first_mode and angle_num == 1:
        return NO_SCAN_VALUE

    return None


def make_image_filename(bee_num, angle_num, captured_at, aruco_id):
    date_part = captured_at.strftime("%Y-%m-%d")
    time_part = captured_at.strftime("%H-%M-%S")
    aruco_part = aruco_id_for_filename(aruco_id)
    return (
        f"bee-{bee_num}_angle-{angle_num}_date-{date_part}"
        f"_time-{time_part}_arucoid-{aruco_part}.jpg"
    )


def is_real_aruco_id(aruco_id):
    if aruco_id is None or aruco_id == NO_SCAN_VALUE:
        return False
    return str(aruco_id).isdigit()


def manual_aruco_prompt_entry(current_aruco_id):
    if is_real_aruco_id(current_aruco_id):
        return str(current_aruco_id)
    return ""


def manual_aruco_prompt_lines(entry, current_aruco_id, message):
    lines = [
        "Manual ArUco ID",
        f"Current: {aruco_id_for_display(current_aruco_id)}",
        f"New: {entry if entry else 'none'}",
        "Type digits, Backspace edits",
        "Enter saves, Esc cancels",
    ]
    if message:
        lines.append(message)
    return lines


def prompt_manual_aruco_id(win, base_img, current_aruco_id):
    entry = manual_aruco_prompt_entry(current_aruco_id)
    message = ""

    while True:
        prompt_img = draw_text_box(
            base_img,
            manual_aruco_prompt_lines(entry, current_aruco_id, message),
            position="top_left",
            scale=0.55,
            thickness=1,
            background_alpha=0.72,
        )
        cv2.imshow(win, prompt_img)
        key = cv2.waitKey(0) & 0xFF

        if key in (27, ord('q')):
            return False, current_aruco_id
        if key in (10, 13):
            return True, int(entry) if entry else None
        if key in (8, 127):
            entry = entry[:-1]
            message = ""
        elif ord('0') <= key <= ord('9'):
            entry += chr(key)
            message = ""
        else:
            message = "Use digits, Backspace, Enter, or Esc."


def print_manual_aruco_update(aruco_id):
    if aruco_id is None:
        print("Manual ArUco ID cleared.")
    else:
        print(f"Manual ArUco ID set to {aruco_id}.")


def preview_command_lines(scanner_settings):
    if manual_aruco_entry_enabled(scanner_settings):
        return ["Press 'c' to capture, 'm' manual ID, 'q' quit"]
    return ["Press 'c' to capture, 'q' to quit"]


def review_command_lines(scanner_settings):
    if manual_aruco_entry_enabled(scanner_settings):
        return [
            "Press 'k' keep, 'm' manual ID",
            "Press 'r' retake, 'q' quit",
        ]
    return ["Press 'k' to keep, 'r' to retake, 'q' to quit"]


def filename_with_aruco_id(path, aruco_id):
    match = BEE_FILENAME_RE.match(path.name)
    if not match:
        return path

    return path.with_name(
        f"bee-{match.group('bee')}_angle-{match.group('angle')}"
        f"_date-{match.group('date')}_time-{match.group('time')}"
        f"_arucoid-{aruco_id_for_filename(aruco_id)}.jpg"
    )


def rename_placeholder_aruco_files(saved_paths, bee_num, aruco_id):
    if not is_real_aruco_id(aruco_id):
        return {}, []

    renamed = {}
    warnings = []
    for path in saved_paths:
        path = Path(path)
        match = BEE_FILENAME_RE.match(path.name)
        if not match or int(match.group("bee")) != bee_num:
            continue
        if match.group("aruco").isdigit():
            continue

        new_path = filename_with_aruco_id(path, aruco_id)
        if new_path == path:
            continue
        if new_path.exists():
            warnings.append(f"Skipped rename; target exists: {new_path}")
            continue
        if not path.exists():
            warnings.append(f"Skipped rename; source missing: {path}")
            continue

        path.rename(new_path)
        renamed[path] = new_path

    return renamed, warnings


def filename_display_lines(filename, max_chars=42):
    parts = filename.split("_")
    segments = [
        part + ("_" if i < len(parts) - 1 else "")
        for i, part in enumerate(parts)
    ]
    lines = []
    current = ""

    for segment in segments:
        if current and len(current) + len(segment) > max_chars:
            lines.append(current)
            current = segment
        else:
            current += segment

    if current:
        lines.append(current)

    return lines


def draw_capture_label(img, bee_num, aruco_id):
    scale = max(1.3, img.shape[1] / 1800)
    thickness = max(2, int(scale * 2))
    return draw_text_box(
        img,
        [f"bee #{bee_num}, aruco id: {aruco_id_for_display(aruco_id)}"],
        position="top_left",
        scale=scale,
        thickness=thickness,
    )


def update_current_save_dir(location, save_dir, bee_num, same_bee_mode):
    now_dir = make_save_dir(location, datetime.now())
    if now_dir == save_dir:
        return save_dir, bee_num

    if same_bee_mode:
        return now_dir, bee_num

    return now_dir, get_next_bee_number(now_dir)


def main(preview_res=(800, 600), still_res=(4056, 3040)):
    prompt_to_set_time()
    selected_location = choose_location()
    save_dir = make_save_dir(selected_location, datetime.now())
    bee_num = get_next_bee_number(save_dir)
    angle_num = 1
    locked_aruco_id = None
    same_bee_mode = False
    ventral_first_mode = False
    current_bee_saved_paths = []
    tmp_path = Path(tempfile.gettempdir()) / f"bee_cam_capture_{os.getpid()}.jpg"
    scanner_settings = load_aruco_settings()

    print("Loading ArUco dictionary...", flush=True)
    aruco_dict, dict_label = load_dictionary(scanner_settings["config"])
    print(f"Dictionary: {dict_label}", flush=True)
    detector_params = make_detector_params_for_settings(scanner_settings)
    if detector_params is None:
        print("Using OpenCV default ArUco detector parameters.", flush=True)
    else:
        print("Using configured ArUco detector parameters.", flush=True)
    print(f"ArUco config: {ARUCO_CONFIG_FILE}", flush=True)
    print(f"ArUco params: {scanner_settings['params_path']}", flush=True)
    print(f"Selected location: {selected_location}", flush=True)
    print(f"Saving images under: {DESKTOP_DIR / selected_location}", flush=True)
    print("Starting camera...", flush=True)

    picam2 = Picamera2()
    preview_config = picam2.create_preview_configuration(
        main={"size": preview_res, "format": "RGB888"}
    )
    still_config = picam2.create_still_configuration(
        main={"size": still_res, "format": "RGB888"}
    )

    win = "Bee Cam"

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, *preview_res)

    try:
        while True:
            save_dir, bee_num = update_current_save_dir(
                selected_location, save_dir, bee_num, same_bee_mode
            )

            picam2.configure(preview_config)
            picam2.start()

            live_aruco_id = None
            manual_aruco_id = None

            # Live preview loop
            while True:
                save_dir, bee_num = update_current_save_dir(
                    selected_location, save_dir, bee_num, same_bee_mode
                )

                frame = picam2.capture_array("main")
                display_frame = flip_preview_for_monitor(frame)
                scan_this_angle = should_scan_for_angle(
                    scanner_settings,
                    angle_num,
                    same_bee_mode,
                    ventral_first_mode,
                )
                scheduled_value = scheduled_no_scan_value(
                    scanner_settings,
                    angle_num,
                    ventral_first_mode,
                )

                if scan_this_angle:
                    corners, ids, _ = run_detection(
                        frame,
                        aruco_dict,
                        detector_params,
                        scanner_settings,
                    )
                    live_aruco_id = first_aruco_id(ids)
                    display_frame = draw_scan_roi_overlay(display_frame, scanner_settings)
                    display_corners = flip_corners_for_monitor(corners, frame.shape)
                    display_frame = draw_aruco_overlay(display_frame, display_corners, ids)

                    if same_bee_mode:
                        preview_lines = [
                            f"Continuing with bee #{bee_num}, scanning angle {angle_num}",
                            aruco_status_line(live_aruco_id, manual_aruco_id),
                            "Ventral-first dorsal scan",
                            "Press 't' to tune ArUco",
                        ]
                    else:
                        preview_lines = [
                            f"Previewing bee #{bee_num}...",
                            aruco_status_line(live_aruco_id, manual_aruco_id),
                            f"Will capture angle {angle_num}",
                            "Press 'v' for ventral-first shot",
                            "Press 't' to tune ArUco",
                            "Press 'x' for no-scan mode",
                        ]
                elif scheduled_value == NO_SCAN_VALUE:
                    live_aruco_id = NO_SCAN_VALUE
                    if aruco_scanning_enabled(scanner_settings) and ventral_first_mode:
                        preview_lines = [
                            f"Previewing bee #{bee_num}...",
                            aruco_status_line(NO_SCAN_VALUE, manual_aruco_id),
                            f"Will capture angle {angle_num}",
                            "Ventral shot: scanning held",
                            "Continue same bee to scan angle 2",
                            "Press 'v' for normal scanning",
                        ]
                    else:
                        preview_lines = [
                            f"Previewing bee #{bee_num}...",
                            aruco_status_line(NO_SCAN_VALUE, manual_aruco_id),
                            f"Will capture angle {angle_num}",
                            "ArUco scanning disabled",
                            "Press 'x' to enable scanning",
                        ]
                elif same_bee_mode:
                    live_aruco_id = locked_aruco_id
                    preview_lines = [
                        f"Continuing with bee #{bee_num}",
                        aruco_status_line(live_aruco_id, manual_aruco_id),
                        f"Will capture angle {angle_num}",
                        "ArUco scanning paused; carrying previous ID",
                    ]
                else:
                    live_aruco_id = None
                    preview_lines = [
                        f"Previewing bee #{bee_num}...",
                        aruco_status_line(live_aruco_id, manual_aruco_id),
                        f"Will capture angle {angle_num}",
                        "ArUco scanning paused",
                    ]

                frame_disp = draw_text_box(
                    display_frame,
                    preview_lines,
                    position="top_left",
                )
                frame_disp = draw_text_box(
                    frame_disp,
                    preview_command_lines(scanner_settings),
                    bottom=True
                )

                cv2.imshow(win, frame_disp)
                key = cv2.waitKey(1) & 0xFF

                if key == ord('c'):
                    captured_at = datetime.now()
                    save_dir = make_save_dir(selected_location, captured_at)
                    scanned_capture_aruco_id = (
                        live_aruco_id
                        if scan_this_angle or scheduled_value == NO_SCAN_VALUE
                        else locked_aruco_id
                    )
                    captured_aruco_id = effective_aruco_id(
                        scanned_capture_aruco_id,
                        manual_aruco_id,
                    )

                    if tmp_path.exists():
                        tmp_path.unlink()

                    picam2.switch_mode_and_capture_file(still_config, str(tmp_path))
                    picam2.stop()

                    snap = cv2.imread(str(tmp_path))
                    if snap is None:
                        if tmp_path.exists():
                            tmp_path.unlink()
                        picam2.configure(preview_config)
                        picam2.start()
                        continue

                    if scan_this_angle and manual_aruco_id is None:
                        _, still_ids, _ = run_detection(
                            snap,
                            aruco_dict,
                            detector_params,
                            scanner_settings,
                            color_code=cv2.COLOR_BGR2GRAY,
                        )
                        still_aruco_id = first_aruco_id(still_ids)
                        if still_aruco_id is not None:
                            captured_aruco_id = still_aruco_id

                    snap_labeled = draw_capture_label(snap, bee_num, captured_aruco_id)
                    disp = fit_image_for_display(snap_labeled, preview_res)
                    review_lines = review_command_lines(scanner_settings)
                    captured_base = draw_text_box(
                        disp,
                        review_lines,
                        bottom=True
                    )

                    # Review loop
                    while True:
                        cv2.imshow(win, captured_base)
                        key2 = cv2.waitKey(0) & 0xFF

                        if key2 == ord('k'):
                            final_name = make_image_filename(
                                bee_num,
                                angle_num,
                                captured_at,
                                captured_aruco_id,
                            )
                            final_path = save_dir / final_name
                            if not cv2.imwrite(str(final_path), snap_labeled):
                                raise RuntimeError(f"Could not save image to {final_path}")
                            if tmp_path.exists():
                                tmp_path.unlink()

                            current_bee_saved_paths.append(final_path)
                            renamed_paths, rename_warnings = rename_placeholder_aruco_files(
                                current_bee_saved_paths,
                                bee_num,
                                captured_aruco_id,
                            )
                            if renamed_paths:
                                current_bee_saved_paths = [
                                    renamed_paths.get(path, path)
                                    for path in current_bee_saved_paths
                                ]
                                for old_path, new_path in renamed_paths.items():
                                    print(f"Renamed with ArUco ID: {old_path} -> {new_path}")
                            for warning in rename_warnings:
                                print(f"[WARN] {warning}")

                            print(f"Saved: {final_path}")

                            saved_lines = (
                                ["Saved filename:"]
                                + filename_display_lines(final_name)
                            )
                            if renamed_paths:
                                saved_lines.append(
                                    f"Updated {len(renamed_paths)} earlier filename(s)"
                                )
                            saved_lines.append(f"Take another picture of bee #{bee_num}? (y/n)")
                            confirm_img = draw_text_box(
                                disp,
                                saved_lines,
                                bottom=True,
                                scale=0.55,
                                thickness=1,
                            )

                            while True:
                                cv2.imshow(win, confirm_img)
                                key3 = cv2.waitKey(0) & 0xFF
                                if key3 == ord('y'):
                                    same_bee_mode = True
                                    locked_aruco_id = captured_aruco_id
                                    angle_num += 1
                                    break
                                elif key3 == ord('n'):
                                    same_bee_mode = False
                                    locked_aruco_id = None
                                    ventral_first_mode = False
                                    current_bee_saved_paths = []
                                    angle_num = 1
                                    save_dir = make_save_dir(selected_location, datetime.now())
                                    bee_num = get_next_bee_number(save_dir)
                                    break
                                elif key3 == ord('q'):
                                    return

                            break

                        elif key2 == ord('r'):
                            if os.path.exists(tmp_path):
                                os.remove(tmp_path)
                            break

                        elif key2 == ord('m') and manual_aruco_entry_enabled(scanner_settings):
                            changed, entered_aruco_id = prompt_manual_aruco_id(
                                win,
                                captured_base,
                                captured_aruco_id,
                            )
                            if changed:
                                captured_aruco_id = entered_aruco_id
                                manual_aruco_id = entered_aruco_id
                                print_manual_aruco_update(captured_aruco_id)

                                snap_labeled = draw_capture_label(
                                    snap,
                                    bee_num,
                                    captured_aruco_id,
                                )
                                disp = fit_image_for_display(snap_labeled, preview_res)
                                captured_base = draw_text_box(
                                    disp,
                                    review_lines,
                                    bottom=True,
                                )

                        elif key2 == ord('q') or cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                            if os.path.exists(tmp_path):
                                os.remove(tmp_path)
                            return

                    break

                elif key == ord('q') or cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                    return
                elif key == ord('m') and manual_aruco_entry_enabled(scanner_settings):
                    current_aruco_id = effective_aruco_id(
                        live_aruco_id,
                        manual_aruco_id,
                    )
                    changed, entered_aruco_id = prompt_manual_aruco_id(
                        win,
                        frame_disp,
                        current_aruco_id,
                    )
                    if changed:
                        manual_aruco_id = entered_aruco_id
                        if same_bee_mode:
                            locked_aruco_id = entered_aruco_id
                        print_manual_aruco_update(manual_aruco_id)
                elif key == ord('x') and not same_bee_mode:
                    new_state = not aruco_scanning_enabled(scanner_settings)
                    set_aruco_scanning_enabled(scanner_settings, new_state)
                    print(f"ArUco scanning {'enabled' if new_state else 'disabled'}.")
                elif (
                    key == ord('v')
                    and not same_bee_mode
                    and angle_num == 1
                    and aruco_scanning_enabled(scanner_settings)
                ):
                    ventral_first_mode = not ventral_first_mode
                    mode = "enabled" if ventral_first_mode else "disabled"
                    print(f"Ventral-first mode {mode} for bee #{bee_num}.")
                elif (
                    key == ord('t')
                    and scan_this_angle
                    and aruco_scanning_enabled(scanner_settings)
                ):
                    detector_params = run_aruco_tuning_mode(
                        win,
                        picam2,
                        aruco_dict,
                        scanner_settings,
                        detector_params,
                    )

    finally:
        try:
            picam2.stop()
        except Exception:
            pass
        if tmp_path.exists():
            tmp_path.unlink()
        picam2.close()
        cv2.destroyAllWindows()
        print("Exiting.")


if __name__ == "__main__":
    main()
