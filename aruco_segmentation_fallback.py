#!/usr/bin/env python3
"""YOLO segmentation fallback for locating and decoding ArUco tag outlines."""

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np

from aruco_bit_decoder import (
    ArucoBitDecoder,
    DEFAULT_DECODER_CONFIG,
    deep_merge,
    order_quad_points,
    scan_roi_rect,
)


DEFAULT_SEGMENT_CONFIG = deep_merge(
    DEFAULT_DECODER_CONFIG,
    {
        "model_path": "models/aruco_paper_seg.pt",
        "min_mask_area_ratio": 0.0005,
        "quad_approx_epsilon_start": 0.01,
        "quad_approx_epsilon_max": 0.12,
        "quad_search_max_vertices": 16,
        "piecewise_rectification_enabled": False,
        "piecewise_max_data_bit_errors": 0,
        "piecewise_max_border_bit_errors": 4,
        "piecewise_min_hamming_margin": 3,
    },
)


@dataclass
class SegmentationDecodeResult:
    marker_id: int
    corners: np.ndarray
    outline: np.ndarray
    confidence: float
    data_bit_errors: int
    border_bit_errors: int
    hamming_margin: int
    rectification: str = "quad"
    source: str = "yolo_segmentation"


def contour_area(points):
    points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if len(points) < 3:
        return 0.0
    return abs(float(cv2.contourArea(points)))


def sort_around_centroid(points):
    points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    return points[np.argsort(angles)]


def polyline_length(points):
    points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def simplify_hull_for_quad_search(hull, max_vertices):
    hull = np.asarray(hull, dtype=np.float32).reshape(-1, 2)
    if len(hull) <= max_vertices:
        return hull

    contour = hull.reshape(-1, 1, 2)
    perimeter = cv2.arcLength(contour, True)
    for epsilon_ratio in np.linspace(0.005, 0.06, 12):
        approx = cv2.approxPolyDP(contour, epsilon_ratio * perimeter, True).reshape(-1, 2)
        if 4 <= len(approx) <= max_vertices:
            return approx.astype(np.float32)

    indices = np.linspace(0, len(hull) - 1, max_vertices, dtype=int)
    return hull[indices].astype(np.float32)


def largest_area_quad(points, max_vertices):
    candidates = simplify_hull_for_quad_search(points, max_vertices)
    best = None
    best_area = 0.0

    for combo in combinations(candidates, 4):
        quad = sort_around_centroid(combo)
        contour = quad.reshape(-1, 1, 2).astype(np.float32)
        if not cv2.isContourConvex(contour):
            continue
        area = contour_area(quad)
        if area > best_area:
            best_area = area
            best = quad

    return best


def outline_to_quad(points, config):
    points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) < 4:
        return None

    hull = cv2.convexHull(points.reshape(-1, 1, 2)).reshape(-1, 2).astype(np.float32)
    if len(hull) < 4:
        return None
    if len(hull) == 4:
        return order_quad_points(hull)

    contour = hull.reshape(-1, 1, 2)
    perimeter = cv2.arcLength(contour, True)
    eps_start = float(config["quad_approx_epsilon_start"])
    eps_max = float(config["quad_approx_epsilon_max"])
    for epsilon_ratio in np.linspace(eps_start, eps_max, 16):
        approx = cv2.approxPolyDP(contour, epsilon_ratio * perimeter, True).reshape(-1, 2)
        if len(approx) == 4 and cv2.isContourConvex(approx.reshape(-1, 1, 2)):
            return order_quad_points(approx)

    quad = largest_area_quad(hull, int(config["quad_search_max_vertices"]))
    if quad is not None:
        return order_quad_points(quad)

    rect = cv2.boxPoints(cv2.minAreaRect(hull.astype(np.float32)))
    return order_quad_points(rect)


def path_between_indices(points, start_idx, end_idx):
    points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    n = len(points)
    if n == 0:
        return points

    if start_idx <= end_idx:
        forward = points[start_idx:end_idx + 1]
    else:
        forward = np.vstack([points[start_idx:], points[:end_idx + 1]])

    if end_idx <= start_idx:
        backward = points[end_idx:start_idx + 1][::-1]
    else:
        backward = np.vstack([points[end_idx:], points[:start_idx + 1]])[::-1]

    if polyline_length(forward) <= polyline_length(backward):
        return forward
    return backward


def interpolate_polyline(points, t_values):
    points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    t_values = np.asarray(t_values, dtype=np.float32)
    if len(points) == 0:
        return np.zeros((len(t_values), 2), dtype=np.float32)
    if len(points) == 1:
        return np.repeat(points, len(t_values), axis=0)

    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
    total = cumulative[-1]
    if total <= 0:
        return np.repeat(points[:1], len(t_values), axis=0)

    distances = np.clip(t_values, 0.0, 1.0) * total
    segment_idx = np.searchsorted(cumulative, distances, side="right") - 1
    segment_idx = np.clip(segment_idx, 0, len(points) - 2)
    segment_start = cumulative[segment_idx]
    segment_len = np.maximum(lengths[segment_idx], 1e-6)
    local_t = ((distances - segment_start) / segment_len).reshape(-1, 1)
    return points[segment_idx] * (1.0 - local_t) + points[segment_idx + 1] * local_t


def nearest_outline_indices(outline, corners):
    outline = np.asarray(outline, dtype=np.float32).reshape(-1, 2)
    corners = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    indices = []
    for corner in corners:
        distances = np.linalg.norm(outline - corner, axis=1)
        indices.append(int(np.argmin(distances)))
    if len(set(indices)) != 4:
        return None
    return indices


def piecewise_warp(frame, outline, corners, side_pixels, color_code):
    if frame.ndim == 3:
        gray = cv2.cvtColor(frame, color_code)
    else:
        gray = frame

    outline = np.asarray(outline, dtype=np.float32).reshape(-1, 2)
    corners = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    indices = nearest_outline_indices(outline, corners)
    if indices is None:
        return None

    top = path_between_indices(outline, indices[0], indices[1])
    right = path_between_indices(outline, indices[1], indices[2])
    bottom = path_between_indices(outline, indices[3], indices[2])
    left = path_between_indices(outline, indices[0], indices[3])

    if min(len(top), len(right), len(bottom), len(left)) < 2:
        return None

    u = np.linspace(0.0, 1.0, side_pixels, dtype=np.float32)
    v = np.linspace(0.0, 1.0, side_pixels, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)

    top_pts = interpolate_polyline(top, u)
    bottom_pts = interpolate_polyline(bottom, u)
    left_pts = interpolate_polyline(left, v)
    right_pts = interpolate_polyline(right, v)

    tl, tr, br, bl = corners
    bilinear = (
        (1.0 - uu)[..., None] * (1.0 - vv)[..., None] * tl
        + uu[..., None] * (1.0 - vv)[..., None] * tr
        + uu[..., None] * vv[..., None] * br
        + (1.0 - uu)[..., None] * vv[..., None] * bl
    )
    boundary_patch = (
        (1.0 - vv)[..., None] * top_pts[None, :, :]
        + vv[..., None] * bottom_pts[None, :, :]
        + (1.0 - uu)[..., None] * left_pts[:, None, :]
        + uu[..., None] * right_pts[:, None, :]
        - bilinear
    )

    map_x = boundary_patch[..., 0].astype(np.float32)
    map_y = boundary_patch[..., 1].astype(np.float32)
    return cv2.remap(
        gray,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


class ArucoSegmentationFallback:
    def __init__(self, config, aruco_dict, repo_dir):
        self.config = deep_merge(DEFAULT_SEGMENT_CONFIG, config or {})
        self.aruco_dict = aruco_dict
        self.repo_dir = Path(repo_dir)
        self.enabled = bool(self.config.get("enabled", True))
        self.model = None
        self.status = "disabled"
        self.decoder = None
        self.piecewise_decoder = None

        if self.enabled:
            self.status = "not loaded"

    @property
    def model_path(self):
        model_path = Path(str(self.config.get("model_path", "")))
        if not model_path.is_absolute():
            model_path = self.repo_dir / model_path
        return model_path

    def available(self):
        if not self.enabled:
            return False
        if not self.model_path.exists():
            self.status = f"model not found: {self.model_path}"
            return False
        return True

    def load(self):
        if not self.available():
            return False

        try:
            from ultralytics import YOLO
        except ImportError:
            self.status = "ultralytics is not installed"
            return False

        try:
            if self.decoder is None:
                self.decoder = ArucoBitDecoder(self.aruco_dict, self.config)
            if self.piecewise_decoder is None:
                piecewise_config = deep_merge(
                    self.config,
                    {
                        "max_data_bit_errors": self.config["piecewise_max_data_bit_errors"],
                        "max_border_bit_errors": self.config["piecewise_max_border_bit_errors"],
                        "min_hamming_margin": self.config["piecewise_min_hamming_margin"],
                    },
                )
                self.piecewise_decoder = ArucoBitDecoder(self.aruco_dict, piecewise_config)
            self.model = YOLO(str(self.model_path))
        except Exception as exc:
            self.model = None
            self.status = f"load failed: {exc}"
            return False

        self.status = f"loaded: {self.model_path}"
        return True

    def _ensure_loaded(self):
        if self.model is not None:
            return True
        return self.load()

    def _best_outline(self, results, frame_shape):
        threshold = float(self.config["confidence_threshold"])
        min_area_ratio = float(self.config["min_mask_area_ratio"])
        frame_area = max(1, frame_shape[0] * frame_shape[1])
        best = None

        for result in results:
            if result.masks is None:
                continue

            polygons = result.masks.xy or []
            boxes = result.boxes
            for idx, polygon in enumerate(polygons):
                confidence = 1.0
                if boxes is not None and boxes.conf is not None and idx < len(boxes.conf):
                    confidence = float(boxes.conf[idx])
                if confidence < threshold:
                    continue

                outline = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
                outline = outline[np.isfinite(outline).all(axis=1)]
                if len(outline) < 4:
                    continue

                area = contour_area(outline)
                if area / frame_area < min_area_ratio:
                    continue

                candidate = (confidence, area, outline)
                if best is None or candidate[:2] > best[:2]:
                    best = candidate

        return best

    def _detect_frame_and_offset(self, frame, scanner_settings):
        x_offset = 0
        y_offset = 0
        detect_frame = frame

        if self.config.get("use_scan_roi", True) and scanner_settings is not None:
            x1, y1, x2, y2 = scan_roi_rect(frame.shape, scanner_settings)
            detect_frame = frame[y1:y2, x1:x2]
            x_offset = x1
            y_offset = y1

        return detect_frame, x_offset, y_offset

    def detect(self, frame, scanner_settings=None, color_code=cv2.COLOR_BGR2GRAY):
        if not self._ensure_loaded():
            return None

        detect_frame, x_offset, y_offset = self._detect_frame_and_offset(
            frame,
            scanner_settings,
        )

        if color_code == cv2.COLOR_RGB2GRAY and detect_frame.ndim == 3:
            model_frame = cv2.cvtColor(detect_frame, cv2.COLOR_RGB2BGR)
        else:
            model_frame = detect_frame

        results = self.model.predict(
            source=model_frame,
            imgsz=int(self.config["imgsz"]),
            conf=float(self.config["confidence_threshold"]),
            verbose=False,
        )
        best = self._best_outline(results, detect_frame.shape)
        if best is None:
            return None

        confidence, _, outline = best
        offset = np.array([x_offset, y_offset], dtype=np.float32)
        outline = outline + offset
        corners = outline_to_quad(outline, self.config)
        if corners is None:
            return None

        decoded = self.decoder.decode(frame, corners, color_code=color_code)
        rectification = "quad"
        source = "yolo_segmentation"

        if decoded is None and self.config.get("piecewise_rectification_enabled", False):
            warped = piecewise_warp(
                frame,
                outline,
                corners,
                self.piecewise_decoder.side_pixels,
                color_code,
            )
            if warped is not None:
                decoded = self.piecewise_decoder.decode_warped(warped)
                rectification = "piecewise"
                source = "yolo_segmentation_piecewise"

        if decoded is None:
            return None

        marker_id, data_errors, border_errors, margin = decoded
        return SegmentationDecodeResult(
            marker_id=marker_id,
            corners=corners.reshape(1, 4, 2).astype(np.float32),
            outline=outline.astype(np.float32),
            confidence=confidence,
            data_bit_errors=data_errors,
            border_bit_errors=border_errors,
            hamming_margin=margin,
            rectification=rectification,
            source=source,
        )
