#!/usr/bin/env python3
"""Shared ArUco bit decoding utilities for learned tag localization."""

import cv2
import numpy as np


DEFAULT_DECODER_CONFIG = {
    "enabled": True,
    "confidence_threshold": 0.45,
    "imgsz": 640,
    "use_scan_roi": True,
    "allowed_id_min": 0,
    "allowed_id_max": 999,
    "marker_size_bits": 4,
    "marker_border_bits": 1,
    "warp_pixels_per_cell": 32,
    "cell_margin_ratio": 0.25,
    "max_data_bit_errors": 0,
    "max_border_bit_errors": 6,
    "min_hamming_margin": 2,
}


def deep_merge(default, loaded):
    merged = dict(default)
    for key, value in loaded.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def order_quad_points(points):
    """Return points as top-left, top-right, bottom-right, bottom-left."""
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    rect = np.zeros((4, 2), dtype=np.float32)
    sums = pts.sum(axis=1)
    diffs = np.diff(pts, axis=1).reshape(-1)

    rect[0] = pts[np.argmin(sums)]
    rect[2] = pts[np.argmax(sums)]
    rect[1] = pts[np.argmin(diffs)]
    rect[3] = pts[np.argmax(diffs)]
    return rect


def marker_image(aruco_dict, marker_id, side_pixels, border_bits):
    if hasattr(cv2.aruco, "generateImageMarker"):
        try:
            return cv2.aruco.generateImageMarker(
                aruco_dict,
                int(marker_id),
                int(side_pixels),
                borderBits=int(border_bits),
            )
        except TypeError:
            out = np.zeros((side_pixels, side_pixels), dtype=np.uint8)
            cv2.aruco.generateImageMarker(
                aruco_dict,
                int(marker_id),
                int(side_pixels),
                out,
                int(border_bits),
            )
            return out

    if hasattr(cv2.aruco, "drawMarker"):
        out = np.zeros((side_pixels, side_pixels), dtype=np.uint8)
        cv2.aruco.drawMarker(
            aruco_dict,
            int(marker_id),
            int(side_pixels),
            out,
            int(border_bits),
        )
        return out

    raise RuntimeError("This OpenCV build cannot generate ArUco marker images.")


def cell_bit_matrix(gray, cell_count, cell_margin_ratio):
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

    _, binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY | cv2.THRESH_OTSU,
    )

    height, width = binary.shape[:2]
    cell_w = width / cell_count
    cell_h = height / cell_count
    bits = np.zeros((cell_count, cell_count), dtype=np.uint8)

    for row in range(cell_count):
        for col in range(cell_count):
            x1 = int(round(col * cell_w))
            y1 = int(round(row * cell_h))
            x2 = int(round((col + 1) * cell_w))
            y2 = int(round((row + 1) * cell_h))

            margin_x = int(round((x2 - x1) * cell_margin_ratio))
            margin_y = int(round((y2 - y1) * cell_margin_ratio))
            sx1 = min(max(x1 + margin_x, 0), width - 1)
            sy1 = min(max(y1 + margin_y, 0), height - 1)
            sx2 = min(max(x2 - margin_x, sx1 + 1), width)
            sy2 = min(max(y2 - margin_y, sy1 + 1), height)

            cell = binary[sy1:sy2, sx1:sx2]
            bits[row, col] = 1 if np.mean(cell) >= 127 else 0

    return bits


def border_mask(cell_count, border_bits):
    mask = np.zeros((cell_count, cell_count), dtype=bool)
    border_bits = int(border_bits)
    mask[:border_bits, :] = True
    mask[-border_bits:, :] = True
    mask[:, :border_bits] = True
    mask[:, -border_bits:] = True
    return mask


class ArucoBitDecoder:
    def __init__(self, aruco_dict, config):
        self.aruco_dict = aruco_dict
        self.config = deep_merge(DEFAULT_DECODER_CONFIG, config or {})
        self.marker_size_bits = int(self.config["marker_size_bits"])
        self.border_bits = int(self.config["marker_border_bits"])
        self.cell_count = self.marker_size_bits + 2 * self.border_bits
        self.pixels_per_cell = int(self.config["warp_pixels_per_cell"])
        self.side_pixels = self.cell_count * self.pixels_per_cell
        self.cell_margin_ratio = float(self.config["cell_margin_ratio"])
        self._border_mask = border_mask(self.cell_count, self.border_bits)
        self._data_mask = ~self._border_mask
        self._references = self._build_references()

    def _build_references(self):
        min_id = max(0, int(self.config["allowed_id_min"]))
        max_id = max(min_id, int(self.config["allowed_id_max"]))
        refs = []

        for marker_id in range(min_id, max_id + 1):
            try:
                img = marker_image(
                    self.aruco_dict,
                    marker_id,
                    self.side_pixels,
                    self.border_bits,
                )
            except cv2.error:
                continue
            bits = cell_bit_matrix(img, self.cell_count, self.cell_margin_ratio)
            refs.append((marker_id, bits))

        if not refs:
            raise ValueError("No ArUco reference markers could be generated.")
        return refs

    def warp(self, frame, corners, color_code):
        if frame.ndim == 3:
            gray = cv2.cvtColor(frame, color_code)
        else:
            gray = frame

        rect = np.asarray(corners, dtype=np.float32).reshape(4, 2)
        dst = np.array(
            [
                [0, 0],
                [self.side_pixels - 1, 0],
                [self.side_pixels - 1, self.side_pixels - 1],
                [0, self.side_pixels - 1],
            ],
            dtype=np.float32,
        )
        transform = cv2.getPerspectiveTransform(rect, dst)
        return cv2.warpPerspective(gray, transform, (self.side_pixels, self.side_pixels))

    def decode(self, frame, corners, color_code=cv2.COLOR_BGR2GRAY):
        warped = self.warp(frame, corners, color_code)
        return self.decode_warped(warped)

    def decode_warped(self, warped):
        observed = cell_bit_matrix(warped, self.cell_count, self.cell_margin_ratio)
        candidates = []

        for rotation in range(4):
            rotated = np.rot90(observed, rotation)
            for marker_id, ref_bits in self._references:
                data_errors = int(np.count_nonzero(
                    rotated[self._data_mask] != ref_bits[self._data_mask]
                ))
                border_errors = int(np.count_nonzero(
                    rotated[self._border_mask] != ref_bits[self._border_mask]
                ))
                candidates.append((data_errors, border_errors, marker_id))

        candidates.sort(key=lambda item: (item[0], item[1], item[2]))
        best_data, best_border, best_id = candidates[0]
        best_score = best_data + best_border
        second_score = 999
        for data_errors, border_errors, marker_id in candidates[1:]:
            if marker_id != best_id:
                second_score = data_errors + border_errors
                break
        margin = int(second_score - best_score)

        if best_data > int(self.config["max_data_bit_errors"]):
            return None
        if best_border > int(self.config["max_border_bit_errors"]):
            return None
        if margin < int(self.config["min_hamming_margin"]):
            return None

        return best_id, best_data, best_border, margin


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
