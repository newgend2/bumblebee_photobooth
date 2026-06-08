"""
ArUco marker scanner for Raspberry Pi 5 + Camera Module 3 + SSD1306 OLED.

Controls (cv2 window):
  Preview : SPACE = capture  |  +/= = focus closer  |  - = focus further  |  Q = quit
  Captured: SPACE = detect   |  R = retake           |  Q = quit
  Result  : SPACE = scan again                       |  Q = quit

Hardware:
  - Raspberry Pi 5
  - Pi Camera Module 3 (via picamera2)
  - SSD1306 128x64 OLED on I2C (SDA=GPIO2, SCL=GPIO3)

Dependencies:
  pip install picamera2 opencv-contrib-python adafruit-circuitpython-ssd1306 pillow numpy
"""

from pathlib import Path
import time

import board
import busio
import adafruit_ssd1306
import cv2
import numpy as np
from PIL import Image, ImageDraw
from picamera2 import Picamera2

# =========================
# USER SETTINGS
# =========================

DICT_MODE             = "custom"   # "default" or "custom"
MARKER_SIZE_BITS      = 4
PREDEFINED_DICT_COUNT = 1000
CUSTOM_DICT_NPZ       = "./CUSTOM_4X4_4000_FROM_DICT_4X4_1000.npz"

TUNING_PRESET    = "very_small_marker"
MANUAL_OVERRIDES = {}

# Full-resolution still capture
CAPTURE_WIDTH  = 4608
CAPTURE_HEIGHT = 2592

# Live preview resolution (lower = faster loop)
PREVIEW_WIDTH  = 1280
PREVIEW_HEIGHT = 720

# Lens position in diopters (1/metres).
#   0.0  = infinity focus
#   1.0  = 1 m
#   5.0  = 20 cm
#   9.0  = ~11 cm  (default — close macro focus)
#   15.0 = maximum close-up (camera-dependent)
LENS_POSITION = 15.0
LENS_STEP     = 0.5   # amount each +/- keypress moves the lens

# Maximum long-edge size when displaying still images on screen (px)
DISPLAY_MAX_PX = 1280

OUTPUT_PATH = "detected_overlay.jpg"

OLED_WIDTH  = 128
OLED_HEIGHT = 64

WINDOW = "ArUco Scanner"


# =========================
# DICTIONARY HELPERS
# =========================

def get_predefined_dict_id(marker_size_bits: int, dict_count: int):
    dict_name = f"DICT_{marker_size_bits}X{marker_size_bits}_{dict_count}"
    if not hasattr(cv2.aruco, dict_name):
        raise ValueError(f"OpenCV does not provide {dict_name}")
    return getattr(cv2.aruco, dict_name), dict_name


def load_predefined_dictionary(marker_size_bits: int, dict_count: int):
    dict_id, dict_name = get_predefined_dict_id(marker_size_bits, dict_count)
    if hasattr(cv2.aruco, "getPredefinedDictionary"):
        aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
    else:
        aruco_dict = cv2.aruco.Dictionary_get(dict_id)
    return aruco_dict, dict_name


def load_custom_dictionary(npz_path: str):
    data = np.load(npz_path)
    aruco_dict = cv2.aruco.Dictionary(
        data["bytesList"],
        int(data["markerSize"][0]),
        int(data["maxCorrectionBits"][0]),
    )
    return aruco_dict, Path(npz_path).stem


def load_dictionary():
    if DICT_MODE == "default":
        return load_predefined_dictionary(MARKER_SIZE_BITS, PREDEFINED_DICT_COUNT)
    elif DICT_MODE == "custom":
        return load_custom_dictionary(CUSTOM_DICT_NPZ)
    else:
        raise ValueError("DICT_MODE must be 'default' or 'custom'")


# =========================
# DETECTOR PARAMETERS
# =========================

def make_detector_params():
    if hasattr(cv2.aruco, "DetectorParameters"):
        p = cv2.aruco.DetectorParameters()
    else:
        p = cv2.aruco.DetectorParameters_create()

    p.markerBorderBits = 1

    if TUNING_PRESET == "default":
        pass

    elif TUNING_PRESET == "small_marker":
        p.minMarkerPerimeterRate = 0.1
        p.maxMarkerPerimeterRate = 5.0
        p.adaptiveThreshWinSizeMin = 3
        p.adaptiveThreshWinSizeMax = 23
        p.adaptiveThreshWinSizeStep = 2
        p.adaptiveThreshConstant = 7
        p.polygonalApproxAccuracyRate = 0.7
        p.perspectiveRemovePixelPerCell = 8
        p.perspectiveRemoveIgnoredMarginPerCell = 0.13
        p.cornerRefinementMethod = getattr(cv2.aruco, "CORNER_REFINE_SUBPIX", 1)
        p.cornerRefinementWinSize = 5
        p.cornerRefinementMaxIterations = 50
        p.cornerRefinementMinAccuracy = 0.01

    elif TUNING_PRESET == "very_small_marker":
        # Targets markers that are only ~10-40px wide in frame.
        #
        # Key design choices vs small_marker:
        #   - minMarkerPerimeterRate dropped to 0.02 to catch tiny candidates
        #   - adaptiveThreshWinSizeMax kept small (13) so the local window
        #     doesn't dwarf the marker and wash out its edges
        #   - perspectiveRemovePixelPerCell lowered to 3 so bit sampling
        #     works when each cell is only 2-3px across
        #   - errorCorrectionRate and maxErroneousBitsInBorderRate tightened
        #     to suppress the false positives that come with aggressive size limits
        p.minMarkerPerimeterRate = 0.02
        p.maxMarkerPerimeterRate = 8.0
        p.adaptiveThreshWinSizeMin = 3
        p.adaptiveThreshWinSizeMax = 13
        p.adaptiveThreshWinSizeStep = 2
        p.adaptiveThreshConstant = 7
        p.polygonalApproxAccuracyRate = 0.2
        p.perspectiveRemovePixelPerCell = 3
        p.perspectiveRemoveIgnoredMarginPerCell = 0.10
        p.cornerRefinementMethod = getattr(cv2.aruco, "CORNER_REFINE_SUBPIX", 1)
        p.cornerRefinementWinSize = 3
        p.cornerRefinementMaxIterations = 30
        p.cornerRefinementMinAccuracy = 0.01
        p.errorCorrectionRate = 0.3
        p.maxErroneousBitsInBorderRate = 0.2

    else:
        raise ValueError(f"Unknown TUNING_PRESET: {TUNING_PRESET!r}")

    for k, v in MANUAL_OVERRIDES.items():
        if not hasattr(p, k):
            print(f"[WARN] DetectorParameters has no attribute '{k}', skipping")
            continue
        setattr(p, k, v)

    return p


# =========================
# OLED HELPERS
# =========================

def make_oled():
    i2c = busio.I2C(board.SCL, board.SDA)
    oled = adafruit_ssd1306.SSD1306_I2C(OLED_WIDTH, OLED_HEIGHT, i2c)
    oled.fill(0)
    oled.show()
    return oled


def render_oled(oled, lines: list[str]):
    img  = Image.new("1", (OLED_WIDTH, OLED_HEIGHT), 0)
    draw = ImageDraw.Draw(img)
    for i, line in enumerate(lines[:4]):
        draw.text((0, i * 16), line, fill=1)
    oled.image(img)
    oled.show()


def _wrap(text: str, width: int) -> list[str]:
    lines = []
    while len(text) > width:
        lines.append(text[:width])
        text = text[width:]
    if text:
        lines.append(text)
    return lines


# =========================
# CAMERA
# =========================

def make_camera() -> Picamera2:
    cam = Picamera2()
    cfg = cam.create_preview_configuration(
        main={"format": "RGB888", "size": (PREVIEW_WIDTH, PREVIEW_HEIGHT)}
    )
    cam.configure(cfg)
    cam.start()
    time.sleep(2)  # let AGC/AWB settle
    cam.set_controls({"AfMode": 0, "LensPosition": LENS_POSITION})
    return cam


def capture_still(cam: Picamera2, lens_pos: float) -> np.ndarray:
    """Switch to full-res still mode, grab one frame, return to preview mode."""
    still_cfg = cam.create_still_configuration(
        main={"format": "RGB888", "size": (CAPTURE_WIDTH, CAPTURE_HEIGHT)}
    )
    frame = cam.switch_mode_and_capture_array(still_cfg, "main")
    # Mode switch resets controls, so re-apply manual focus
    cam.set_controls({"AfMode": 0, "LensPosition": lens_pos})
    return frame


# =========================
# DISPLAY HELPERS
# =========================

def scale_to_fit(img_bgr: np.ndarray) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    scale = min(1.0, DISPLAY_MAX_PX / max(h, w))
    if scale < 1.0:
        return cv2.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img_bgr


def draw_hud(img_bgr: np.ndarray, lines: list[str]) -> np.ndarray:
    """Render a darkened strip at the bottom of the image with instruction text."""
    out   = img_bgr.copy()
    h = out.shape[0]
    font  = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thick = 1
    lh    = 22   # line height px
    pad   = 6

    strip_h = len(lines) * lh + pad * 2
    strip   = out[h - strip_h:h, :]
    out[h - strip_h:h, :] = (strip * 0.35).astype(np.uint8)

    for i, line in enumerate(lines):
        y = h - strip_h + pad + (i + 1) * lh - 4
        cv2.putText(out, line, (pad, y), font, scale, (255, 255, 255), thick, cv2.LINE_AA)

    return out


# =========================
# DETECTION & ANNOTATION
# =========================

def run_detection(frame: np.ndarray, aruco_dict, detector_params):
    """Returns (corners, ids, rejected). ids is a sorted flat list of ints."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)
        corners, ids_raw, rejected = detector.detectMarkers(gray)
    else:
        corners, ids_raw, rejected = cv2.aruco.detectMarkers(
            gray, aruco_dict, parameters=detector_params
        )

    if ids_raw is not None and len(ids_raw):
        # Sort by ID value while keeping corners paired with their ID
        pairs = sorted(zip(ids_raw.flatten().tolist(), corners), key=lambda x: x[0])
        ids = [p[0] for p in pairs]
        corners = [p[1] for p in pairs]
    else:
        ids = []
    return corners, ids, rejected


def draw_overlay(frame: np.ndarray, corners, ids: list[int]) -> np.ndarray:
    """Return a BGR copy of frame with marker outlines and ID labels drawn on."""
    overlay = frame.copy()

    if not ids:
        return overlay

    ids_np = np.array(ids, dtype=np.int32).reshape(-1, 1)
    cv2.aruco.drawDetectedMarkers(overlay, corners, ids_np)

    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.5, overlay.shape[1] / 3000)
    thickness  = max(1, int(font_scale * 2))

    for marker_id, marker_corners in zip(ids, corners):
        pts   = marker_corners.reshape((4, 2)).astype(int)
        label = f"ID {marker_id}"
        lx    = int(pts[0][0])
        ly    = max(int(pts[0][1]) - 8, 20)

        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        cv2.rectangle(
            overlay,
            (lx - 2, ly - th - baseline),
            (lx + tw + 2, ly + baseline),
            (0, 0, 0),
            cv2.FILLED,
        )
        cv2.putText(overlay, label, (lx, ly), font, font_scale, (0, 255, 0), thickness)

    return overlay


# =========================
# MAIN
# =========================

def main():
    print("Loading ArUco dictionary...")
    aruco_dict, dict_label = load_dictionary()
    detector_params = make_detector_params()
    print(f"Dictionary: {dict_label}")

    print("Initialising OLED...")
    oled = make_oled()
    render_oled(oled, ["ArUco scanner", "Starting..."])

    print("Starting camera...")
    cam = make_camera()
    lens_pos = LENS_POSITION

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)

    try:
        while True:

            # ── LIVE PREVIEW ──────────────────────────────────────────────────
            render_oled(oled, [
                "Preview",
                f"Focus: {lens_pos:.1f}",
                "SPC:capture",
                "Q:quit",
            ])

            while True:
                frame_prev = cam.capture_array()   # RGB888 at preview resolution
                hud        = draw_hud(frame_prev, [
                    f"Lens: {lens_pos:.1f} diopters  ( + / - to adjust )",
                    "SPACE: capture    Q: quit",
                ])
                cv2.imshow(WINDOW, hud)
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    return
                elif key in (ord(' '), 13):        # space or enter → capture
                    break
                elif key in (ord('+'), ord('=')):
                    lens_pos = min(15.0, round(lens_pos + LENS_STEP, 1))
                    cam.set_controls({"AfMode": 0, "LensPosition": lens_pos})
                    render_oled(oled, ["Preview", f"Focus: {lens_pos:.1f}", "SPC:capture", "Q:quit"])
                elif key == ord('-'):
                    lens_pos = max(0.0, round(lens_pos - LENS_STEP, 1))
                    cam.set_controls({"AfMode": 0, "LensPosition": lens_pos})
                    render_oled(oled, ["Preview", f"Focus: {lens_pos:.1f}", "SPC:capture", "Q:quit"])

            # ── CAPTURE ───────────────────────────────────────────────────────
            render_oled(oled, ["Capturing..."])
            cv2.imshow(WINDOW, draw_hud(frame_prev, ["Capturing — please wait..."]))
            cv2.waitKey(1)

            print("Capturing high-res still...")
            frame = capture_still(cam, lens_pos)
            print(f"Captured {frame.shape[1]}x{frame.shape[0]}")

            # ── REVIEW CAPTURED IMAGE ─────────────────────────────────────────
            render_oled(oled, ["Review", "SPC:detect", "R:retake", "Q:quit"])

            while True:
                display = draw_hud(
                    scale_to_fit(frame),
                    ["SPACE: detect    R: retake    Q: quit"],
                )
                cv2.imshow(WINDOW, display)
                key = cv2.waitKey(50) & 0xFF

                if key == ord('r'):
                    render_oled(oled, ["Retaking..."])
                    cv2.imshow(WINDOW, draw_hud(scale_to_fit(frame), ["Retaking — please wait..."]))
                    cv2.waitKey(1)
                    print("Retaking...")
                    frame = capture_still(cam, lens_pos)
                    print(f"Captured {frame.shape[1]}x{frame.shape[0]}")
                    render_oled(oled, ["Review", "SPC:detect", "R:retake", "Q:quit"])
                elif key in (ord(' '), 13):
                    break
                elif key == ord('q'):
                    return

            # ── DETECTION ─────────────────────────────────────────────────────
            render_oled(oled, ["Detecting..."])
            cv2.imshow(WINDOW, draw_hud(scale_to_fit(frame), ["Running detection..."]))
            cv2.waitKey(1)

            print("Running detection...")
            corners, ids, _ = run_detection(frame, aruco_dict, detector_params)

            if ids:
                print(f"Detected {len(ids)} marker(s): {ids}")
                render_oled(oled, [f"{len(ids)} marker(s):"] + _wrap(", ".join(str(i) for i in ids), 21))
            else:
                print("No markers detected.")
                render_oled(oled, ["No markers", "detected."])

            overlay = draw_overlay(frame, corners, ids)
            cv2.imwrite(OUTPUT_PATH, overlay)
            print(f"Saved overlay to: {OUTPUT_PATH}")

            # ── SHOW RESULT ───────────────────────────────────────────────────
            if ids:
                result_line = f"Detected {len(ids)}: " + ", ".join(str(i) for i in ids)
            else:
                result_line = "No markers detected"

            while True:
                display = draw_hud(
                    scale_to_fit(overlay),
                    [result_line, "SPACE: scan again    Q: quit"],
                )
                cv2.imshow(WINDOW, display)
                key = cv2.waitKey(50) & 0xFF

                if key in (ord(' '), 13):
                    break
                elif key == ord('q'):
                    return

    finally:
        cv2.destroyAllWindows()
        cam.stop()
        oled.fill(0)
        oled.show()
        print("Done.")


if __name__ == "__main__":
    main()
