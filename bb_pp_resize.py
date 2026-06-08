#!/usr/bin/env python3
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
TIMEZONE = "America/Los_Angeles"
TIME_INPUT_FORMAT = "%Y-%m-%d %H-%M-%S"

DICT_MODE = "custom"   # "default" or "custom"
MARKER_SIZE_BITS = 4
PREDEFINED_DICT_COUNT = 1000
TUNING_PRESET = "very_small_marker"
MANUAL_OVERRIDES = {}
_CUSTOM_DICT_BYTES_OWNER = None
USE_TUNED_DETECTOR_PARAMS = False

BEE_FILENAME_RE = re.compile(
    r"^bee-(\d+)_angle-(\d+)_date-\d{4}-\d{2}-\d{2}"
    r"_time-\d{2}-\d{2}-\d{2}_arucoid-(?:nocode|\d+)\.jpg$",
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
    dict_name = f"DICT_{marker_size}X{marker_size}_{PREDEFINED_DICT_COUNT}"
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


def load_dictionary():
    if DICT_MODE == "default":
        return load_predefined_dictionary(MARKER_SIZE_BITS, PREDEFINED_DICT_COUNT)
    if DICT_MODE == "custom":
        return load_custom_dictionary(CUSTOM_DICT_NPZ)
    raise ValueError("DICT_MODE must be 'default' or 'custom'")


def make_detector_params():
    if hasattr(cv2.aruco, "DetectorParameters_create"):
        p = cv2.aruco.DetectorParameters_create()
    else:
        p = cv2.aruco.DetectorParameters()

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


def run_detection(frame, aruco_dict, detector_params, color_code=cv2.COLOR_RGB2GRAY):
    gray = cv2.cvtColor(frame, color_code)

    if detector_params is None:
        corners, ids_raw, rejected = cv2.aruco.detectMarkers(gray, aruco_dict)
    elif hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)
        corners, ids_raw, rejected = detector.detectMarkers(gray)
    else:
        corners, ids_raw, rejected = cv2.aruco.detectMarkers(
            gray, aruco_dict, parameters=detector_params
        )

    if ids_raw is not None and len(ids_raw):
        pairs = sorted(zip(ids_raw.flatten().tolist(), corners), key=lambda x: x[0])
        ids = [p[0] for p in pairs]
        corners = [p[1] for p in pairs]
    else:
        ids = []

    return corners, ids, rejected


def first_aruco_id(ids):
    return ids[0] if ids else None


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
        lx = int(pts[0][0])
        ly = max(int(pts[0][1]) - 8, 20)

        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        cv2.rectangle(
            overlay,
            (lx - 2, ly - th - baseline),
            (lx + tw + 2, ly + baseline),
            (0, 0, 0),
            cv2.FILLED,
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


def draw_zoom_inset(img, crop_w=80, crop_h=80, inset_x=10, inset_y=10):
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2

    x1 = max(cx - crop_w // 2, 0)
    y1 = max(cy - crop_h // 2, 0)
    x2 = min(x1 + crop_w, w)
    y2 = min(y1 + crop_h, h)

    inset_crop = img[y1:y2, x1:x2]
    zoomed = cv2.resize(
        inset_crop,
        (crop_w * 2, crop_h * 2),
        interpolation=cv2.INTER_NEAREST
    )

    zh, zw = zoomed.shape[:2]
    if inset_y + zh <= h and inset_x + zw <= w:
        img[inset_y:inset_y + zh, inset_x:inset_x + zw] = zoomed

    return img


def draw_zoom_inset_top_right(img, crop_w=80, crop_h=80):
    inset_x = max(img.shape[1] - (crop_w * 2) - 10, 10)
    return draw_zoom_inset(img, crop_w=crop_w, crop_h=crop_h, inset_x=inset_x, inset_y=10)


def draw_text_box(
    img,
    text_lines,
    bottom=False,
    position="top_right",
    scale=0.7,
    thickness=2,
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

    cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 0, 0), cv2.FILLED)

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
    return str(aruco_id) if aruco_id is not None else "nocode"


def aruco_id_for_display(aruco_id):
    return str(aruco_id) if aruco_id is not None else "none"


def make_image_filename(bee_num, angle_num, captured_at, aruco_id):
    date_part = captured_at.strftime("%Y-%m-%d")
    time_part = captured_at.strftime("%H-%M-%S")
    aruco_part = aruco_id_for_filename(aruco_id)
    return (
        f"bee-{bee_num}_angle-{angle_num}_date-{date_part}"
        f"_time-{time_part}_arucoid-{aruco_part}.jpg"
    )


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
    tmp_path = Path(tempfile.gettempdir()) / f"bee_cam_capture_{os.getpid()}.jpg"

    print("Loading ArUco dictionary...", flush=True)
    aruco_dict, dict_label = load_dictionary()
    print(f"Dictionary: {dict_label}", flush=True)
    if USE_TUNED_DETECTOR_PARAMS:
        print("Creating ArUco detector parameters...", flush=True)
        detector_params = make_detector_params()
    else:
        print("Using OpenCV default ArUco detector parameters.", flush=True)
        detector_params = None
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

            # Live preview loop
            while True:
                save_dir, bee_num = update_current_save_dir(
                    selected_location, save_dir, bee_num, same_bee_mode
                )

                frame = picam2.capture_array("main")
                if same_bee_mode:
                    live_aruco_id = locked_aruco_id
                    preview_lines = [
                        f"Continuing with bee #{bee_num}, ID: {aruco_id_for_display(live_aruco_id)}",
                        f"Will capture angle {angle_num}",
                        "ArUco scanning paused",
                    ]
                else:
                    corners, ids, _ = run_detection(frame, aruco_dict, detector_params)
                    live_aruco_id = first_aruco_id(ids)
                    frame = draw_aruco_overlay(frame, corners, ids)
                    preview_lines = [
                        f"Previewing bee #{bee_num}...",
                        f"Aruco ID: {aruco_id_for_display(live_aruco_id)}",
                        f"Will capture angle {angle_num}",
                    ]

                frame = draw_zoom_inset_top_right(frame)

                frame_disp = draw_text_box(
                    frame,
                    preview_lines,
                    position="top_left",
                )
                frame_disp = draw_text_box(
                    frame_disp,
                    ["Press 'c' to capture, 'q' to quit"],
                    bottom=True
                )

                cv2.imshow(win, frame_disp)
                key = cv2.waitKey(1) & 0xFF

                if key == ord('c'):
                    captured_at = datetime.now()
                    save_dir = make_save_dir(selected_location, captured_at)
                    captured_aruco_id = locked_aruco_id if same_bee_mode else live_aruco_id

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

                    if not same_bee_mode:
                        _, still_ids, _ = run_detection(
                            snap,
                            aruco_dict,
                            detector_params,
                            color_code=cv2.COLOR_BGR2GRAY,
                        )
                        still_aruco_id = first_aruco_id(still_ids)
                        if still_aruco_id is not None:
                            captured_aruco_id = still_aruco_id

                    snap_labeled = draw_capture_label(snap, bee_num, captured_aruco_id)
                    disp = fit_image_for_display(snap_labeled, preview_res)
                    disp = draw_zoom_inset_top_right(disp)
                    captured_base = draw_text_box(
                        disp,
                        ["Press 'k' to keep, 'r' to retake, 'q' to quit"],
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

                            print(f"Saved: {final_path}")

                            saved_lines = (
                                ["Saved filename:"]
                                + filename_display_lines(final_name)
                                + [f"Take another picture of bee #{bee_num}? (y/n)"]
                            )
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

                        elif key2 == ord('q') or cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                            if os.path.exists(tmp_path):
                                os.remove(tmp_path)
                            return

                    break

                elif key == ord('q') or cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                    return

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
