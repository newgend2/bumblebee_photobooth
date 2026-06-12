#!/usr/bin/env python3
"""Create ROI crops for ArUco outline annotation."""

import argparse
import csv
from pathlib import Path

import cv2


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", nargs="+", help="Image files or directories")
    parser.add_argument("--out", default="aruco_paper_seg/images/to_annotate")
    parser.add_argument("--roi-ratio", type=float, default=0.35)
    parser.add_argument("--quality", type=int, default=95)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Review each crop and nudge/resize before saving.",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Optional CSV path for source/crop coordinates.",
    )
    parser.add_argument("--max-display-width", type=int, default=1400)
    parser.add_argument("--max-display-height", type=int, default=900)
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


def scan_roi_rect(shape, ratio):
    h, w = shape[:2]
    ratio = max(0.01, min(1.0, float(ratio)))
    side = max(8, int(min(w, h) * ratio))
    x1 = max(0, (w - side) // 2)
    y1 = max(0, (h - side) // 2)
    x2 = min(w, x1 + side)
    y2 = min(h, y1 + side)
    return x1, y1, x2, y2


def clamp_rect(x1, y1, side, shape):
    h, w = shape[:2]
    side = max(8, min(int(side), min(w, h)))
    x1 = max(0, min(int(x1), w - side))
    y1 = max(0, min(int(y1), h - side))
    return x1, y1, x1 + side, y1 + side


def output_path(out_dir, src_path, used_names):
    base = f"{src_path.stem}_roi.jpg"
    candidate = out_dir / base
    counter = 2
    while candidate.name in used_names or candidate.exists():
        candidate = out_dir / f"{src_path.stem}_roi_{counter}.jpg"
        counter += 1
    used_names.add(candidate.name)
    return candidate


def draw_review_image(image, rect, image_path, index, total, max_size):
    max_w, max_h = max_size
    h, w = image.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    disp = cv2.resize(image, (int(w * scale), int(h * scale))) if scale < 1 else image.copy()

    x1, y1, x2, y2 = [int(round(v * scale)) for v in rect]
    cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 255), 2)

    lines = [
        f"{index}/{total} {image_path.name}",
        "Enter/space save  arrows/WASD move  +/- resize  c center  n skip  q quit",
    ]
    y = 24
    for line in lines:
        cv2.putText(
            disp,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            disp,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        y += 28

    return disp


def interactive_rect(image, image_path, start_rect, roi_ratio, index, total, max_size):
    rect = start_rect
    side = rect[2] - rect[0]
    step = max(4, int(side * 0.08))
    resize_step = max(4, int(side * 0.05))
    center_rect = start_rect
    win = "ArUco crop review"

    while True:
        cv2.imshow(
            win,
            draw_review_image(image, rect, image_path, index, total, max_size),
        )
        key = cv2.waitKeyEx(0)

        if key in (ord("q"), 27):
            return None, "quit"
        if key in (ord("n"),):
            return None, "skip"
        if key in (ord(" "), 10, 13):
            return rect, "save"
        if key == ord("c"):
            rect = center_rect
            side = rect[2] - rect[0]
            continue

        x1, y1, x2, y2 = rect
        if key in (81, 2424832, ord("a")):
            x1 -= step
        elif key in (83, 2555904, ord("d")):
            x1 += step
        elif key in (82, 2490368, ord("w")):
            y1 -= step
        elif key in (84, 2621440, ord("s")):
            y1 += step
        elif key in (ord("+"), ord("="), ord("]")):
            side += resize_step
            x1 -= resize_step // 2
            y1 -= resize_step // 2
        elif key in (ord("-"), ord("_"), ord("[")):
            side -= resize_step
            x1 += resize_step // 2
            y1 += resize_step // 2
        else:
            continue

        rect = clamp_rect(x1, y1, side, image.shape)


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.manifest) if args.manifest else out_dir / "crop_manifest.csv"
    used_names = set()
    count = 0
    rows = []
    images = list(iter_images(args.source))

    for idx, image_path in enumerate(images, start=1):
        if args.limit is not None and count >= args.limit:
            break

        image = cv2.imread(str(image_path))
        if image is None:
            print(f"[WARN] Could not read: {image_path}")
            continue

        x1, y1, x2, y2 = scan_roi_rect(image.shape, args.roi_ratio)
        if args.interactive:
            rect, action = interactive_rect(
                image,
                image_path,
                (x1, y1, x2, y2),
                args.roi_ratio,
                idx,
                len(images),
                (args.max_display_width, args.max_display_height),
            )
            if action == "quit":
                break
            if action == "skip":
                print(f"Skipped: {image_path}")
                continue
            x1, y1, x2, y2 = rect

        crop = image[y1:y2, x1:x2]
        target = output_path(out_dir, image_path, used_names)
        params = [cv2.IMWRITE_JPEG_QUALITY, max(1, min(100, args.quality))]
        if not cv2.imwrite(str(target), crop, params):
            print(f"[WARN] Could not write: {target}")
            continue

        count += 1
        rows.append({
            "crop": str(target),
            "source": str(image_path),
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "roi_ratio": args.roi_ratio,
        })
        print(f"{image_path} -> {target}")

    if rows:
        with manifest_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["crop", "source", "x1", "y1", "x2", "y2", "roi_ratio"],
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote crop manifest: {manifest_path}")

    if args.interactive:
        cv2.destroyAllWindows()

    print(f"Wrote {count} ROI crop(s) to {out_dir}")


if __name__ == "__main__":
    main()
