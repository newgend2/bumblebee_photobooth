#!/usr/bin/env python3
"""Convert Label Studio polygon exports to Ultralytics YOLO segmentation labels."""

import argparse
import json
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
TARGET_LABEL = "aruco_paper"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("export_json", help="Label Studio JSON export file")
    parser.add_argument(
        "--images",
        nargs="+",
        required=True,
        help="Image files or directories to search for matching filenames.",
    )
    parser.add_argument("--labels-out", default="label_studio_yolo_seg_labels")
    parser.add_argument(
        "--min-points",
        type=int,
        default=4,
        help="Minimum polygon points to export.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=8,
        help="Maximum polygon points to export.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing label files.",
    )
    parser.add_argument(
        "--write-empty",
        action="store_true",
        help="Write empty .txt labels for tasks with no marker polygon.",
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


def image_lookup(paths):
    by_name = {}
    by_stem = {}
    by_relative = {}
    for path in iter_images(paths):
        by_name.setdefault(path.name, path)
        by_stem.setdefault(path.stem, path)
        parts = path.parts
        if "raw_data" in parts:
            idx = parts.index("raw_data")
            by_relative.setdefault(str(Path(*parts[idx:])), path)
        by_relative.setdefault(str(path), path)
    return by_name, by_stem, by_relative


def task_image_name(task):
    data = task.get("data", {})
    value = data.get("image") or data.get("img") or data.get("url")
    if not value:
        return None

    parsed = urlparse(value)
    query_path = parse_qs(parsed.query).get("d", [None])[0]
    if query_path:
        return Path(unquote(query_path)).name

    clean_path = parsed.path or value
    return Path(unquote(clean_path)).name


def task_relative_path(task):
    data = task.get("data", {})
    if data.get("relative_path"):
        return str(data["relative_path"])

    value = data.get("image") or data.get("img") or data.get("url")
    if not value:
        return None

    parsed = urlparse(value)
    query_path = parse_qs(parsed.query).get("d", [None])[0]
    if query_path:
        return str(Path(unquote(query_path)))
    return None


def latest_results(task):
    annotations = task.get("annotations") or []
    if annotations:
        return annotations[-1].get("result", [])
    completions = task.get("completions") or []
    if completions:
        return completions[-1].get("result", [])
    return []


def clamp(value):
    return max(0.0, min(1.0, float(value)))


def point_from_polygon(point):
    return clamp(float(point[0]) / 100.0), clamp(float(point[1]) / 100.0)


def polygon_area(points):
    if len(points) < 3:
        return 0.0
    area = 0.0
    for idx, (x1, y1) in enumerate(points):
        x2, y2 = points[(idx + 1) % len(points)]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def label_line_from_points(points, min_points=4, max_points=8):
    if len(points) < min_points or len(points) > max_points:
        raise ValueError(
            f"Polygon must have {min_points}-{max_points} points; got {len(points)}."
        )
    if polygon_area(points) <= 0:
        raise ValueError("Polygon area is zero.")

    values = [0]
    for x, y in points:
        values.extend([clamp(x), clamp(y)])

    return " ".join(
        str(value) if isinstance(value, int) else f"{value:.6f}"
        for value in values
    )


def label_lines_from_results(results, min_points=4, max_points=8):
    lines = []
    warnings = []

    for result in results:
        value = result.get("value", {})
        if result.get("type") != "polygonlabels":
            continue
        labels = value.get("polygonlabels") or []
        if TARGET_LABEL not in labels:
            continue

        points = [point_from_polygon(point) for point in value.get("points", [])]
        try:
            lines.append(label_line_from_points(points, min_points, max_points))
        except ValueError as exc:
            warnings.append(str(exc))

    if not lines and not warnings:
        warnings.append(f"Missing {TARGET_LABEL} polygon.")
    return lines, warnings


def main():
    args = parse_args()
    tasks = json.loads(Path(args.export_json).read_text(encoding="utf-8"))
    if isinstance(tasks, dict):
        tasks = [tasks]

    by_name, by_stem, by_relative = image_lookup(args.images)
    labels_out = Path(args.labels_out)
    labels_out.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped = 0

    for task in tasks:
        image_name = task_image_name(task)
        if image_name is None:
            skipped += 1
            continue

        relative_path = task_relative_path(task)
        image_path = (
            by_relative.get(relative_path)
            or by_name.get(image_name)
            or by_stem.get(Path(image_name).stem)
        )
        if image_path is None:
            print(f"[WARN] No image match for task image: {image_name}")
            skipped += 1
            continue

        lines, warnings = label_lines_from_results(
            latest_results(task),
            args.min_points,
            args.max_points,
        )
        if not lines:
            if args.write_empty:
                lines = []
            else:
                print(f"[WARN] {' '.join(warnings)} Image: {image_name}")
                skipped += 1
                continue

        label_path = labels_out / f"{image_path.stem}.txt"
        if label_path.exists() and not args.overwrite:
            raise SystemExit(
                f"Output exists: {label_path}. Use --overwrite or choose another --labels-out."
            )
        label_text = "\n".join(lines)
        if label_text:
            label_text += "\n"
        label_path.write_text(label_text, encoding="utf-8")
        written += 1

    print(f"Wrote {written} YOLO segmentation label file(s) to {labels_out}")
    if skipped:
        print(f"Skipped {skipped} task(s)")


if __name__ == "__main__":
    main()
