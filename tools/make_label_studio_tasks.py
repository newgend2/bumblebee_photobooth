#!/usr/bin/env python3
"""Create Label Studio JSON tasks while preserving source-folder metadata."""

import argparse
import csv
import json
import re
from pathlib import Path
from urllib.parse import quote


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
FILENAME_RE = re.compile(
    r"bee-(?P<bee>\d+)_angle-(?P<angle>\d+)"
    r"_date-(?P<date>\d{4}-\d{2}-\d{2})"
    r"_time-(?P<time>\d{2}-\d{2}-\d{2})"
    r"_arucoid-(?P<aruco>nocode|noscan|\d+)",
    re.IGNORECASE,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+", help="Image files or directories")
    parser.add_argument(
        "--document-root",
        required=True,
        help="Value of LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT.",
    )
    parser.add_argument("--out", default="label_studio_tasks.json")
    parser.add_argument(
        "--crop-manifest",
        default=None,
        help="CSV from make_aruco_crops.py with source/crop coordinates.",
    )
    parser.add_argument(
        "--jsonl",
        action="store_true",
        help="Write newline-delimited JSON instead of a JSON array.",
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


def resolve_manifest_path(path_text, manifest_dir):
    path = Path(path_text)
    if path.is_absolute():
        return path.resolve()
    if path.exists():
        return path.resolve()
    return (manifest_dir / path).resolve()


def load_crop_manifest(path):
    if path is None:
        return {}

    manifest_path = Path(path).resolve()
    rows = {}
    with manifest_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            crop_path = resolve_manifest_path(row["crop"], manifest_path.parent)
            source_path = resolve_manifest_path(row["source"], manifest_path.parent)
            rows[str(crop_path)] = {
                "source_path": str(source_path),
                "crop_x1": row.get("x1", ""),
                "crop_y1": row.get("y1", ""),
                "crop_x2": row.get("x2", ""),
                "crop_y2": row.get("y2", ""),
                "crop_roi_ratio": row.get("roi_ratio", ""),
            }
    return rows


def metadata_from_path(path, document_root, crop_metadata=None):
    rel_path = path.relative_to(document_root)
    crop_metadata = crop_metadata or {}
    source_path_text = crop_metadata.get("source_path")
    source_path = Path(source_path_text) if source_path_text else None
    match = FILENAME_RE.search((source_path or path).stem)
    metadata = {
        "image": f"/data/local-files/?d={quote(str(rel_path))}",
        "relative_path": str(rel_path),
        "source_folder": str(rel_path.parent),
        "filename": path.name,
    }
    if source_path is not None:
        try:
            source_rel_path = source_path.relative_to(document_root)
            metadata["source_relative_path"] = str(source_rel_path)
            metadata["source_folder"] = str(source_rel_path.parent)
        except ValueError:
            metadata["source_path"] = str(source_path)

    metadata.update({
        key: value
        for key, value in crop_metadata.items()
        if key != "source_path" and value != ""
    })
    if match:
        metadata.update({
            "date": match.group("date"),
            "bee": match.group("bee"),
            "angle": match.group("angle"),
            "aruco_id": match.group("aruco"),
        })
    return metadata


def main():
    args = parse_args()
    document_root = Path(args.document_root).resolve()
    crop_manifest = load_crop_manifest(args.crop_manifest)
    tasks = []

    for image_path in iter_images(args.images):
        image_path = image_path.resolve()
        try:
            data = metadata_from_path(
                image_path,
                document_root,
                crop_manifest.get(str(image_path)),
            )
        except ValueError:
            raise SystemExit(
                f"{image_path} is not under document root {document_root}"
            )
        tasks.append({"data": data})

    out_path = Path(args.out)
    if args.jsonl:
        out_path.write_text(
            "\n".join(json.dumps(task, sort_keys=True) for task in tasks) + "\n",
            encoding="utf-8",
        )
    else:
        out_path.write_text(json.dumps(tasks, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"Wrote {len(tasks)} task(s) to {out_path}")


if __name__ == "__main__":
    main()
