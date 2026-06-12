#!/usr/bin/env python3
"""Split paired YOLO image/label files into train/val/test folders."""

import argparse
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
SPLITS = ("train", "val", "test")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--images",
        nargs="+",
        required=True,
        help="Image files or directories to search.",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        required=True,
        help="YOLO .txt label files or directories to search.",
    )
    parser.add_argument("--out", default="aruco_paper_seg")
    parser.add_argument("--train", type=float, default=0.8)
    parser.add_argument("--val", type=float, default=0.1)
    parser.add_argument("--test", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=67)
    parser.add_argument(
        "--group-regex",
        default=r"(?:date-)?(\d{4}-\d{2}-\d{2}).*bee-(\d+)|bee-(\d+).*date-(\d{4}-\d{2}-\d{2})",
        help="Regex used to keep related images in the same split.",
    )
    parser.add_argument(
        "--stratify-by-parent",
        action="store_true",
        help="Split each source parent folder separately to preserve folder balance.",
    )
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="Include images with empty label files as negative examples.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output image/label files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned split without copying files.",
    )
    return parser.parse_args()


def iter_files(paths, extensions):
    for text in paths:
        path = Path(text)
        if path.is_file() and path.suffix.lower() in extensions:
            yield path
        elif path.is_dir():
            for child in sorted(path.rglob("*")):
                if child.is_file() and child.suffix.lower() in extensions:
                    yield child


def unique_by_stem(paths):
    by_stem = {}
    duplicates = defaultdict(list)

    for path in paths:
        stem = path.stem
        if stem in by_stem:
            duplicates[stem].append(path)
        else:
            by_stem[stem] = path

    if duplicates:
        examples = ", ".join(sorted(list(duplicates))[:5])
        raise SystemExit(
            "Duplicate file stems found; rename files before splitting. "
            f"Examples: {examples}"
        )

    return by_stem


def group_key(stem, pattern):
    match = pattern.search(stem)
    if match:
        groups = [group for group in match.groups() if group is not None]
        return "_".join(groups) if groups else match.group(0)
    return stem


def split_groups(groups, ratios, seed, counts=None):
    train_ratio, val_ratio, test_ratio = ratios
    total_ratio = train_ratio + val_ratio + test_ratio
    if total_ratio <= 0:
        raise SystemExit("At least one split ratio must be greater than 0.")

    normalized = {
        "train": train_ratio / total_ratio,
        "val": val_ratio / total_ratio,
        "test": test_ratio / total_ratio,
    }

    group_items = list(groups.items())
    random.Random(seed).shuffle(group_items)
    total_images = sum(len(items) for _, items in group_items)
    targets = {
        "train": total_images * normalized["train"],
        "val": total_images * normalized["val"],
        "test": total_images * normalized["test"],
    }
    assignments = {split: [] for split in SPLITS}
    counts = dict(counts or {split: 0 for split in SPLITS})

    for _, items in group_items:
        candidates = [
            split
            for split in SPLITS
            if normalized[split] > 0 or split == "train"
        ]
        split = max(candidates, key=lambda name: targets[name] - counts[name])
        assignments[split].extend(items)
        counts[split] += len(items)

    return assignments


def merge_assignments(target, source):
    for split in SPLITS:
        target[split].extend(source[split])


def ensure_output_dirs(out_root):
    for split in SPLITS:
        (out_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_root / "labels" / split).mkdir(parents=True, exist_ok=True)


def copy_pair(image_path, label_path, out_root, split, overwrite):
    out_image = out_root / "images" / split / image_path.name
    out_label = out_root / "labels" / split / f"{image_path.stem}.txt"

    for target in (out_image, out_label):
        if target.exists() and not overwrite:
            raise SystemExit(
                f"Output exists: {target}. Use --overwrite or choose a new --out."
            )

    shutil.copy2(image_path, out_image)
    shutil.copy2(label_path, out_label)


def main():
    args = parse_args()
    images = unique_by_stem(iter_files(args.images, IMAGE_EXTENSIONS))
    labels = unique_by_stem(iter_files(args.labels, {".txt"}))
    group_pattern = re.compile(args.group_regex)
    groups = defaultdict(list)
    skipped_missing = []
    skipped_empty = []

    for stem, image_path in sorted(images.items()):
        label_path = labels.get(stem)
        if label_path is None:
            skipped_missing.append(image_path)
            continue
        if not args.allow_empty and not label_path.read_text(encoding="utf-8").strip():
            skipped_empty.append(image_path)
            continue
        key = group_key(stem, group_pattern)
        if args.stratify_by_parent:
            key = (str(image_path.parent), key)
        groups[key].append((image_path, label_path))

    if not groups:
        raise SystemExit("No matching image/label pairs found.")

    ratios = (args.train, args.val, args.test)
    if args.stratify_by_parent:
        assignments = {split: [] for split in SPLITS}
        groups_by_parent = defaultdict(dict)
        for (parent, group_name), items in groups.items():
            groups_by_parent[parent][group_name] = items
        for idx, parent in enumerate(sorted(groups_by_parent)):
            parent_assignments = split_groups(
                groups_by_parent[parent],
                ratios,
                args.seed + idx,
            )
            merge_assignments(assignments, parent_assignments)
            parent_counts = {
                split: len(parent_assignments[split])
                for split in SPLITS
            }
            print(f"{parent}: {parent_counts}")
    else:
        assignments = split_groups(groups, ratios, args.seed)

    for split in SPLITS:
        print(f"{split}: {len(assignments[split])} image/label pair(s)")
    if skipped_missing:
        print(f"Skipped {len(skipped_missing)} image(s) without labels.")
    if skipped_empty:
        print(f"Skipped {len(skipped_empty)} image(s) with empty labels.")

    if args.dry_run:
        return

    out_root = Path(args.out)
    ensure_output_dirs(out_root)
    for split, pairs in assignments.items():
        for image_path, label_path in pairs:
            copy_pair(image_path, label_path, out_root, split, args.overwrite)

    print(f"Wrote dataset split under {out_root}")


if __name__ == "__main__":
    main()
