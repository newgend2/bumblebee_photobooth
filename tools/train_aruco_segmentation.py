#!/usr/bin/env python3
"""Train the ArUco paper outline YOLO segmentation model."""

import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="aruco_paper_seg.yaml")
    parser.add_argument("--model", default="yolo26n-seg.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=-1)
    parser.add_argument("--device", default=None)
    parser.add_argument("--project", default="runs/aruco_seg")
    parser.add_argument("--name", default="train")
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "Ultralytics is not installed. Install it on the training PC with "
            "`pip install ultralytics`, then rerun this script."
        ) from exc

    data_path = Path(args.data)
    if not data_path.exists():
        raise SystemExit(f"Dataset YAML not found: {data_path}")

    model = YOLO(args.model)
    train_args = {
        "data": str(data_path),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "project": args.project,
        "name": args.name,
    }
    if args.device is not None:
        train_args["device"] = args.device

    results = model.train(**train_args)
    print(results)
    print()
    print("Copy the best checkpoint to models/aruco_paper_seg.pt before Pi use.")


if __name__ == "__main__":
    main()
