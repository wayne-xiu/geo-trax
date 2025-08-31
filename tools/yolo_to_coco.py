#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""
Convert YOLO format annotations to COCO format for easier visualization and compatibility.

This script transforms annotation files from YOLO's normalized coordinates format to COCO's
JSON format with absolute pixel coordinates. It preserves bounding box information and class
labels, making the annotations compatible with visualization tools that support COCO format.

Usage:
    python tools/yolo_to_coco.py INPUT_LABELS [--input_images PATH] [--output_labels PATH]
                               [--class_map PATH] [--decimal_places N]

Arguments:
    INPUT_LABELS             Directory containing YOLO format annotation files (.txt)

Options:
    --input_images, -ii      Relative path to images with respect to output labels directory
                             Default: '../images'
    --output_labels, -ol     Path to save COCO annotations (defaults to INPUT_LABELS if not provided)
    --class_map, -cm         Path to YAML file mapping class IDs to labels
                             Default: '../models/yolov8s_merger8_exp1.yaml'
    --decimal_places, -dp    Number of decimal places for rounding bounding box coordinates
                             Default: 2

Input Format:
    YOLO format: class_id center_x center_y width height
    Where coordinates and dimensions are normalized between 0 and 1

Output Format:
    COCO JSON files with absolute pixel coordinates and class labels

Notes:
    - Input images must exist for the script to calculate absolute pixel coordinates
    - Class map file should contain a dictionary mapping numeric IDs to string labels
    - The script will skip processing labels without corresponding images
"""

import argparse
import json
from pathlib import Path

import cv2
import yaml

# Image file formats to consider (case-insensitive)
IMAGE_FORMATS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def convert_annotations(args):
    """
    Convert YOLO annotations to COCO format.
    """

    labels_dir = args.input_labels
    output_dir = args.output_labels if args.output_labels else labels_dir
    decimal_places = args.decimal_places
    class_map = args.class_map

    # Check if the input images path is a directory
    images_dir = output_dir / args.input_images
    if not images_dir.is_dir():
        print(f"Error: Input images path '{images_dir}' is not a directory.")
        return

    # Load all image paths
    image_paths = [f for f in images_dir.rglob("*") if f.is_file() and f.suffix.lower() in IMAGE_FORMATS]

    # Check if there are any image files in the input directory
    if not image_paths:
        print(f"Error: No image files found in input directory '{images_dir}'.")
        return

    # Load all label paths
    label_paths = [f for f in labels_dir.rglob("*") if f.is_file() and f.suffix.lower() == ".txt"]

    # Check if there are any label files in the input directory
    if not label_paths:
        print(f"Error: No label files found in input directory '{labels_dir}'.")
        return

    # Check if the number of images and labels match
    if len(image_paths) != len(label_paths):
        print(f"Warning: Number of images ({len(image_paths)}) and labels ({len(label_paths)}) do not match.")

    # Load class ID to label mapping
    try:
        with open(class_map, 'r') as f:
            class_id_to_label = yaml.safe_load(f)
        if not isinstance(class_id_to_label, dict):
            raise ValueError("Class map YAML file must contain a dictionary mapping class IDs to labels.")
    except Exception as e:
        print(f"Error loading class map file '{class_map}': {e}")
        print("Using default class mapping.")
        class_id_to_label = {}

    print(f"Found {len(image_paths)} images and {len(label_paths)} label files.")
    print(f"Converting annotations with {decimal_places} decimal places precision...")

    # Track processing statistics
    processed_count = 0
    skipped_count = 0

    # Loop through all the images and labels
    for image_path in image_paths:
        # Load the corresponding label file
        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            print(f"Warning: Label file '{label_path}' not found. Skipping image '{image_path.name}'.")
            skipped_count += 1
            continue

        # Get image size
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Error: Unable to read image '{image_path}'. Skipping.")
            skipped_count += 1
            continue

        height, width, _ = image.shape

        coco_annotations = {
            "version": "5.5.0",
            "flags": {},
            "shapes": [],
            "imagePath": str(args.input_images / image_path.name),
            "imageData": None,
            "imageHeight": height,
            "imageWidth": width,
        }

        # Read the label file
        with open(label_path, "r") as label_file:
            for annotation_line in label_file:
                parts = annotation_line.strip().split(" ")
                if len(parts) < 5:
                    print(f"Warning: Invalid line in label file '{label_path}': {annotation_line.strip()}")
                    continue

                class_id, x, y, w, h = parts[:5]
                class_id = int(class_id)

                x, y, w, h = float(x), float(y), float(w), float(h)

                # Convert normalized YOLO coordinates to absolute pixel coordinates
                x1 = round((x - w / 2) * width, decimal_places)
                y1 = round((y - h / 2) * height, decimal_places)
                x2 = round((x + w / 2) * width, decimal_places)
                y2 = round((y + h / 2) * height, decimal_places)

                coco_annotations["shapes"].append(
                    {
                        "label": class_id_to_label.get(class_id, str(class_id)),
                        "points": [[x1, y1], [x2, y2]],
                        "group_id": None,
                        "description": "",
                        "shape_type": "rectangle",
                        "flags": {},
                        "mask": None,
                    }
                )

        # Save the COCO annotations
        output_dir.mkdir(parents=True, exist_ok=True)
        coco_path = output_dir / f"{image_path.stem}.json"
        with open(coco_path, "w") as json_file:
            json.dump(coco_annotations, json_file, indent=2)

        processed_count += 1
        if processed_count % 10 == 0:
            print(f"Processed {processed_count} images...")

    print(f"Conversion complete: {processed_count} files processed, {skipped_count} files skipped.")


def parse_cli_args():
    """
    Parse command-line arguments for the YOLO to COCO conversion script.
    """
    parser = argparse.ArgumentParser(description="Convert YOLO annotations to COCO format.")

    parser.add_argument("input_labels", type=Path, help="Directory containing YOLO format annotation files (.txt)")
    parser.add_argument("--input_images", "-ii", type=Path, default='../images', help="Relative path to images with respect to the input labels directory. Default: '../images'")
    parser.add_argument("--output_labels", "-ol", type=Path, help="Path to save COCO annotations. If not provided, annotations will be saved in the input labels directory")
    parser.add_argument("--class_map", "-cm", type=Path, default="models/yolov8s_merger8_exp1.yaml", help="Path to YAML file mapping class IDs to labels")
    parser.add_argument("--decimal_places", "-dp", type=int, default=2, help="Number of decimal places for rounding bounding box coordinates. Default: 2")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_cli_args()
    convert_annotations(args)
