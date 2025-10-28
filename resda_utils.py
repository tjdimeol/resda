#!/usr/bin/env python
# coding: utf-8
"""
Universal utility functions for ReSDA pipeline.
Dataset-agnostic functions used across all stages.
"""

import os
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import label as label_nd


def read_class_colors(image):
    """Read class colors from classColor.txt file."""
    count = 0
    filename = open('classColor.txt', 'r')
    colorlist =[]

    while True:
        count += 1
        line = filename.readline()

        if not line:
            break
        x = line[:-2]
        x = x.split("(",2)
        color = x[1]
        colorlist.append(color)

    cmap = [eval(i) for i in colorlist]

    filename.close()

    return colorlist, np.array(cmap)


def dataset_cmap(segmented_img):
    """
    Generate colormap and dictionary mapping class names to their occurrence counts in the segmented image.

    Args:
    - segmented_img: Segmented image array

    Returns:
    - np.array: Colormap for the dataset
    - dict: Dictionary mapping class names to their occurrence counts
    """
    # Read class names and their respective colors
    classes, cmap = read_class_colors(segmented_img)

    # Initialize dictionary to count class occurrences
    coldict = dict(zip(classes, [0] * len(classes)))

    # Count class occurrences in the segmented image
    unique, counts = np.unique(segmented_img, return_counts=True)
    for class_index, count in zip(unique, counts):
        if class_index < len(classes):  # Ensure we don't access out of bounds
            class_name = classes[class_index]
            coldict[class_name] += count

    segmented_image_colored = cmap[segmented_img]

    return segmented_image_colored, coldict


def ensure_uint8_rgb(image):
    """Ensure image is uint8 and RGB format."""
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)

    if len(image.shape) == 2 or image.shape[2] == 1:
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    return image


def place_labels(colored_image, segmented_image, classes, min_distance=10, save_file=None):
    """
    Place text labels on segmented image at class centroids.

    Args:
        colored_image: RGB colored segmentation image
        segmented_image: Segmentation map with class indices
        classes: List of class names
        min_distance: Minimum distance between labels to avoid overlap
        save_file: Optional file path to save labeled image

    Returns:
        pil_image: PIL Image with labels
        placed_labels: List of (centroid, class_name) tuples
    """
    labeled_image = ensure_uint8_rgb(colored_image)  # Ensure it's RGB and uint8

    height, width = segmented_image.shape
    placed_labels = []

    pil_image = Image.fromarray(labeled_image)
    draw = ImageDraw.Draw(pil_image)
    font_path = os.path.expanduser("~/.fonts/Ubuntu-R.ttf")  # Use expanduser for home directory
    font_size = 20
    font = ImageFont.truetype(font_path, font_size)

    for class_index, class_name in enumerate(classes):
        if class_index == 0:
            continue

        binary_mask = (segmented_image == class_index).astype(np.uint8)
        labeled_array, num_features = label_nd(binary_mask)

        if num_features == 0:
            continue

        sizes = [(labeled_array == i).sum() for i in range(1, num_features + 1)]
        sorted_indices = np.argsort(sizes)[::-1]

        for component_index in sorted_indices:
            largest_component_mask = (labeled_array == component_index + 1)
            labeled_indices = np.argwhere(largest_component_mask)
            centroid = labeled_indices.mean(axis=0).astype(int)

            if centroid.shape != (2,):
                continue  # skip invalid centroids

            too_close = False
            for placed_label in placed_labels:
                placed_label_array = np.array(placed_label[0])  # Convert to array
                if np.linalg.norm(centroid - placed_label_array) < min_distance:
                    too_close = True
                    break

            if not too_close:
                # Draw text on the PIL image
                draw.text((centroid[1], centroid[0]), class_name, font=font, fill=(0, 0, 0))
                placed_labels.append((centroid.tolist(), class_name))
                break

    return pil_image, placed_labels
