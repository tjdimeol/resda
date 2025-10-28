#!/usr/bin/env python
# coding: utf-8
"""
ReSDA Stage 3: Merge baseline segmentation with SAM instance segmentation.

Combines OFA baseline (stage 1) with Qwen+SAM bboxes (stage 2) into final segmentation.
Simply paints bbox regions onto baseline - no additional model inference.
"""

import os
import sys
import numpy as np
from resda_utils import dataset_cmap, ensure_uint8_rgb, place_labels
from resda_dataset_transforms import get_transformer


def merge_segmentations(seg_baseline, sam_masks, classes):
    """
    Merge baseline segmentation with SAM instance masks.

    Paints SAM-segmented pixels with their detected class IDs, overriding the baseline.

    Args:
        seg_baseline: Baseline segmentation map [H, W] with class indices
        sam_masks: dict {class_name: [mask1, mask2, ...]} where masks are boolean arrays [H, W]
        classes: Full list of 150 class names

    Returns:
        merged_seg: Final segmentation map [H, W] with SAM masks overlaid
    """
    print("\n" + "="*60)
    print("MERGING BASELINE + SAM INSTANCE SEGMENTATIONS")
    print("="*60)

    # Start with baseline
    merged_seg = seg_baseline.copy()
    img_h, img_w = seg_baseline.shape

    if not sam_masks:
        print("No SAM masks to merge - returning baseline segmentation")
        return merged_seg

    # Paint each SAM mask with its class
    total_pixels_changed = 0
    for class_name, mask_list in sam_masks.items():
        if class_name in classes:
            class_idx = classes.index(class_name)

            for i, mask in enumerate(mask_list):
                # Paint pixels where mask is True
                pixels_in_mask = np.sum(mask)
                merged_seg[mask] = class_idx
                total_pixels_changed += pixels_in_mask

                print(f"  Painted '{class_name}' (class {class_idx}) mask {i+1}: {pixels_in_mask:,} pixels")
        else:
            print(f"  WARNING: '{class_name}' not in classes list - skipping masks")

    # Count statistics
    total_pixels = img_h * img_w
    pct_changed = 100 * total_pixels_changed / total_pixels

    print(f"\nMerge complete:")
    print(f"  Pixels changed: {total_pixels_changed:,} / {total_pixels:,} ({pct_changed:.2f}%)")
    print(f"  SAM classes painted: {list(sam_masks.keys())}")

    return merged_seg


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python resdaEX_merge.py <stage1_output.npz> <stage2_output.npz>")
        sys.exit(1)

    stage1_path = sys.argv[1]
    stage2_path = sys.argv[2]

    print("="*60)
    print("ReSDA STAGE 3: MERGE")
    print("="*60)
    print(f"Stage 1 input: {stage1_path}")
    print(f"Stage 2 input: {stage2_path}")
    print()

    # Load stage 1 baseline segmentation
    stage1_data = np.load(stage1_path, allow_pickle=True)
    seg_baseline = stage1_data['segmentation']
    classes = list(stage1_data['classes'])
    filename = str(stage1_data['filename'])
    dataset = str(stage1_data['dataset'])

    print(f"Loaded baseline segmentation:")
    print(f"  Shape: {seg_baseline.shape}")
    print(f"  Classes: {len(classes)}")
    print(f"  Filename: {filename}")
    print(f"  Dataset: {dataset}")
    print()

    # Load stage 2 SAM masks
    stage2_data = np.load(stage2_path, allow_pickle=True)
    sam_masks = stage2_data['sam_masks'].item()  # dict stored in npz

    print(f"Loaded SAM masks:")
    print(f"  Classes detected: {list(sam_masks.keys())}")
    print(f"  Total masks: {sum(len(v) for v in sam_masks.values())}")
    print()

    # Perform merge
    merged_seg = merge_segmentations(seg_baseline, sam_masks, classes)

    # Convert to colored image for visualization
    segmented_img_merged, coldict_merged = dataset_cmap(merged_seg)
    segmented_img_merged = ensure_uint8_rgb(segmented_img_merged)

    # Place labels on merged result
    labeled_image_merged, _ = place_labels(
        segmented_img_merged, merged_seg, classes,
        min_distance=15
    )

    # Save merged visualization
    filename_base = os.path.splitext(filename)[0]
    output_dir = './output/segmented'
    os.makedirs(output_dir, exist_ok=True)

    output_path = f'{output_dir}/{filename_base}_MERGED.jpg'
    labeled_image_merged.save(output_path)

    print(f"\n✓ Saved merged segmentation to: {output_path}")

    # Save merged segmentation array for evaluation
    merged_npy_path = f'{output_dir}/{filename_base}_MERGED.npy'
    np.save(merged_npy_path, merged_seg)
    print(f"✓ Saved merged array to: {merged_npy_path}")

    print("\n" + "="*60)
    print("STAGE 3 COMPLETE")
    print("="*60)
