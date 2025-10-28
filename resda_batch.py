#!/usr/bin/env python
# coding: utf-8
"""
ReSDA Batch Processing Entry Point

Initializes OFA model ONCE, then processes N images through the full 3-stage pipeline.
"""

import os
import sys
import argparse
import subprocess
import glob
import numpy as np

# ============================================================================
# PARSE ARGUMENTS
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='ReSDA Batch Processing Pipeline')
    parser.add_argument('--test-images', required=True,
                        help='Path to test image directory (e.g., ./aeroscapes/images/testing)')
    parser.add_argument('--test-annotations', required=True,
                        help='Path to test annotation directory (e.g., ./aeroscapes/annotations/testing)')
    parser.add_argument('--num-files', type=int, default=1,
                        help='Number of images to process (default: 1)')
    return parser.parse_args()


# ============================================================================
# OFA MODEL INITIALIZATION (RUNS ONCE)
# ============================================================================

print("="*60)
print("ReSDA BATCH PROCESSING - INITIALIZATION")
print("="*60)

import torch
from fairseq import checkpoint_utils, utils, tasks
from tasks import *
from criterions import *
from models.segofa import SegOFAModel

tasks.register_task('segmentation', SegmentationTask)

# turn on cuda if GPU is available
use_cuda = torch.cuda.is_available()
print(f"CUDA available: {use_cuda}")

# Load pretrained ckpt & config
ckpt = './experiment_outputs/aero/checkpoint.best_mIoU_0.0020.pt'
print(f"Loading checkpoint: {ckpt}")

overrides={"prompt_prefix": 'what is the segmentation map of the image? object:',
           'batch_size_valid': 1,
           'patch_image_size': 512,
           'orig_patch_image_size': 512,
           'num_seg_tokens': 150,
           'category_list': 'car, wall, building, sky, lawn, tree, mall, road, motorcycle, reflection, grass, shadow, sidewalk, human, earth, pine, elm, mountain, plant, facade, highway, water, barn, supermarket, bush, house, sea, roof, window, vegetation, juniper, hedge, fence, garage, rock, container, swimming-pool, driveway, railing, pipeline, pipes, solar, column, sign, spruce, fir, sand, playground, skyscraper, garden, beach, grandstand, path, stairs, runway, warehouse, backhoe, crop, factory, stairway, river, bridge, hospital, windmill, market, pond, stream, construction, hill, bench, landfill, cactus, palm, clearcut, park, soccer-field, boat, lot, square, hovel, bus, roundabout, light, truck, tower, awning, streetlight, forest, ar-marker, plane, waves, lamp, post, land, obstacle, bald-tree, tractor, farm, silo, brook, door, van, ship, fountain, railroad, canopy, fence-pole, antenna, fire, statue, track, stadium, waterfall, tent, clouds, structure, hotel, freeway, airport, cow, dog, storage-tank, baseball-diamond, tennis-court, gravel, animal, bicycle, lake, basketball-court, meadow, vehicle, flag, street, walkway, helicopter, man, woman, person, paved-area, harbor, desert, dirt, woodland, bike, drone, background, boardwalk, conflicting, cell-tower, other'
          }

models, cfg, task = checkpoint_utils.load_model_ensemble_and_task(
        utils.split_paths(ckpt),
        arg_overrides=overrides
    )

model = models[0]
if use_cuda:
    model.cuda()
model.eval()

print("✓ OFA model loaded and ready")

# Import functions from resda_baseline_aero.py
import resda_baseline_aero
from resda_baseline_aero import (
    get_classes, weight_booster, objective,
    classes, gt_classes
)

# Inject loaded model into resda_baseline_aero's namespace
resda_baseline_aero.model = model
resda_baseline_aero.task = task
resda_baseline_aero.cfg = cfg

print("✓ Imported functions from resda_baseline_aero.py")


# ============================================================================
# MAIN BATCH PROCESSING LOOP
# ============================================================================

if __name__ == "__main__":
    args = parse_args()

    # Extract dataset name from checkpoint
    dataset = ckpt.split('/')[-2]

    # Get file lists
    GTfiles = sorted(glob.glob(os.path.join(args.test_annotations, "*.png")))
    SEGfiles = sorted(glob.glob(os.path.join(args.test_images, "*.jpg")))

    if not GTfiles or not SEGfiles:
        print(f"ERROR: No files found!")
        print(f"  GT directory: {args.test_annotations}")
        print(f"  Image directory: {args.test_images}")
        sys.exit(1)

    # Limit to num_files
    GTfiles = GTfiles[:args.num_files]
    SEGfiles = SEGfiles[:args.num_files]

    print("\n" + "="*60)
    print(f"PROCESSING {len(SEGfiles)} IMAGES")
    print("="*60)
    print(f"Test images: {args.test_images}")
    print(f"Test annotations: {args.test_annotations}")
    print(f"Dataset: {dataset}")
    print()

    for i, (GTf, f) in enumerate(zip(GTfiles, SEGfiles)):
        print("\n" + "="*80)
        print(f"IMAGE {i+1}/{len(SEGfiles)}: {os.path.basename(f)}")
        print("="*80)

        # ====================================================================
        # STAGE 1: OFA BASELINE SEGMENTATION
        # ====================================================================
        print("\n[STAGE 1/3] Running OFA baseline segmentation...")

        # Get classes from VLM
        bc, ow = get_classes(f)
        override_weights = ow
        boosted_classes = bc

        print(f"Boosted classes: {boosted_classes}")
        weights = weight_booster(boosted_classes, classes, boost=0, neutral=0, override_weights=override_weights)

        # Run objective function with loaded model
        baseline_mIoU, seg_baseline = objective(GTf, f, weights, classes, gt_classes)
        print(f"✓ Baseline mIoU: {baseline_mIoU:.4f}")

        # Save baseline segmentation to .npz
        filename = os.path.basename(f)
        output_dir = './output'
        os.makedirs(output_dir, exist_ok=True)

        stage1_output = f'{output_dir}/stage1_baseline.npz'
        np.savez(stage1_output,
                 segmentation=seg_baseline,
                 classes=classes,
                 filename=filename,
                 dataset=dataset)

        print(f"✓ Saved stage 1 output: {stage1_output}")

        # ====================================================================
        # STAGE 2: QWEN+SAM INSTANCE DETECTION
        # ====================================================================
        print("\n[STAGE 2/3] Running Qwen+SAM instance detection...")

        # Call stage 2 via subprocess (tosam environment)
        stage2_cmd = [
            'conda', 'run', '-n', 'tosam', 'python',
            'resda_sam_instance.py', f
        ]

        result = subprocess.run(stage2_cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"ERROR in stage 2:")
            print(result.stderr)
            sys.exit(1)

        print(result.stdout)
        print(f"✓ Stage 2 complete")

        # ====================================================================
        # STAGE 3: MERGE BASELINE + INSTANCES
        # ====================================================================
        print("\n[STAGE 3/3] Merging baseline + instances...")

        stage2_output = f'{output_dir}/stage2_instances.npz'

        # Call stage 3 via subprocess (tosam environment)
        stage3_cmd = [
            'conda', 'run', '-n', 'tosam', 'python',
            'resdaEX_merge.py', stage1_output, stage2_output
        ]

        result = subprocess.run(stage3_cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"ERROR in stage 3:")
            print(result.stderr)
            sys.exit(1)

        print(result.stdout)
        print(f"✓ Stage 3 complete")

        print("\n" + "="*80)
        print(f"✓ PIPELINE COMPLETE FOR IMAGE {i+1}/{len(SEGfiles)}")
        print("="*80)

    print("\n" + "="*80)
    print(f"✓ ALL {len(SEGfiles)} IMAGES PROCESSED SUCCESSFULLY")
    print("="*80)
