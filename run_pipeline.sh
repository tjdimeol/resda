#!/bin/bash
# run_pipeline.sh - Three-stage segmentation pipeline orchestrator
# Handles multi-environment execution for OFA baseline + SAM instances + merge
#
# Environment requirements:
#   - resda: OFA baseline (PyTorch 1.12.1, transformers 4.18.0, old dependencies)
#   - qsam: SAM + Qwen-VL (PyTorch 2.0+, segment-anything, openai package)
#           NOTE: Update environment name here if renamed
#
# Usage: bash run_pipeline.sh <image_path>

set -e  # Exit on error

# Configuration
IMAGE_PATH="$1"
PIPELINE_DIR="/home/tdimeola/resda"
OUTPUT_DIR="$PIPELINE_DIR/output"
STAGE1_OUTPUT="$OUTPUT_DIR/stage1_baseline.npy"
STAGE1_META="$OUTPUT_DIR/stage1_metadata.json"
STAGE2_OUTPUT="$OUTPUT_DIR/stage2_instances.npy"
STAGE2_META="$OUTPUT_DIR/stage2_metadata.json"
FINAL_OUTPUT="$OUTPUT_DIR/final_merged.jpg"

# Environment names
OFA_ENV="resda"
SAM_ENV="tosam"

# Validate input
if [ -z "$IMAGE_PATH" ]; then
    echo "Error: No image path provided"
    echo "Usage: bash run_pipeline.sh <image_path>"
    exit 1
fi

if [ ! -f "$IMAGE_PATH" ]; then
    echo "Error: Image file not found: $IMAGE_PATH"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "========================================="
echo "ReSDA Three-Stage Segmentation Pipeline"
echo "========================================="
echo "Image: $IMAGE_PATH"
echo "OFA Environment: $OFA_ENV"
echo "SAM Environment: $SAM_ENV"
echo ""

# ============================================================================
# STAGE 1: OFA Baseline Segmentation
# ============================================================================
echo "[Stage 1/3] Running OFA baseline segmentation..."
echo "  Environment: $OFA_ENV"
echo "  Output: $STAGE1_OUTPUT"

conda run -n "$OFA_ENV" python "$PIPELINE_DIR/resda_baseline_aero.py" \
    "$IMAGE_PATH"

if [ $? -ne 0 ]; then
    echo "ERROR: Stage 1 failed"
    exit 1
fi

echo "  ✓ Baseline segmentation complete"
echo ""

# ============================================================================
# STAGE 2: SAM Instance Segmentation
# ============================================================================
echo "[Stage 2/3] Running SAM instance segmentation..."
echo "  Environment: $SAM_ENV"
echo "  Input: $STAGE1_OUTPUT (for missing class analysis)"
echo "  Output: $STAGE2_OUTPUT"

conda run -n "$SAM_ENV" python "$PIPELINE_DIR/resda_sam_instance.py" \
    "$IMAGE_PATH"

if [ $? -ne 0 ]; then
    echo "ERROR: Stage 2 failed"
    exit 1
fi

echo "  ✓ Instance segmentation complete"
echo ""

# ============================================================================
# STAGE 3: Merge Baseline + Instances
# ============================================================================
echo "[Stage 3/3] Merging baseline and instance segmentations..."
echo "  Environment: $SAM_ENV"
echo "  Output: $FINAL_OUTPUT"

conda run -n "$SAM_ENV" python "$PIPELINE_DIR/resdaEX_merge.py" \
    "$STAGE1_OUTPUT" \
    "$STAGE2_OUTPUT"

if [ $? -ne 0 ]; then
    echo "ERROR: Stage 3 failed"
    exit 1
fi

echo "  ✓ Merge complete"
echo ""

# ============================================================================
# Pipeline Complete
# ============================================================================
echo "========================================="
echo "Pipeline completed successfully!"
echo "========================================="
echo "Final output: $FINAL_OUTPUT"
echo "Intermediate files:"
echo "  - $STAGE1_OUTPUT"
echo "  - $STAGE1_META"
echo "  - $STAGE2_OUTPUT"
echo "  - $STAGE2_META"
echo ""
