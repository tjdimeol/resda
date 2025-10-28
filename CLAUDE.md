# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## ⚠️ CRITICAL RULE: No Unauthorized Changes ⚠️

**NEVER modify code that the user has not specifically requested to be changed.**

**NEVER add features, optimizations, or "improvements" without explicit user direction.**

**You are an ASSISTANT. You do NOT run this show. The user runs this show.**

This means:
- ❌ DO NOT hardcode filters, whitelists, or blacklists unless explicitly asked
- ❌ DO NOT add validation logic, error handling, or "safety checks" unless explicitly asked
- ❌ DO NOT refactor, optimize, or "clean up" code unless explicitly asked
- ❌ DO NOT assume you know what the user wants - ASK FIRST
- ❌ DO NOT make changes "to help" or "to fix issues" - PROPOSE and WAIT for approval
- ✅ DO present options and ask which approach to take
- ✅ DO explain what you're about to change and wait for explicit approval
- ✅ DO implement ONLY what was requested, nothing more

**Violating this rule significantly slows down research progress and wastes time.**

If you think something needs to be changed that wasn't explicitly requested:
1. STOP
2. Explain why you think it needs changing
3. ASK the user if they want you to proceed
4. WAIT for explicit approval

## Project Overview

**REZ/RESDA** is an extended semantic segmentation research project built upon the original IFSeg (Image-free Semantic Segmentation) codebase. While IFSeg demonstrated vision-language model approaches for segmentation without labeled training images, this project has been extensively modified and enhanced.

**Note on Directory Name**: The repository directory is still named `ifseg` for historical reasons (renaming would break existing paths and configurations), but the active project is REZ/RESDA with significant extensions beyond the original IFSeg work.

**Original IFSeg Foundation**: Vision-language model performing semantic segmentation using text prompts and artificial images, based on OFA (One For All) transformer architecture (CVPR 2023).

**REZ/RESDA Extensions**: This codebase includes extensive modifications, additional datasets (aerial/UAV imagery), custom visualization tools, and experimental enhancements beyond the base IFSeg implementation.

## Batch Processing Architecture (ReSDA Pipeline)

### Current Design (2025-10-26)

**Problem:** Reinitializing OFA model for every image is slow and causes CUDA issues.

**Solution:** `resda_batch.py` as main entry point that initializes OFA once, then processes N images.

**Architecture:**
```
resda_batch.py (resda environment)
├── Initialize OFA model ONCE (lines 28-55 from original)
├── Parse args: --test-images, --test-annotations, --num-files
├── Get file lists: GTfiles, SEGfiles
└── For each image pair:
    ├── Stage 1: Import and call objective() from resda_baseline_aero.py
    │   └── Save stage1_baseline.npz
    ├── Stage 2: subprocess → conda run -n tosam python resda_sam_instance.py
    │   └── Save stage2_instances.npz
    └── Stage 3: subprocess → conda run -n tosam python resdaEX_merge.py
        └── Save merged visualization
```

**Key Design Decisions:**

1. **Model Initialization Location (TEMPORARY - revisit for cleanup)**
   - **Decision:** Keep model init code in BOTH files (Option 1)
   - `resda_batch.py`: Initializes model for batch processing
   - `resda_baseline_aero.py`: Also has init code for standalone testing
   - Uses `if __name__ == "__main__"` to skip init when imported

2. **Alternative (for future cleanup):**
   - **Option 2:** Remove ALL init from `resda_baseline_aero.py`
   - Make it pure functions only (objective, get_classes, etc.)
   - Always run via `resda_batch.py`, never standalone
   - Cleaner separation but loses quick debugging capability

3. **Why Not Batch Per Stage?**
   - ❌ Rejected: Load all images in stage 1, then all in stage 2, etc.
   - Reason: Would still require reloading OFA for each stage 1 image
   - Current approach: Full pipeline per image = model loaded once

**Required Arguments:**
- `--test-images`: Path to test image directory (e.g., `./aeroscapes/images/testing`)
- `--test-annotations`: Path to annotation directory (e.g., `./aeroscapes/annotations/testing`)
- `--num-files`: Number of images to process (default: 1)

**Usage:**
```bash
python resda_batch.py \
    --test-images ./aeroscapes/images/testing \
    --test-annotations ./aeroscapes/annotations/testing \
    --num-files 5
```

**TODO - Cleanup Decisions:**
- [ ] Decide whether to keep standalone capability in resda_baseline_aero.py
- [ ] If removing, migrate to Option 2 (pure functions only)
- [ ] If keeping, document clear separation between batch and standalone modes

### Implementation Status (2025-10-26 Evening)

**✅ COMPLETED:**

1. **resda_batch.py created and working**
   - Initializes OFA model once
   - Processes N images through full pipeline
   - Tested successfully with 1 and 2 images
   - Increments properly through file list

2. **resda_baseline_aero.py refactored**
   - Main loop wrapped in `if __name__ == "__main__":`
   - Functions importable without running loop
   - Maintains standalone capability for debugging

3. **resda_sam_instance.py enhanced**
   - Added command-line argument support: `python resda_sam_instance.py <image_path>`
   - Collects SAM masks during processing (not just bboxes)
   - Saves masks to .npz: `sam_masks = {class_name: [mask1, mask2, ...]}`
   - Can run standalone for faster iteration (skip slow stage 1)

4. **resdaEX_merge.py fixed**
   - Now uses actual SAM pixel-level masks instead of rectangular bboxes
   - Paints segmented pixels: `merged_seg[mask] = class_idx`
   - Proper instance segmentation with object boundaries

5. **Pipeline tested end-to-end**
   - Stage 1 → Stage 2 → Stage 3 working
   - SAM masks correctly merged onto baseline
   - Person detection working with pixel-level precision
   - Visualizations show proper segmentation boundaries

**🔧 KNOWN ISSUES:**

1. **Qwen prompt needs refinement**
   - Currently detects some unwanted classes
   - May benefit from better boundedness criteria
   - TODO: Review and optimize prompt for aerial imagery

**📝 STANDALONE TESTING WORKFLOWS:**

**Test stage 2 + 3 only (skip slow stage 1):**
```bash
# Use existing stage1_baseline.npz from previous run
conda activate tosam
python resda_sam_instance.py ./aeroscapes/images/testing/000001_001.jpg
python resdaEX_merge.py ./output/stage1_baseline.npz ./output/stage2_instances.npz
```

**Test full pipeline:**
```bash
python resda_batch.py \
    --test-images ./aeroscapes/images/testing \
    --test-annotations ./aeroscapes/annotations/testing \
    --num-files 1
```

**Test individual stages:**
```bash
# Stage 1 (standalone)
python resda_baseline_aero.py

# Stage 2 (standalone)
conda activate tosam
python resda_sam_instance.py <image_path>

# Stage 3 (standalone)
conda activate tosam
python resdaEX_merge.py ./output/stage1_baseline.npz ./output/stage2_instances.npz
```

## Setup and Installation

### Environment Setup
```bash
# Install PyTorch with CUDA 11.6
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116

# Install mmsegmentation
pip install openmim
mim install mmcv-full==1.6.2
git clone https://github.com/open-mmlab/mmsegmentation.git
cd mmsegmentation && git checkout v0.28.0 && pip install -v -e .
cd ..

# Install custom fairseq and dependencies
pip install -e ./custom_fairseq/
pip install -r requirements.txt
```

### Download Pretrained Model
Download OFA-Base checkpoint (`ofa_base.pt`) from [OFA checkpoints](https://github.com/OFA-Sys/OFA/blob/main/checkpoints.md) and place in project root.

## Dataset Preparation

REZ/RESDA encodes images as base64 strings in TSV files for efficient processing (inherited from IFSeg). Each dataset requires conversion from standard format to TSV.

### Convert Datasets to TSV Format

**COCO-Stuff:**
```bash
# Download from https://cocodataset.org
# Use notebooks to generate TSV files:
jupyter notebook convert_segmentation_coco.ipynb  # Fine-grained (171 classes)
jupyter notebook convert_segmentation_coco_unseen_split.ipynb  # Unseen split (15 classes)
```

**ADE20K:**
```bash
# Download from https://groups.csail.mit.edu/vision/datasets/ADE20K/
jupyter notebook convert_segmentation_ade.ipynb  # Generates validation.tsv
```

**Additional Aerial/UAV Datasets (REZ/RESDA Extensions):**
These datasets and conversion scripts are additions to the original IFSeg codebase:
- `convert_segmentation_uavid.py` - UAVid dataset
- `convert_segmentation_aeroscapes.py` - Aeroscapes dataset
- `convert_segmentation_udd6.py` - UDD6 dataset
- `convert_segmentation_droneSeg.py` - DroneSeg dataset
- `convert_segmentation_iSAID.py` - iSAID dataset
- `convert_segmentation_aid.py` - AID dataset

**Expected Directory Structure:**
```
ifseg/  # Legacy directory name (actually REZ/RESDA project)
├── dataset/
│   ├── coco/
│   │   ├── unseen_val2017.tsv
│   │   └── fineseg_refined_val2017.tsv
│   └── ade/
│       └── validation.tsv
├── ofa_base.pt  # Pretrained checkpoint
└── run_scripts/
```

## Training and Inference

### Running Experiments

All training scripts are in `run_scripts/` and use PyTorch Distributed Data Parallel:

**Image-Free Segmentation on COCO Unseen (15 categories):**
```bash
bash run_scripts/coco_unseen.sh
```

**Image-Free Segmentation on ADE20K (150 categories):**
```bash
bash run_scripts/ade.sh
```

**Image-Free Segmentation on COCO-Stuff (171 categories):**
```bash
bash run_scripts/coco_fine.sh
```

### Key Training Parameters

Edit parameters in shell scripts before running:

- `GPUS_PER_NODE`: Number of GPUs (default: 2-4)
- `num_seg_tokens`: Number of segmentation classes (e.g., 15, 150, 171)
- `category_list`: Comma-separated class names (e.g., `'wall, building, sky, floor, tree'`)
- `artificial_image_type`: Synthetic image generation method (e.g., `rand_k-1-33`)
- `batch_size`: Batch size per GPU
- `max_epoch`: Training epochs
- `lr`: Learning rate (typically `5.0e-5`)
- `patch_image_size`: Input image resolution (default: 512)

### Custom Category Lists

To train on novel categories, modify the script:
```bash
num_seg_tokens=10  # Number of classes + 1
category_list='sky, tree, road, grass, sidewalk, car, building, person, water, mountain'
```

## Architecture

### Core Components

**OFA-Based Architecture** (`ofa_module/`)
- Base model: OFA (One For All) transformer encoder-decoder
- Modified with segmentation-specific tokens and projection layers
- ResNet feature extractor with iterative refinement

**Task Implementation** (`tasks/mm_tasks/segmentation.py`)
- Registers `segmentation` task with fairseq framework
- Handles data loading, preprocessing, and evaluation
- Configures prompt templates and category mappings

**Dataset Handling** (`data/mm_data/segmentation_dataset.py`)
- Loads TSV files with base64-encoded images
- Applies mmsegmentation transforms (resize, flip, normalize)
- Generates artificial images for image-free training
- Creates text prompts with category lists

**Segmentation Criterion** (`criterions/seg_criterion.py`)
- Custom loss function for segmentation
- ResNet-based iterative refinement (`resnet_iters`, `resnet_topk`)
- Supports unsupervised/image-free mode
- Handles upsampling logits to image resolution

**Trainer** (`trainer.py`)
- Modified fairseq Trainer with segmentation-specific hooks
- Handles distributed training and checkpointing
- Manages optimizer and learning rate scheduling

### Training Flow

1. **Initialization**: Load OFA pretrained checkpoint, freeze embeddings and ResNet
2. **Data Loading**: Read TSV, decode base64 images, generate artificial images
3. **Forward Pass**:
   - Text prompt → encoder
   - Artificial image → ResNet features
   - Decoder generates segmentation tokens
   - Iterative refinement with ResNet top-k filtering
4. **Loss Computation**: Cross-entropy on segmentation token predictions
5. **Evaluation**: Generate segmentation maps, compute mIoU on validation set

### Key Flags

- `--unsupervised-segmentation=true`: Enable image-free mode
- `--artificial-image-type=rand_k-1-33`: Random synthetic images
- `--freeze-entire-resnet=true`: Freeze ResNet backbone
- `--init-seg-with-text=true`: Initialize seg tokens from text embeddings
- `--decoder-type=surrogate`: Use surrogate decoder
- `--upscale-lprobs=true`: Upsample logits to original resolution

## Visualization

Generate web-based visualizations of segmentation results:

```bash
jupyter notebook visualize_segmentation_web.ipynb
```

Download pretrained checkpoint for visualization: [Google Drive Link](https://drive.google.com/file/d/167sIrrSsBTRQlrVHYMKYoWA5A9r04eAD/view?usp=sharing)

**REZ/RESDA Custom Visualization Tools:**
The project includes extensive custom visualization notebooks for different datasets:
- `visualize_segmentation_web-18a - UAVid.ipynb` - UAVid dataset visualization
- `visualize_segmentation_web-18b - droneSeg.ipynb` - DroneSeg dataset visualization
- `visualize_segmentation_web-18c - aeroscapes.ipynb` - Aeroscapes dataset visualization
- `visualize_segmentation_web-18d - udd6.ipynb` - UDD6 dataset visualization
- Various experimental notebooks with tiling, weighting, and uncertainty quantification approaches

## Distributed Training Configuration

All scripts use `torch.distributed.launch`:
- Master address/port configured in script (default: `127.0.0.1:9999`)
- Set `GPUS_PER_NODE` to control parallelism
- Gradients synchronized across GPUs with DDP
- FP16 training enabled by default

## Checkpointing

- Checkpoints saved to `./experiment_outputs/<session_name>/`
- Best checkpoint based on `mIoU` metric
- Keeps last epoch and best checkpoint only
- Load checkpoint with `--restore-file=<checkpoint.pt>`

## Common Issues

**Out of Memory**: Reduce `batch_size` or `patch_image_size` in training scripts

**Dataset Loading Errors**: Ensure TSV files are generated correctly with proper base64 encoding

**CUDA Errors**: Match PyTorch version (1.12.1) with CUDA toolkit version (11.6)

## Code Organization

- `train.py` - Main training entry point
- `trainer.py` - Custom trainer with segmentation hooks
- `ofa_module/` - OFA model implementation
- `tasks/mm_tasks/` - Task definitions (segmentation, etc.)
- `data/mm_data/` - Dataset loaders
- `criterions/` - Loss functions
- `utils/` - Helper utilities (BPE, checkpoint management, etc.)
- `custom_fairseq/` - Modified fairseq library
- `run_scripts/` - Training shell scripts
- `convert_segmentation_*.py` - Dataset conversion utilities
- `visualize_segmentation_*.ipynb` - Visualization notebooks

## Project History and Relationship to IFSeg

This codebase began as the IFSeg (Image-free Semantic Segmentation) implementation but has been significantly extended into the REZ/RESDA project. The directory name remains `ifseg` to avoid breaking existing paths, but the project scope has grown considerably beyond the original IFSeg paper.

## Related Documentation

**Original IFSeg References:**
- [OFA GitHub](https://github.com/OFA-Sys/OFA)
- [IFSeg Paper](https://arxiv.org/abs/2303.14396)
- [Fairseq](https://github.com/pytorch/fairseq)
- [mmsegmentation](https://github.com/open-mmlab/mmsegmentation)

**Note**: Many components have been customized beyond the original implementations for REZ/RESDA research objectives.

---

## Current Research Focus: Minority Class Detection in Aerial Imagery (2025-10)

### The Core Problem: Tiling vs No-Tiling Trade-off

**Background**: Aerial imagery segmentation faces a fundamental challenge with perspective - objects far from camera appear compressed. Two approaches exist:

1. **Tiling Approach** (4 tiles per image):
   - ✅ Better attention to each quadrant separately
   - ✅ Successfully detects minority classes (person, bike, car)
   - ❌ Slow (4x processing time)
   - ❌ Leaves visible artifacts at tile boundaries
   - **Status**: Works but not ideal

2. **No-Tiling Approach** (full image):
   - ✅ Fast (1x processing time)
   - ❌ Minority classes suppressed by majority neighbors
   - **Issue**: ResNet refinement destroys minorities through iterative averaging
   - **Status**: Current focus of optimization

### The Minority Class Suppression Problem

**Discovery**: ResNet iterative refinement catastrophically destroys minority class detection.

**Test Case**: Image `000001_001.jpg` (person in white shirt on road)
- Ground truth: 972 person pixels
- With `resnet_iters=25`: Only 4 pixels detected (99.6% loss!)
- With `resnet_iters=0`: ~7,000 pixels detected (over-detection but person visible)

**Root Cause**:
```python
# Lines ~1048-1054 in vsw_aeroscapes_no_tilings_uncertainty.py
for _ in range(resnet_iters):  # 25 iterations
    batch_probs_resnet_topk = batch_probs_resnet[batch_ind, topk_ind]
    batch_probs_resnet = batch_probs_resnet_topk.mean(dim=-2)  # Average with 5 neighbors
```
- Person pixels surrounded by road/vegetation pixels
- After 25 iterations of averaging with dominant neighbors → person signal diluted to zero
- Even `resnet_iters=10` destroyed detection

### Current Solution: Skip ResNet Refinement

**File**: `vsw_aeroscapes_no_tilings_uncertainty.py`

**Key Parameters** (lines 977-980):
```python
resnet_topk = 5
resnet_iters_minority = 0    # Effectively skipped
resnet_iters_majority = 25   # Currently unused
crf_iters = 30               # Experimentally determined optimal value
```

**Results**:
- Person detected: ~7k pixels (vs 972 GT, acceptable over-detection)
- mIoU improved +4 points from crf=10 to crf=30
- mIoU dropped -10 points at crf=40 (person disappeared again)
- **Trade-off**: Mask discontinuities/noise without ResNet smoothing

### Attempted Solutions (Failed)

1. **Selective ResNet refinement with protection mask** (lines 1014-1049)
   - Create mask for high-confidence minority pixels
   - Only refine majority classes
   - **Failed**: Protected person pixels "leaked" to neighbors (972 GT → 37k detected)
   - Averaging still included protected neighbors, causing probability bleeding

2. **Global class weight adjustments**
   - Boost minority classes, dampen majority classes
   - **Problem**: Any weight adjustment affects spatial relationships
   - Changes propagate through ResNet averaging, CRF, and argmax boundaries
   - No clean way to boost one class without affecting neighbors

3. **Simple heuristic detection** (color-based)
   - Detect white/light objects as "person"
   - **Failed spectacularly**: Detected clouds instead of person
   - Proved need for proper object detection

### Next Exploration: Object Detection + Spatial Boosting

**Concept**: Use pre-trained object detector to provide spatial hints to segmentation model.

**Architecture**:
```
Current Pipeline (unchanged):
Image → OFA Model → Probabilities → CRF → Segmentation

Enhanced Pipeline (add 2 steps):
Image → [NEW: Detector finds person bbox]
     ↓
     → OFA Model → Probabilities
     ↓
     → [NEW: Boost person probs inside bbox]
     ↓
     → CRF → Segmentation
```

**Detector Options**:

1. **Faster R-CNN** (torchvision - already installed)
   - Pre-trained on COCO (80 classes including person)
   - NOT zero-shot (fixed vocabulary)
   - No library upgrades needed
   - **Test**: `test_fasterrcnn_detection.py`
   - **Status**: Ready to test

2. **OWL-ViT** (requires transformers upgrade)
   - True zero-shot detection (text queries)
   - Can detect any class via prompt
   - **Problem**: Library dependency conflicts
     - Requires transformers 4.35+
     - Conflicts with numpy 1.23.5 (tensorflow needs <2.0)
     - Created `resda_test` conda environment to test
   - **Test**: `test_owlvit_detection.py`
   - **Status**: Blocked by dependency hell

3. **Grounding DINO** (requires additional packages)
   - State-of-the-art zero-shot detection
   - More complex setup
   - **Status**: Not attempted yet

### Integration Plan (If Detection Works)

Add ~30 lines to `vsw_aeroscapes_no_tilings_uncertainty.py`:

**Before model inference** (line ~940):
```python
# Detect minority objects
person_boxes = detect_objects(pil_image, ["person"])  # Returns [(x1,y1,x2,y2)]
```

**After computing logits** (line ~1018):
```python
# Create spatial boost map
if len(person_boxes) > 0:
    spatial_boost = torch.ones((H, W), device=batch_logits.device)
    for (x1, y1, x2, y2) in person_boxes:
        y1_px, y2_px = int(y1 * H), int(y2 * H)
        x1_px, x2_px = int(x1 * W), int(x2 * W)
        spatial_boost[y1_px:y2_px, x1_px:x2_px] = 5.0  # Boost factor

    person_idx = classes.index('person')
    batch_logits[:, :, person_idx] *= spatial_boost.flatten()
```

**Benefits**:
- Spatial boosting only where detector found object
- No global weight adjustments (avoids neighbor propagation issues)
- Could potentially re-enable ResNet refinement with spatial protection
- Detector-agnostic (works with Faster R-CNN, OWL-ViT, or others)

### Key Test Images

**Primary**: `./aeroscapes/images/testing/000001_001.jpg`
- Person in white shirt walking on road
- Surrounded by vegetation, greenhouses, mountains
- Ground truth: `./aeroscapes/annotations/testing/000001_001.png` (972 person pixels)

**Outputs**:
- Tiled (resnet=25): `./aeroscapes/segmented/000001_001tiledSeg.jpg` - smooth, NO person
- Untiled (resnet=0): `./vsw/segmented/000001_001UNtiledSeg.jpg` - noisy, person visible

### Current Status & Next Steps

**As of 2025-10-21 (Latest)**:

## ✅ BREAKTHROUGH: Dual-Pass Merge Approach WORKING!

**Successfully implemented Option C architecture** - simple merge without second segmentation:

### Architecture
1. **Pass 1 - Baseline Segmentation**:
   - GPT-4o via `get_classes()` identifies ALL significant classes in image
   - Weight identified classes at 100 via `weight_booster()`
   - Run `objective()` with ResNet=25, CRF=10
   - Returns `(mIoU, seg_baseline)` - full segmentation with all classes
   - **Issue**: Minorities suppressed by ResNet refinement

2. **Analyze Baseline for Missing Classes**:
   - Count pixels for each LM-expected class in baseline segmentation
   - Classes with <100 pixels are considered "suppressed/missing"
   - **Data-driven approach**: Only target classes that were actually suppressed

3. **Targeted Bbox Detection**:
   - Qwen-VL (`get_bounding_boxes()`) detects ONLY missing/suppressed classes
   - Returns `{class_name: [(x1,y1,x2,y2), ...]}`
   - **No second segmentation needed** - just trust Qwen's bboxes!

4. **Merge**:
   - `merge_segmentations()` simply paints bbox regions onto baseline
   - Bbox class IDs override baseline in those rectangular regions
   - Saves to `./vsw/segmented/{filename}_MERGED.jpg`

### Results
✅ **Person detected!** (was suppressed to 4 pixels in baseline)
✅ **Car detected!** (wasn't even found in tiled approach)
✅ Majority classes preserved (sky, clouds, vegetation, road, hill)
✅ Simple, clean, no second model inference

### Qwen-VL Limitations Discovered
- **500 Internal Server Error** on complex scenes with many objects
- **Solution**: Limit to top 10 most prominent objects in prompt
- Testing to find upper limit of object count

### Current Challenges
1. **Road overflow**: Road class dominates and bleeds into neighbors during ResNet
   - Need generalized solution for dominant majority classes
   - Separate issue from minority detection

2. **Bbox count limit**: Testing Qwen's max object capacity
   - Currently limited to 10 objects in prompt
   - Finding optimal balance between coverage and reliability

**Key Files**:
- `vsw_bb_aeroscapes_no_tilings.py` - **WORKING** dual-pass merge implementation
- `vsw_bb_aeroscapes_no_tilings_blt.py` - Bbox limit testing (currently set to 10 objects, with data-driven missing class detection)
- `vsw_bb_aeroscapes_no_tilingsTEST.py` - Archived: bbox-only segmentation experiments
- `vsw_aeroscapes_no_tilings.py` - Original clean baseline
- `vsw_aeroscapes_no_tilings_uncertainty.py` - DEPRECATED uncertainty approach
- `visualize_bboxes.py` - Utility to draw bboxes on images
- `test_gpt4o_bbox.py` - Test VLM bbox accuracy (supports GPT-4o and Qwen-VL)

**Key Functions**:
- `get_classes(f)` - GPT-4o identifies significant classes (Pass 1)
- `get_bounding_boxes(f, target_classes=None)` - Qwen-VL detects bboxes for specific classes
- `merge_segmentations(seg_baseline, bboxes, classes)` - Paints bbox regions onto baseline
- `objective(GTfile, file, weights, classes, gt_classes)` - Returns `(mIoU, segmentation_map)`

### Important: Library Constraints

**DO NOT upgrade libraries in main `ifseg` environment** - the codebase depends on specific old versions:
- transformers 4.18.0.dev0 (newer versions break OFA)
- numpy 1.23.5 (newer breaks tensorflow)
- PyTorch 1.12.1+cu116

**For testing new detectors**: Use `resda_test` conda environment (cloned from ifseg)

### Key Insights Learned

1. **ResNet refinement is binary**: Either destroys minorities (any iters > 0) or preserves them (iters = 0)
2. **CRF alone insufficient**: Provides some smoothing but creates discontinuities
3. **Global weight adjustments fail**: Cannot isolate minority boost without spatial propagation
4. **Tiling naturally helps minorities**: Smaller context = fewer dominant neighbors
5. **Spatial boosting is promising**: Detection-based hints avoid global weight issues

### Related Classes Being Studied

**Minority classes** (small objects that get suppressed):
- person, human, man, woman
- bike, bicycle
- car, vehicle
- dog, animal
- drone, helicopter, boat

**Majority classes** (tend to over-predict):
- road, highway, street (biggest offender)
- vegetation, grass, tree
- sky, clouds

**Ground truth classes for Aeroscapes**:
- Vegetation, Road, Person, Obstacle, Construction, Bike, Car, Sky, Drone, Animal, Boat, Background

### LM-Based Class Identification

The pipeline uses GPT-4o to identify classes in images (`get_classes()` function):
- Analyzes image to suggest relevant classes
- Returns `boosted_classes` and `override_weights`
- These feed into `weight_booster()` to create weight vector (1D: length 150)
- **Important**: This system already exists and works

### Spatial Bounding Box Weights (NEW APPROACH)

**Function**: `get_bbox_weights(f, img_h, img_w, classes)` in `vsw_bb_aeroscapes_no_tilings.py`

**Returns**:
- `boosted_classes`: List of detected bounded object classes
- `bboxes`: Dict mapping class names to bbox coordinates `{class: [(x1,y1,x2,y2), ...]}`
- `bbox_weights`: **Spatial weight map** [H, W, C] (720 × 1280 × 150)

**Key Difference from Standard Weights**:
- Standard weights: 1D vector [C] - same weight for class across entire image
- Bbox weights: 3D tensor [H, W, C] - weight varies by pixel location AND class
  - Inside person bbox → person class = 100, all others = 0
  - Inside car bbox → car class = 100, all others = 0
  - Outside all bboxes → all classes = 0

**Prompt Strategy**:
- Ask for BOUNDED objects only (person, car, building, tree)
- Exclude unbounded regions (sky, road, vegetation when covering large areas)
- GPT-4o returns exact pixel coordinates: `boosted_classes = ['person'], bounding_boxes = {'person': [(652, 470, 682, 530)]}`

**Coordinate Accuracy**:
- Claude (Sonnet 4.5) has ~80px x-offset issue when providing bboxes
- GPT-4o provides accurate coordinates (verified via `test_gpt4o_bbox.py`)
- Ground truth for test image: person at (652, 470) to (682, 530)

**TODO**: How to apply [H, W, C] weights to logits during inference
- Current `objective()` function expects 1D weights [C]
- Need to figure out how to broadcast/apply spatial weights to batch_logits

---

## SAM Integration for Pixel-Perfect Instance Segmentation (2025-10)

### Understanding SAM's Role

**Critical Distinction**: SAM (Segment Anything Model) is purely spatial/geometric - it does NOT work with class lists or semantic understanding.

**How SAM Works in This Pipeline:**
1. **Input**: Bounding box coordinates `[x1, y1, x2, y2]` (no class information)
2. **Output**: Binary mask of pixels forming a coherent object within/around that box
3. **SAM has no idea WHAT it's segmenting** - it just knows "segment whatever coherent thing is in this box"
4. The class label (e.g., "person") is metadata from Qwen for tracking purposes only

**Why SAM Cannot Handle Unbounded Regions:**
- Cannot provide a meaningful bounding box for "sky", "road", or "vegetation" as continuous regions
- A box around "sky" would just be the top half of the image
- SAM would segment texture discontinuities, not semantic regions
- Unbounded regions require **semantic segmentation** (OFA pipeline), not instance segmentation (SAM)

### Enhanced Three-Stage Pipeline Architecture

**Combining OFA + Qwen + SAM for Optimal Results:**

```
Stage 1: OFA Baseline Segmentation (handles unbounded regions)
├─ Qwen-VL identifies ALL classes: sky, clouds, road, vegetation, person, car, etc.
├─ Weight identified classes via weight_booster()
├─ OFA segments entire image with ResNet=25, CRF=10
├─ Result: Complete semantic coverage of unbounded regions
└─ Issue: Minorities (person, car) suppressed by ResNet refinement

Stage 2: Qwen + SAM Instance Segmentation (rescues bounded objects)
├─ Qwen-VL provides bboxes for discrete objects: person, car, sign, building, etc.
├─ SAM generates pixel-perfect masks (NOT rectangles!)
├─ Result: Accurate instance boundaries for each bounded object
└─ Advantage: Exact object contours vs. rectangular bbox overlay

Stage 3: Intelligent Merge
├─ Start with OFA baseline segmentation (complete scene coverage)
├─ Overlay SAM masks for bounded objects
├─ SAM pixels override baseline ONLY within actual object shape
└─ Result: Precise instance boundaries + complete semantic coverage
```

**Key Advantage Over Rectangular Bbox Merge:**

Instead of painting rectangular boxes onto baseline (includes background pixels), SAM provides exact object contours. When merging "person" pixels, only the actual person-shaped region overrides the baseline, not the rectangular area around them.

**Division of Labor:**
- **OFA**: Full semantic segmentation of continuous regions (sky, road, vegetation, water, etc.)
- **Qwen-VL**: Class identification + bounding box localization for discrete objects
- **SAM**: Pixel-perfect instance masks with accurate object boundaries

**Bounded vs. Unbounded Objects:**

| Type | Examples | Best Approach | Why |
|------|----------|---------------|-----|
| **Bounded** | person, car, building (specific), tree (individual), sign, boat | Qwen bbox → SAM mask | Discrete instances with clear boundaries |
| **Unbounded** | sky, clouds, vegetation (region), road (surface), water, hill | OFA semantic seg | Continuous regions without instance boundaries |

### Implementation Files

**Active Development:**
- `qsam_seg/qsam.py` - Qwen + SAM pipeline (bounding box detection + instance segmentation)
  - Input: Image path
  - Qwen detects bounded objects and returns bboxes
  - SAM generates pixel-perfect masks for each bbox
  - Outputs:
    - `qsam_seg/bboxes/{image}_bboxes.jpg` - Visualization of Qwen detections
    - `qsam_seg/segmented/{image}_segmented.jpg` - SAM segmentation overlay

**Integration Target:**
- `vsw_bb_aeroscapes_no_tilings.py` - Will be enhanced to replace rectangular bbox merge with SAM masks
  - Currently: Paints rectangular regions onto baseline
  - Enhanced: Overlay SAM pixel-perfect masks onto baseline

### SAM Model Setup

**Checkpoint**: `checkpoint/sam_vit_h_4b8939.pth` (ViT-H variant)

**Download**: https://github.com/facebookresearch/segment-anything#model-checkpoints

**Installation**:
```bash
pip install segment-anything
```

**Known Issue**: PyTorch 2.5.0 CUDA compatibility
```python
import torch
torch.backends.cuda.enable_cudnn_sdp(False)  # Fix for CUDA unknown error
```

### Next Steps

1. Verify SAM mask quality on test images
2. Integrate SAM into dual-pass merge pipeline
3. Compare results: rectangular bbox merge vs. SAM pixel-perfect merge
4. Evaluate mIoU improvement from accurate instance boundaries
5. Test on minority classes: person, car, bike, animal, boat

---

## Dual-Pass Qwen Pipeline (2025-10-24)

### Working Implementation: vsw_aero_no_tilings_allQwen.py

**Architecture: Two Independent Qwen Passes**

```
Pass 1: Class Identification + Baseline Segmentation
├─ Qwen-VL identifies ALL classes in image (bounded + unbounded)
├─ Returns: boosted_classes, override_weights
├─ OFA generates baseline segmentation with all identified classes
└─ Result: Complete semantic segmentation (sky, clouds, road, vegetation, etc.)

Pass 2: Bounded Object Detection (Independent)
├─ Qwen-VL finds BOUNDED objects with strict criteria
├─ Returns: bounding_boxes {class: [(x1,y1,x2,y2), ...]}
├─ Filter: Remove oversized boxes (>40% image area)
├─ Filter: Remove overlapping boxes (keep smaller)
└─ Result: Clean bboxes for discrete objects (person, car, sign, building)

Merge
├─ Start with baseline segmentation
├─ Paint bbox regions onto baseline (override baseline pixels)
└─ Result: Unbounded regions from baseline + precise bounded objects from bboxes
```

**CRITICAL: The Two Passes Are INDEPENDENT**

- ❌ DO NOT analyze baseline to find "missing classes"
- ❌ DO NOT filter Pass 1 classes before Pass 2
- ❌ DO NOT pass "target_classes" to bbox detection based on baseline
- ✅ Let Qwen independently decide what bounded objects to detect
- ✅ The two passes serve different purposes and should not be conflated

**Pass 1 Prompt**: Identifies all visible classes (from 150-class list) for full scene understanding

**Pass 2 Prompt**: Finds bounded objects using strict criteria (see below)

### Bounded Object Criteria (Pass 2)

Qwen must verify ALL criteria before returning a bounding box:

1. **Discrete instance**: Bbox contains 99% of a single object/class type
2. **Fills >70% of bbox area**: Object is not sparse, elongated, or scattered
3. **Clearly identifiable boundaries**: Object has edges separating it from surroundings
4. **Localized (CRITICAL)**: Occupies contiguous spatial area, not scattered across image

**Examples:**

✅ **Boundable:**
- person (discrete, compact, clear edges, localized)
- car (discrete, fills bbox well, clear edges, localized)
- individual building (discrete structure, compact, clear edges)
- sign (discrete, fills bbox, clear edges, localized)

❌ **Not Boundable:**
- field/crop (fails localized - too dispersed)
- fence (fails fill criterion - too thin/elongated)
- shadow (not 99% single type - mixed with underlying surfaces)
- scattered vegetation (fails localized - not contiguous)
- road (continuous surface, no discrete boundaries)

### Post-Processing: Overlap Filtering

**Function**: `filter_overlapping_bboxes(bboxes, img_w, img_h, max_area_pct=40)`

Removes problematic boxes:
1. **Size filter**: Remove boxes >40% of image area
2. **Overlap filter**: When boxes overlap, keep smaller box and remove larger box

**Why this works:**
- Large boxes are usually unbounded regions incorrectly identified as bounded (crop, hill, structure)
- Smaller boxes are usually discrete objects we want (person, car, sign)
- Overlap removal prevents occlusion and duplicate predictions

### Implementation Files

**Active:**
- `vsw_aero_no_tilings_allQwen.py` - ✅ **WORKING** dual-pass pipeline (Qwen for both passes)
  - Pass 1: Qwen class identification → OFA baseline
  - Pass 2: Qwen bbox detection → filter → merge
  - Outputs:
    - `./vsw/bboxes/{image}_bboxes.jpg` - Bounding box visualization
    - `./vsw/segmented/{image}_MERGED.jpg` - Final merged segmentation

**Deprecated:**
- `vsw_bb_aeroscapes_no_tilings.py` - GPT-4o for Pass 1, Qwen for Pass 2
- `vsw_bb_aeroscapes_no_tilings_blt.py` - Bbox limit testing (10 object limit)

### Key Functions

```python
# Pass 1: Class identification
get_classes(f) → (boosted_classes, override_weights)
weight_booster(boosted_classes, classes, override_weights) → weights
objective(GTf, f, weights, classes, gt_classes) → (mIoU, seg_baseline)

# Pass 2: Bbox detection
get_bounding_boxes(f, target_classes=None) → (bboxes, detected_classes)
filter_overlapping_bboxes(bboxes, img_w, img_h, max_area_pct=40) → filtered_bboxes

# Merge
merge_segmentations(seg_baseline, bboxes, classes) → merged_seg
```

### Lessons Learned

1. **Don't conflate the passes**: Pass 1 identifies all classes for scene understanding. Pass 2 finds bounded objects independently. They serve different purposes.

2. **Pixel counting is a trap**: Analyzing baseline to find "missing classes" creates coupling between passes and breaks when classes are present but poorly segmented.

3. **Let Qwen decide**: Qwen with proper criteria is better at identifying bounded vs unbounded objects than hardcoded lists or pixel-based heuristics.

4. **Overlap filtering is essential**: Qwen sometimes returns huge boxes for unbounded regions. Size + overlap filtering removes these while keeping discrete objects.

5. **Strict boundedness criteria work**: The 4-criteria prompt (discrete, >70% fill, clear boundaries, localized) significantly reduces unbounded region false positives.

---

## Medical Imaging Domain Adaptation (Theoretical Discussion - 2025-10-24)

### Transfer Feasibility: Aerial → Medical

**System Strengths That Transfer:**
- Dual-pass architecture (semantic context + instance precision)
- Bounded vs unbounded distinction (organs/tissues vs tumors/lesions)
- Zero-shot capability via VLMs (no labeled training data needed)

**Critical Challenges:**
1. **VLM Performance on Medical Images:**
   - NOT trained on raw DICOM files (trained on web images)
   - May miss subtle density/contrast differences critical in medical imaging
   - Modality-specific knowledge gaps (MRI sequences, CT windows)
   - Clinical accuracy requirements (cannot afford hallucinations)

2. **Bounded vs Unbounded Redefinition:**
   - Bounded: tumors, lesions, nodules, polyps, cysts
   - Ambiguous: organs (liver, kidney - bounded but large/irregular)
   - Unbounded: tissue types (gray matter, white matter, CSF)
   - Medical challenge: infiltrative tumors, diffuse disease (no clear boundaries)

3. **Bounding Box Criteria Adaptation:**
   - Many medical structures are infiltrative (no clear boundaries)
   - Lesions can be scattered (multiple metastases)
   - Organs have irregular shapes
   - Would need relaxed criteria (60% fill, softer boundaries)

### Latest Research (2024-2025)

**Vision-Language Models for Medical Segmentation:**
- Meta-analysis: Dice coefficient 0.73 (0.68-0.78) for VLM-based segmentation
- Key finding: Image features often dominate over language prompts
- Transfer learning from general VLMs shows promise but needs finetuning

**SAM Adaptations:**
- **MedSAM**: 1.57M image-mask pairs, 10 modalities, DSC 94.0-98.4% depending on task
- **MedSAM2 (2025)**: 3D volumes + video, ~450k volumes, memory-attention for slice neighbors
- **SAM-Med2D**: SOTA for 2D cross-modal segmentation
- **Critical limitation**: All struggle with weak boundaries/low contrast (common in medical)

**Zero-Shot Foundation Models:**
- **VISTA3D (NVIDIA)**: SOTA for 3D auto (127 classes) + interactive segmentation
- Comprehensive review: Zero-shot SAM "impressive for some, moderate-poor for others"
- Retrieval-augmented approaches (DINOv2 + SAM2) show promise

### Key Research Links

1. Vision-language foundation models review: https://pmc.ncbi.nlm.nih.gov/articles/PMC12411343/
2. VLM medical image analysis comprehensive review: https://arxiv.org/html/2506.18378v1
3. MedSAM GitHub: https://github.com/bowang-lab/MedSAM
4. MedSAM2 explained: https://learnopencv.com/medsam2-explained/
5. MedSAM Nature paper: https://www.nature.com/articles/s41467-024-44824-z
6. Zero-shot foundation models review: https://pmc.ncbi.nlm.nih.gov/articles/PMC12209621/
7. SAM.MD zero-shot capabilities: https://arxiv.org/abs/2304.05396
8. Retrieval-augmented few-shot: https://arxiv.org/html/2408.08813v1
9. Transfer learning VLMs to medical: https://github.com/naamiinepal/medvlsm
10. SAM4MIS comprehensive survey: https://github.com/YichiZhang98/SAM4MIS

### Relevance to Current System

**Direct Parallels:**
- Medical research uses same dual-pass approach (VLM context + SAM precision)
- Bounded vs unbounded distinction validated in medical domain
- Zero-shot philosophy is cutting-edge in medical imaging research

**Potential Advantages:**
- Text-prompted segmentation aligns with medical VLM trends
- Overlap filtering could address SAM's weak boundary problem
- Dual-pass independence avoids coupling issues in medical research

**Remaining Challenges:**
- Medical VLMs still rely heavily on image features over language
- Low contrast boundaries problematic (MRI, ultrasound)
- Regulatory validation requirements for clinical use

---

## ReSDA Directory Structure and Multi-Environment Pipeline (2025-10-25)

### Directory Organization

**~/resda/** - Clean active development directory (selective copy from ifseg)

**What was copied:**
- Core pipeline: `vsw_aero_no_tilings_allQwen.py`
- OFA infrastructure: `ofa_module/`, `custom_fairseq/`, `utils/`, `tasks/`, `data/`, `criterions/`, `models/`
- Support files: `trainer.py`, `classColor.txt`, `crf.py`
- Checkpoints: `ofa_base.pt` (~2.5GB)
- SAM integration: `qsam_seg/`
- Output directories: `output/bboxes/`, `output/segmented/`
- Datasets: Symlinked to `~/ifseg/aeroscapes` (saves disk space)

**~/ifseg/** - Archive (untouched, contains all experimental work)

**Disk usage:** ~2.5GB (mostly OFA checkpoint; datasets symlinked)

### Multi-Environment Architecture

**Problem:** Dependency conflicts between OFA and SAM
- OFA: PyTorch 1.12.1, transformers 4.18.0.dev0, numpy 1.23.5 (cannot upgrade)
- SAM: PyTorch 2.0+, modern libraries

**Solution:** File-based three-stage pipeline with environment isolation

**Environments:**
- `resda` - OFA baseline segmentation (Stage 1)
- `qwen` - SAM instance segmentation (Stage 2 & 3) - **NOTE: Being renamed**

### Three-Stage Pipeline Architecture

```
Stage 1: OFA Baseline Segmentation (resda env)
├─ Input: Original image
├─ Process: Qwen identifies classes → OFA segments entire scene
├─ Output:
│  ├─ temp/{image}_baseline.npy - Segmentation map [H, W] with class indices
│  └─ temp/{image}_baseline_meta.json - Metadata (dims, classes, etc.)
└─ Handles: Unbounded regions (sky, clouds, road, vegetation, etc.)

Stage 2: SAM Instance Segmentation (qwen/renamed env)
├─ Input: Original image
├─ Process: Qwen detects bounded objects → SAM generates pixel-perfect masks
├─ Output:
│  ├─ temp/{image}_sam_masks.npy - Binary masks [N, H, W] for N objects
│  └─ temp/{image}_sam_meta.json - Metadata (bboxes, classes, etc.)
└─ Handles: Bounded objects (person, car, sign, building, etc.)

Stage 3: Merge Segmentations (qwen/renamed env)
├─ Input: baseline.npy + sam_masks.npy
├─ Process: Pure numpy operations (no model dependencies)
├─ Output:
│  ├─ output/bboxes/{image}_bboxes.jpg - Bbox visualization
│  └─ output/segmented/{image}_merged.jpg - Final segmentation
└─ Merges: Baseline + SAM masks (SAM pixels override baseline)
```

### File Format Specifications

**Segmentation Maps (.npy):**
```python
# baseline_seg.npy
shape: (H, W)
dtype: uint8 or int16
values: Class indices (0-149)

# Save
np.save("temp/image_baseline.npy", seg_array)

# Load
seg = np.load("temp/image_baseline.npy")
```

**SAM Masks (.npy):**
```python
# sam_masks.npy
shape: (N, H, W)  # N = number of detected objects
dtype: bool or uint8
values: Binary masks (True/1 where object present)

# Each mask[i] corresponds to detections[i] in metadata
```

**Metadata (.json):**
```json
// baseline_meta.json
{
  "image_dims": [720, 1280],
  "classes": ["sky", "clouds", "road", ...],
  "num_classes": 24
}

// sam_meta.json
{
  "image_dims": [720, 1280],
  "detections": [
    {
      "class": "person",
      "bbox": [653, 460, 681, 527],
      "mask_idx": 0
    },
    {
      "class": "car",
      "bbox": [703, 391, 722, 407],
      "mask_idx": 1
    }
  ]
}
```

### Shell Script Orchestrator

**run_pipeline.sh** will execute:
```bash
# Stage 1: OFA baseline
conda run -n resda python stage1_ofa_baseline.py --image $IMAGE

# Stage 2: SAM instances
conda run -n [new_name] python stage2_sam_instances.py --image $IMAGE

# Stage 3: Merge
conda run -n [new_name] python stage3_merge.py --image $IMAGE
```

**Advantages:**
- No dependency conflicts (environments isolated)
- Modular testing (test each stage independently)
- Fault tolerance (if SAM fails, baseline still exists)
- No Python wrapper complexity (bash handles env switching)
- Intermediate files inspectable for debugging

### Implementation Status

**Completed:**
- ✅ resda directory structure
- ✅ Environment isolation strategy
- ✅ File format specification

**In Progress:**
- ⏳ Shell script orchestrator (design phase)
- ⏳ Environment renaming (qwen → [new_name])

**Next Steps:**
1. Finalize environment name
2. Complete run_pipeline.sh
3. Split vsw_aero_no_tilings_allQwen.py into three stages
4. Test each stage independently
5. Test full pipeline end-to-end
