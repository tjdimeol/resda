# ReSDA: Research Extensions to Semantic Dataset Analysis

Extended semantic segmentation pipeline built on IFSeg/OFA foundation, with SAM integration for minority class detection.

## Overview

ReSDA implements a three-stage pipeline that combines:
1. **OFA Baseline Segmentation** - Vision-language model generates initial 150-class segmentation
2. **Qwen+SAM Instance Detection** - Detects bounded objects via bounding boxes and segments with SAM
3. **Merge** - Overlays instance detections onto baseline for final output

---

## Quick Start

```bash
# Run full pipeline on a single image
bash run_pipeline.sh /path/to/image.jpg

# Outputs:
# - ./output/segmented/{filename}UNtiledSeg.jpg          (Stage 1 visualization)
# - ./qsam_seg/bboxes/{filename}_bboxes.jpg              (Stage 2 bbox visualization)
# - ./qsam_seg/segmented/{filename}_segmented.jpg        (Stage 2 SAM visualization)
# - ./output/segmented/{filename}_MERGED.jpg             (Stage 3 final result)
```

---

## Architecture

### Three-Stage Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│ STAGE 1: OFA Baseline Segmentation (resda environment)             │
├─────────────────────────────────────────────────────────────────────┤
│ Input:  Image                                                       │
│ Model:  OFA (vision-language transformer)                           │
│ VLM:    Qwen-VL-Max identifies significant classes                  │
│ Output: - Baseline segmentation [H, W] (150 classes)                │
│         - Labeled visualization (existing)                          │
│         - ./output/stage1_baseline.npz (for pipeline)               │
└─────────────────────────────────────────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────────┐
│ STAGE 2: Qwen+SAM Instance Detection (tosam environment)           │
├─────────────────────────────────────────────────────────────────────┤
│ Input:  Same image                                                  │
│ VLM:    Qwen-VL-Max detects bounding boxes for bounded objects      │
│ Model:  SAM segments regions inside bboxes                          │
│ Output: - Bounding boxes {class: [(x1,y1,x2,y2), ...]}              │
│         - Bbox + SAM visualizations (existing)                      │
│         - ./output/stage2_instances.npz (for pipeline)              │
└─────────────────────────────────────────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────────┐
│ STAGE 3: Merge (tosam environment)                                 │
├─────────────────────────────────────────────────────────────────────┤
│ Input:  - stage1_baseline.npz (baseline segmentation)               │
│         - stage2_instances.npz (bboxes)                             │
│ Logic:  Paint bbox regions onto baseline (simple numpy assignment)  │
│ Output: - ./output/segmented/{filename}_MERGED.jpg                  │
│         - ./output/segmented/{filename}_MERGED.npy                  │
└─────────────────────────────────────────────────────────────────────┘
```

### Multi-Environment Setup

**Why two environments?**
- OFA requires old PyTorch 1.12.1 + transformers 4.18.0
- SAM requires modern PyTorch 2.0+
- Incompatible dependency trees

**Solution:** File-based pipeline with `conda run`

**Environments:**
- `resda` - OFA baseline (PyTorch 1.12.1, fairseq, mmseg)
- `tosam` - SAM + Qwen-VL (PyTorch 2.0+, segment-anything, openai)

---

## File Structure

### Pipeline Stages
```
resda_baseline_aero.py      - Stage 1: OFA baseline segmentation (aero dataset)
resda_sam_instance.py       - Stage 2: Qwen bbox + SAM instance segmentation
resdaEX_merge.py            - Stage 3: Merge baseline + instances
run_pipeline.sh             - Orchestrates all three stages with conda run
```

### Utilities (Shared Across Stages)
```
resda_utils.py                  - Universal helper functions
  ├── read_class_colors()       - Load class→color mappings
  ├── dataset_cmap()            - Generate colored segmentation from indices
  ├── ensure_uint8_rgb()        - Image format conversion
  └── place_labels()            - Add text labels to segmentation

resda_dataset_transforms.py    - Dataset-specific transformers
  ├── AeroscapesTransformer     - Maps 150→12 classes for Aeroscapes
  ├── UAVidTransformer          - (STUB) TODO: implement
  ├── DroneSegTransformer       - (STUB) TODO: implement
  ├── UDD6Transformer           - (STUB) TODO: implement
  └── get_transformer()         - Factory function (detects from checkpoint path)
```

### Supporting Files
```
classColor.txt              - 150-class color mappings (universal)
classColorGTaero.txt        - Aeroscapes ground truth mappings (12 classes)
classColorGT_uavid.txt      - UAVid ground truth mappings (TODO: verify exists)
classColorGT_droneSeg.txt   - DroneSeg ground truth mappings (TODO: verify exists)
classColorGT_udd6.txt       - UDD6 ground truth mappings (TODO: verify exists)
```

### Legacy Files (Reference Only)
```
junk.py                     - Extracted baseline code (used to create stage 1)
mergejunk.py                - Extracted merge code (used to create stage 3)
qsam_seg/qsam.py            - Original qsam prototype (copied to stage 2)
vsw_aero_no_tilings_allQwen.py - Original monolithic pipeline (archived)
```

---

## Naming Convention Changes

The ReSDA pipeline simplifies file naming by removing redundant prefixes:

### Removed Prefixes (Now Defaults)
- **`vsw`** (Visual Semantic Weighting) - Removed
  - Extensive prompts and class weighting already baked into all resda code
  - No need to distinguish anymore

- **`no_tilings`** - Removed
  - Default in resda is NO tiling (process full image)
  - Opposite of ifseg where tiling is default
  - If we add tiling later, we'll use `_tiled` suffix

- **`allQwen`** - Removed
  - Qwen-VL-Max is current best VLM (subject to change)
  - Don't encode model choice in filename

### Kept in Names
- **`aero`** - Dataset identifier (kept for now)
  - Identifies which checkpoint is used
  - Currently: `./experiment_outputs/aero/checkpoint.best_mIoU_0.0020.pt`
  - TODO: Make configurable via command-line argument instead of filename

### New Naming Pattern
```
resda_baseline_aero.py      - ReSDA baseline stage for aero dataset
resda_sam_instance.py       - ReSDA SAM instance stage (dataset-agnostic)
resdaEX_merge.py            - ReSDA extended merge (dataset-agnostic)
```

---

## Dataset-Specific Components

### Current Dataset: Aeroscapes

**Checkpoint:** `./experiment_outputs/aero/checkpoint.best_mIoU_0.0020.pt`

**Ground Truth Classes (12):**
- Vegetation, Road, Person, Obstacle, Construction
- Bike, Car, Sky, Drone, Animal, Boat, Background

**Transformer:** `AeroscapesTransformer` in `resda_dataset_transforms.py`
- Maps 150-class segmentation → 12 GT classes for mIoU evaluation
- Uses `classColorGTaero.txt` for color mappings

### Adding New Datasets

To add support for a new dataset (e.g., UAVid):

1. **Train checkpoint** (if not already done)
   ```bash
   bash run_scripts/uavid.sh
   ```

2. **Create baseline script** (copy and modify)
   ```bash
   cp resda_baseline_aero.py resda_baseline_uavid.py
   # Update checkpoint path (line 38)
   ckpt = './experiment_outputs/uavid/checkpoint.best_mIoU_XXX.pt'
   ```

3. **Implement transformer** in `resda_dataset_transforms.py`
   ```python
   class UAVidTransformer:
       def createColorTable(self):
           return {
               'Building': [128, 0, 0],
               'Road': [128, 64, 128],
               # ... UAVid-specific classes
           }
   ```

4. **Update factory function** (should auto-detect from `ckpt` path)
   ```python
   # In get_transformer(), 'uavid' already mapped to UAVidTransformer
   ```

5. **Verify color mapping file exists**
   ```bash
   ls classColorGT_uavid.txt
   ```

---

## Intermediate File Formats

### Stage 1 Output: `./output/stage1_baseline.npz`
```python
{
    'segmentation': np.array([H, W], dtype=uint8),  # Class indices 0-149
    'classes': list[str],                            # 150 class names
    'filename': str,                                 # Original image filename
    'dataset': str                                   # e.g., 'aero'
}
```

### Stage 2 Output: `./output/stage2_instances.npz`
```python
{
    'bboxes': dict[str, list[tuple]]  # {class_name: [(x1,y1,x2,y2), ...]}
}
```
Example:
```python
{
    'person': [(652, 470, 682, 530), (120, 350, 145, 400)],
    'car': [(800, 600, 900, 650)]
}
```

---

## Output Directories

```
./output/
├── segmented/                          # Final merged outputs
│   ├── {filename}UNtiledSeg.jpg       # Stage 1 visualization
│   ├── {filename}_MERGED.jpg          # Stage 3 merged visualization
│   └── {filename}_MERGED.npy          # Stage 3 merged array
├── gtsegmented/                        # Ground truth visualizations
│   └── {filename}UNtiledgtSeg.jpg     # GT overlay from stage 1
├── results/                            # CSV evaluation metrics
│   ├── UNgt_results.csv               # Per-class IoU, TP/FP/FN
│   ├── UNreal_results.csv             # Class pixel counts
│   └── UNmIoU_results.csv             # Overall mIoU per image
├── stage1_baseline.npz                 # Pipeline intermediate
└── stage2_instances.npz                # Pipeline intermediate

./qsam_seg/
├── bboxes/
│   └── {filename}_bboxes.jpg          # Stage 2 bbox visualization
└── segmented/
    └── {filename}_segmented.jpg       # Stage 2 SAM visualization
```

---

## Dependencies

### Environment: resda (Stage 1)
```
Python 3.8+
PyTorch 1.12.1+cu116
torchvision 0.13.1+cu116
fairseq (custom version in ./custom_fairseq/)
mmsegmentation 0.28.0
mmcv-full 1.6.2
transformers 4.18.0.dev0
numpy 1.23.5
opencv-python
PIL
pandas
scipy
einops
```

### Environment: tosam (Stages 2-3)
```
Python 3.8+
PyTorch 2.0+
segment-anything
openai (for Qwen-VL API)
numpy
opencv-python
PIL
matplotlib
scipy
```

### API Keys Required
```bash
# Qwen-VL-Max API (used in stages 1 and 2)
# Key is hardcoded in files - TODO: move to environment variable
# Current key in:
#   - resda_baseline_aero.py (line 476)
#   - resda_sam_instance.py (line 16)
```

---

## TODOs

### High Priority
- [ ] **Extract API keys to environment variables**
  - Remove hardcoded Qwen API key from source files
  - Use `os.getenv('QWEN_API_KEY')` instead

- [ ] **Make dataset configurable via CLI**
  - Add `--dataset` argument to `run_pipeline.sh`
  - Pass to `resda_baseline_aero.py` instead of hardcoding in filename
  - Eliminate need for separate `resda_baseline_uavid.py`, etc.

- [ ] **Implement missing dataset transformers**
  - `UAVidTransformer` (currently stub)
  - `DroneSegTransformer` (currently stub)
  - `UDD6Transformer` (currently stub)

### Medium Priority
- [ ] **Add mIoU evaluation for merged output**
  - Currently only evaluates baseline (stage 1)
  - Need to compute mIoU on merged segmentation in stage 3

- [ ] **Handle multiple images in batch**
  - Currently processes one image and breaks (k=1)
  - Need to loop through entire test set

- [ ] **Optimize Qwen prompt for fewer hallucinations**
  - Test model-specific anti-hallucination strategies
  - Current prompt works well for Qwen, needs refinement

### Low Priority
- [ ] **Add tiling support**
  - If needed for very large images
  - Would add `_tiled` suffix to distinguish from default `no_tilings`

- [ ] **Add other VLM options**
  - GPT-4o, Claude, DeepSeek-VL2 as alternatives to Qwen
  - Make VLM selectable via config

- [ ] **Optimize CRF/ResNet parameters per dataset**
  - Currently: resnet_iters=25, crf_iters=10
  - May need dataset-specific tuning

---

## Troubleshooting

### Environment Issues
**Problem:** `ImportError: No module named 'fairseq'`
**Solution:** Activate correct environment
```bash
conda activate resda
```

**Problem:** `ImportError: No module named 'segment_anything'`
**Solution:** Activate tosam environment
```bash
conda activate tosam
```

### Pipeline Issues
**Problem:** Stage 1 works but stage 2 fails with missing .npz
**Solution:** Check that stage 1 completed and saved output
```bash
ls -lh ./output/stage1_baseline.npz
```

**Problem:** Merge fails with "class not in classes list"
**Solution:** Qwen detected a class not in the 150-class list. Check stage 2 output:
```bash
python -c "import numpy as np; d = np.load('./output/stage2_instances.npz', allow_pickle=True); print(d['bboxes'].item())"
```

### Checkpoint Issues
**Problem:** `FileNotFoundError: checkpoint.best_mIoU_0.0020.pt`
**Solution:** Download or train checkpoint for aeroscapes
```bash
# Train from scratch
bash run_scripts/aero.sh

# Or update checkpoint path in resda_baseline_aero.py line 38
```

---

## Research Background

### The Minority Class Suppression Problem

**Observation:** In aerial imagery, small objects (person, car, bike) get destroyed by ResNet iterative refinement.

**Test Case:** `aeroscapes/images/testing/000001_001.jpg`
- Ground truth: 972 person pixels
- With `resnet_iters=25`: Only 4 pixels detected (99.6% loss)
- With `resnet_iters=0`: ~7,000 pixels detected (over-detection but visible)

**Root Cause:**
```python
# ResNet refinement averages each pixel with top-k neighbors
for _ in range(25):  # 25 iterations
    batch_probs = batch_probs_topk.mean(dim=-2)  # Average with neighbors
```
- Person pixels surrounded by road/vegetation
- After 25 iterations → minority signal diluted to zero

**Solution:** Dual-pass architecture
1. Let baseline handle majority classes (sky, road, vegetation)
2. Use separate detector for bounded minorities (person, car)
3. Merge by painting detections onto baseline

**Results:**
- ✅ Person detected (via Qwen bbox + SAM)
- ✅ Car detected (wasn't found in tiled approach)
- ✅ Majority classes preserved from baseline

---

## References

- **Original IFSeg:** Image-free Semantic Segmentation (CVPR 2023)
- **OFA:** One For All: Unified Vision-Language Pre-training
- **SAM:** Segment Anything Model (Meta AI)
- **Qwen-VL:** Qwen Vision-Language Model (Alibaba)

---

## License

MIT
