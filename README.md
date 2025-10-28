# ReSDA: Research Extensions to Semantic Dataset Analysis

**Three-stage semantic segmentation pipeline combining OFA baseline segmentation with SAM instance detection for improved minority class detection in aerial imagery.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

ReSDA solves the **minority class suppression problem** in aerial image segmentation by using a dual-pass architecture:

1. **Stage 1: OFA Baseline** - Vision-language model generates 150-class semantic segmentation
2. **Stage 2: Qwen+SAM Instances** - Detects and segments bounded objects (person, car, bike) that get lost in baseline
3. **Stage 3: Merge** - Overlays instance detections onto baseline for final output

**Key Innovation:** Small objects (person, car) that are suppressed by ResNet iterative refinement in the baseline are recovered via separate bounding box detection and SAM segmentation.

## Quick Start

### Installation

```bash
# 1. Clone repository
git clone https://github.com/yourusername/resda.git
cd resda

# 2. Create conda environments
conda env create -f environment_resda.yml   # OFA baseline (Stage 1)
conda env create -f environment_tosam.yml   # SAM instances (Stages 2-3)

# 3. Download model checkpoints
# - OFA checkpoint: Place in ./experiment_outputs/aero/
# - SAM checkpoint:
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
mkdir -p qsam_seg/checkpoint
mv sam_vit_h_4b8939.pth qsam_seg/checkpoint/

# 4. Set API key (required for Qwen-VL)
export QWEN_API_KEY="your-api-key-here"
```

### Usage

**Batch Processing** (recommended):
```bash
# Process N images through full pipeline
conda activate resda
python resda_batch.py \
    --test-images ./aeroscapes/images/testing \
    --test-annotations ./aeroscapes/annotations/testing \
    --num-files 10
```

**Manual Stage-by-Stage**:
```bash
# Stage 1: OFA baseline (resda environment)
conda activate resda
python resda_baseline_aero_standalone.py

# Stage 2: SAM instances (tosam environment)
conda activate tosam
python resda_sam_instance.py ./path/to/image.jpg

# Stage 3: Merge (tosam environment)
python resdaEX_merge.py ./output/stage1_baseline.npz ./output/stage2_instances.npz
```

### Outputs

All outputs saved to `./output/`:
- `./output/bboxes/{filename}_bboxes.jpg` - Bounding box visualization (Stage 2)
- `./output/segmented/{filename}UNtiledSeg.jpg` - Baseline segmentation (Stage 1)
- `./output/segmented/{filename}_segmented.jpg` - SAM segmentation (Stage 2)
- `./output/segmented/{filename}_MERGED.jpg` - Final merged result (Stage 3)

## Architecture

```
┌──────────────────────────────────────────┐
│ Stage 1: OFA Baseline                    │
│ (resda env, PyTorch 1.12.1)              │
│ Input:  Image                            │
│ Output: 150-class segmentation           │
└──────────────────┬───────────────────────┘
                   │
                   ↓
┌──────────────────────────────────────────┐
│ Stage 2: Qwen+SAM Instances              │
│ (tosam env, PyTorch 2.0+)                │
│ Input:  Image                            │
│ VLM:    Qwen-VL detects bounding boxes   │
│ Model:  SAM segments within bboxes       │
│ Output: Instance masks                   │
└──────────────────┬───────────────────────┘
                   │
                   ↓
┌──────────────────────────────────────────┐
│ Stage 3: Merge                           │
│ (tosam env)                              │
│ Input:  Baseline + Instance masks        │
│ Output: Final segmentation               │
└──────────────────────────────────────────┘
```

**Why two environments?**
- OFA requires old PyTorch 1.12.1 + transformers 4.18.0
- SAM requires modern PyTorch 2.0+
- Incompatible dependency trees

**Solution:** File-based pipeline with `conda run` to switch environments between stages.

## Supported Datasets

Currently configured for:
- ✅ **Aeroscapes** - 12 classes, aerial drone imagery
- 🚧 **UAVid** - Coming soon
- 🚧 **DroneSeg** - Coming soon
- 🚧 **UDD6** - Coming soon

## Requirements

### API Keys
- **Qwen-VL-Max** (required): For vision-language model calls in Stages 1 & 2
  - Get key from: https://dashscope.console.aliyun.com/
  - Set via: `export QWEN_API_KEY="your-key"`

### Hardware
- **GPU**: CUDA-capable GPU with 8GB+ VRAM (required for OFA and SAM)
- **RAM**: 16GB+ recommended
- **Storage**: 50GB+ for models and datasets

## Project Structure

```
resda/
├── resda_batch.py                  # Main entry point (batch processing)
├── resda_baseline_aero.py          # Stage 1: OFA baseline (library mode)
├── resda_baseline_aero_standalone.py  # Stage 1: Standalone mode
├── resda_sam_instance.py           # Stage 2: Qwen+SAM instances
├── resdaEX_merge.py                # Stage 3: Merge
├── resda_utils.py                  # Shared utilities
├── resda_dataset_transforms.py     # Dataset-specific transformers
├── classColor.txt                  # 150-class color mappings
├── classColorGTaero.txt            # Aeroscapes GT color mappings (12 classes)
├── environment_resda.yml           # Conda env for Stage 1
├── environment_tosam.yml           # Conda env for Stages 2-3
└── README_internal.md              # Detailed developer documentation
```

## Color Mapping Files

- **classColor.txt** - Universal 150-class color mappings used by OFA baseline
- **classColorGT{dataset}.txt** - Dataset-specific ground truth mappings:
  - `classColorGTaero.txt` - Aeroscapes (12 classes)
  - `classColorGT_uavid.txt` - UAVid (TBD)
  - `classColorGT_droneSeg.txt` - DroneSeg (TBD)
  - `classColorGT_udd6.txt` - UDD6 (TBD)

These files map the 150 generic classes to dataset-specific ground truth classes for evaluation.

## Documentation

- **[README_internal.md](README_internal.md)** - Comprehensive developer guide
  - Pipeline architecture details
  - File formats and intermediate outputs
  - Dataset-specific configurations
  - Troubleshooting guide
  - Research background and TODOs

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{resda2024,
  title={ReSDA: Research Extensions to Semantic Dataset Analysis},
  author={Your Name},
  year={2024},
  howpublished={\url{https://github.com/yourusername/resda}}
}
```

**Built on:**
- **IFSeg**: Image-free Semantic Segmentation (CVPR 2023)
- **OFA**: One For All Vision-Language Model
- **SAM**: Segment Anything Model (Meta AI)
- **Qwen-VL**: Qwen Vision-Language Model (Alibaba)

## License

MIT License - See LICENSE file for details.

Research code for academic use. Based on IFSeg (see original repository for license).

## Troubleshooting

**Problem:** `ImportError: No module named 'fairseq'`
**Solution:** Activate correct environment: `conda activate resda`

**Problem:** `ImportError: No module named 'segment_anything'`
**Solution:** Activate tosam environment: `conda activate tosam`

**Problem:** Stage 2 fails with JSON parsing error
**Solution:** Qwen response format changed. Check `resda_sam_instance.py` parsing logic (lines 143-172)

**Problem:** Merge fails with "class not in classes list"
**Solution:** Qwen detected a class not in the 150-class list. Check stage 2 output for unexpected classes.

For detailed troubleshooting, see [README_internal.md](README_internal.md#troubleshooting).

## Contact

[Your contact information]

## Acknowledgments

Thanks to the authors of IFSeg, OFA, SAM, and Qwen-VL for their foundational work.
