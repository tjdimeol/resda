import torch
# Fix for CUDA unknown error with PyTorch 2.5.0
torch.backends.cuda.enable_cudnn_sdp(False)

import sys
import argparse
import base64
from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from segment_anything import sam_model_registry, SamPredictor
import os
import matplotlib.pyplot as plt
from pathlib import Path

# 1. Initialize Qwen API client (using OpenAI-compatible endpoint)
client = OpenAI(
    api_key="sk-2b9d06aafce14f70a0070e22f7e0d6d3",
    base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
)

# 2. Load the Segment-Anything Model (SAM)
# Download checkpoint from: https://github.com/facebookresearch/segment-anything#model-checkpoints
sam = sam_model_registry["vit_h"](checkpoint="./qsam_seg/checkpoint/sam_vit_h_4b8939.pth")
sam.to(device="cuda")
predictor = SamPredictor(sam)

# 3. Parse command line arguments
if len(sys.argv) > 1:
    image_path = sys.argv[1]
else:
    # Default for standalone testing
    image_path = "/home/tdimeola/ifseg/vsw/testImages/000001_001.jpg"

image = Image.open(image_path)

# Helper function to encode image as base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# Get image dimensions for prompt
img_w, img_h = image.size

# Encode the image
base64_image = encode_image(image_path)

# Create the chat prompt with the segmentation request
prompt = f"""Please provide the bounding box coordinates for the main objects in the image. Provide as
    many details as possible to help the segmentation model understand the image better. The target objects for example 
    in an image with people, and devices, satchels, or other items they are
    holding. Include anything that is visible in the image, even if it is not prominent. For the output, you need to follow the format:
    - <Question 1>: <Answer 1>.
    - <Question 2>: <Answer 2>.
    ..., etc, where each pair of prompt and answer implies the chain
    of thought, i.e., different levels or different part of the image
    understanding. Following the first chain of thought, review the image again, following a second chain of
    thought to discover items you may have missed or reassess your original evaluation. Then add a second output. 
    If you see an object don't second guess yourself. Use it.

    Found objects must be selected from the following list.:
    ['car', 'wall', 'building', 'sky', 'lawn', 'tree', 'mall', 'road', 'motorcycle', 'reflection',
               'grass', 'shadow', 'sidewalk', 'human', 'earth', 'people', 'billboard', 'mountain', 'plant', 'facade',
               'highway', 'water', 'barn', 'supermarket', 'bush', 'house', 'sea', 'roof', 'window', 'vegetation',
               'juniper', 'hedge', 'fence', 'garage', 'rock', 'container', 'swimming-pool', 'driveway', 'railing',
               'pipeline', 'pipes', 'solar', 'column', 'sign', 'spruce', 'fir', 'sand', 'playground',
               'skyscraper', 'garden', 'beach', 'grandstand', 'path', 'stairs', 'runway', 'warehouse', 'backhoe',
               'crop', 'factory', 'stairway', 'river', 'bridge', 'hospital', 'windmill', 'market', 'pond',
               'stream', 'construction', 'hill', 'bench', 'landfill', 'cactus', 'palm', 'clearcut', 'park',
               'soccer-field', 'boat', 'lot', 'square', 'hovel', 'bus', 'roundabout', 'light', 'truck',
               'tower', 'awning', 'streetlight', 'forest', 'ar-marker', 'plane', 'waves', 'lamp', 'post', 'land',
               'obstacle', 'bald-tree', 'tractor', 'farm', 'silo', 'brook', 'door', 'van', 'ship', 'fountain',
               'railroad', 'canopy', 'fence-pole', 'antenna', 'fire', 'statue', 'track', 'stadium', 'waterfall',
               'tent', 'clouds', 'structure', 'hotel', 'highway', 'airport', 'cow', 'dog', 'storage-tank',
               'baseball-diamond', 'tennis-court', 'gravel', 'animal', 'bicycle', 'lake', 'basketball-court',
               'meadow', 'vehicle', 'flag', 'street', 'walkway', 'helicopter', 'man', 'woman', 'person', 'paved-area',
               'harbor', 'desert', 'dirt', 'woodland', 'bike', 'drone', 'background', 'boardwalk', 'conflicting',
               'cell-tower', 'other']

               Go over the selected classes carefully,  Do **not** skip signs, posts, or other markers. But be sure to include all of them
               when you observe them. 

               **CRITICAL RULES for assembling bounding boxes:**

               1. Must be a Discrete instance - The bounding box contains 99% of a single object/class type
               2. Must fill >80% of bounding box area - The object occupies at least 70% of the rectangular bounding box area (not sparse, elongated, or scattered)
               3. Must have clearly identifiable boundaries - The object has edges that visually separate it from surroundings
               4. Must be localized - Occupies a contiguous spatial area, not scattered across the image (CRITICAL)

               **LIMIT: Return ONLY 100 the bounded objects that most closely fill the above criteria**

               For each bounded object, provide:
               1. Class name (must be from the list above, lowercase)
               2. Bounding box coordinates: (x1, y1, x2, y2) in pixels
                  - x1, y1 = top-left corner
                  - x2, y2 = bottom-right corner
                  - Image size: {img_w} × {img_h} pixels

               **Output format:**
               ```python
               boosted_classes = ['person', 'car', 'building', ...]
               bounding_boxes = {{
                   'person': [(x1, y1, x2, y2)],
                   'car': [(x1, y1, x2, y2)],
                   'building': [(x1, y1, x2, y2)]
               }}
               ```

               Return ONLY these two Python variables.
               """

# Call Qwen API
completion = client.chat.completions.create(
    model="qwen-vl-max",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                    },
                },
            ],
        }
    ],
)

qwen_response = completion.choices[0].message.content
print("=== QWEN RESPONSE ===")
print(qwen_response)
print("=" * 60)

# Parse the bounding box coordinates from the response
# Qwen returns Python variables, not JSON
import re

# Parse boosted_classes
pattern = r'boosted_classes\s*[=]?\s*(\[.*?\])'
match = re.search(pattern, qwen_response, re.DOTALL)
boosted_classes = eval(match.group(1)) if match else []

# Parse bounding_boxes
pattern = r'bounding_boxes\s*[:=]?\s*(\{.*?\})'
match = re.search(pattern, qwen_response, re.DOTALL)
bboxes = eval(match.group(1)) if match else {}

# Handle malformed bbox format (flat list instead of list of tuples)
for class_name, box_list in bboxes.items():
    if box_list and isinstance(box_list[0], int):
        # Convert flat list to tuple: [x1,y1,x2,y2] -> [(x1,y1,x2,y2)]
        if len(box_list) == 4:
            bboxes[class_name] = [tuple(box_list)]
            print(f"  WARNING: Fixed malformed bbox for '{class_name}': {bboxes[class_name]}")

# Convert to format expected by rest of code: list of {label, bbox_2d}
detections = []
for class_name, box_list in bboxes.items():
    for bbox in box_list:
        detections.append({
            'label': class_name,
            'bbox_2d': list(bbox)  # Convert tuple to list [x1, y1, x2, y2]
        })

print(f"Detected classes: {boosted_classes}")
print(f"Bounding boxes: {bboxes}")
print(f"Found {len(detections)} objects")

# Create output directories
output_dir = Path("./output")
bbox_dir = output_dir / "bboxes"
seg_dir = output_dir / "segmented"
bbox_dir.mkdir(parents=True, exist_ok=True)
seg_dir.mkdir(parents=True, exist_ok=True)

# Get image filename without extension
image_name = Path(image_path).stem

# 1. Draw bounding boxes on the image
image_with_boxes = image.copy()
draw = ImageDraw.Draw(image_with_boxes)

# Try to load a font, fall back to default if not available
try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
except:
    font = ImageFont.load_default()

# Draw all bounding boxes
for det in detections:
    box = det['bbox_2d']  # [x1, y1, x2, y2]
    label = det['label']

    # Draw rectangle
    draw.rectangle(box, outline="red", width=3)

    # Draw label background
    text_bbox = draw.textbbox((box[0], box[1] - 25), label, font=font)
    draw.rectangle(text_bbox, fill="red")

    # Draw label text
    draw.text((box[0], box[1] - 25), label, fill="white", font=font)

    print(f"  - {label} at {box}")

# Save image with bounding boxes
bbox_output_path = bbox_dir / f"{image_name}_bboxes.jpg"
image_with_boxes.save(bbox_output_path)
print(f"Saved bounding boxes to: {bbox_output_path}")

# 2. Process all detections with SAM and create segmentation overlay
image_np = np.array(image)
predictor.set_image(image_np)

# Create a blank mask to accumulate all segmentations
h, w = image_np.shape[:2]
combined_mask = np.zeros((h, w), dtype=bool)
color_overlay = np.zeros((h, w, 3), dtype=np.uint8)

# Load class colors from classColor.txt to match baseline segmentation
def read_class_colors():
    """Read class colors from classColor.txt"""
    with open('classColor.txt', 'r') as f:
        colorlist = []
        for line in f:
            if not line.strip():
                continue
            # Parse format: "class_name (r, g, b)"
            color_part = line.strip().split('(', 1)[1].rstrip(')')
            colorlist.append(color_part)
        cmap = [eval(i) for i in colorlist]
    return np.array(cmap)

# Load colormap
classes = ['car', 'wall', 'building', 'sky', 'lawn', 'tree', 'mall', 'road', 'motorcycle', 'reflection',
           'grass', 'shadow', 'sidewalk', 'human', 'earth', 'people', 'billboard', 'mountain', 'plant', 'facade',
           'highway', 'water', 'barn', 'supermarket', 'bush', 'house', 'sea', 'roof', 'window', 'vegetation',
           'juniper', 'hedge', 'fence', 'garage', 'rock', 'container', 'swimming-pool', 'driveway', 'railing',
           'pipeline', 'pipes', 'solar', 'column', 'sign', 'spruce', 'fir', 'sand', 'playground',
           'skyscraper', 'garden', 'beach', 'grandstand', 'path', 'stairs', 'runway', 'warehouse', 'backhoe',
           'crop', 'factory', 'stairway', 'river', 'bridge', 'hospital', 'windmill', 'market', 'pond',
           'stream', 'construction', 'hill', 'bench', 'landfill', 'cactus', 'palm', 'clearcut', 'park',
           'soccer-field', 'boat', 'lot', 'square', 'hovel', 'bus', 'roundabout', 'light', 'truck',
           'tower', 'awning', 'streetlight', 'forest', 'ar-marker', 'plane', 'waves', 'lamp', 'post', 'land',
           'obstacle', 'bald-tree', 'tractor', 'farm', 'silo', 'brook', 'door', 'van', 'ship', 'fountain',
           'railroad', 'canopy', 'fence-pole', 'antenna', 'fire', 'statue', 'track', 'stadium', 'waterfall',
           'tent', 'clouds', 'structure', 'hotel', 'freeway', 'airport', 'cow', 'dog', 'storage-tank',
           'baseball-diamond', 'tennis-court', 'gravel', 'animal', 'bicycle', 'lake', 'basketball-court',
           'meadow', 'vehicle', 'flag', 'street', 'walkway', 'helicopter', 'man', 'woman', 'person', 'paved-area',
           'harbor', 'desert', 'dirt', 'woodland', 'bike', 'drone', 'background', 'boardwalk', 'conflicting',
           'cell-tower', 'other']

class_colors = read_class_colors()  # Shape: (150, 3)
class_to_color = {classes[i]: class_colors[i].tolist() for i in range(len(classes))}

# Process each detection and collect masks
sam_masks = {}  # {class_name: [mask1, mask2, ...]}

for idx, det in enumerate(detections):
    box = det['bbox_2d']
    label = det['label']

    # Run SAM to get the segmentation mask
    input_box = np.array(box)
    masks, scores, logits = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_box[None, :],  # SAM expects shape (1, 4)
        multimask_output=False,
    )

    mask = masks[0]  # Binary mask (H, W)
    print(f"  - Segmented {label}: mask shape {mask.shape}, score {scores[0]:.3f}")

    # Collect mask for stage 3
    if label not in sam_masks:
        sam_masks[label] = []
    sam_masks[label].append(mask)

    # Add this mask to the combined mask with proper class color
    if label in class_to_color:
        color = class_to_color[label]
    else:
        # Fallback to red if class not found
        color = [255, 0, 0]
        print(f"  WARNING: Class '{label}' not found in colormap, using red")
    color_overlay[mask] = color

# Create visualization: original image with colored mask overlay
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Original image
axes[0].imshow(image)
axes[0].set_title("Original Image")
axes[0].axis('off')

# Bounding boxes
axes[1].imshow(image_with_boxes)
axes[1].set_title("Qwen Bounding Boxes")
axes[1].axis('off')

# Segmentation overlay
axes[2].imshow(image)
axes[2].imshow(color_overlay, alpha=0.5)  # 50% transparent overlay
axes[2].set_title("SAM Segmentation")
axes[2].axis('off')

plt.tight_layout()

# Save segmentation visualization
seg_output_path = seg_dir / f"{image_name}_segmented.jpg"
plt.savefig(seg_output_path, dpi=150, bbox_inches='tight')
print(f"Saved segmentation to: {seg_output_path}")
plt.close()

# Save SAM masks to .npz for stage 3 (in addition to existing visualizations)
stage2_output = Path("./output") / "stage2_instances.npz"
stage2_output.parent.mkdir(parents=True, exist_ok=True)

np.savez(str(stage2_output),
         sam_masks=sam_masks)

print(f"\n✓ Saved stage 2 output to: {stage2_output}")
print(f"  Classes detected: {list(sam_masks.keys())}")
print(f"  Total masks: {sum(len(v) for v in sam_masks.values())}")


