#!/usr/bin/env python
# coding: utf-8

# **REZ**

# In[1]:


import os
import torch
import numpy as np
from fairseq import checkpoint_utils, distributed_utils, options, tasks, utils
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from tasks import *
from criterions import *
from matplotlib import cm

from models.segofa import SegOFAModel
from PIL import Image
import cv2
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from scipy.ndimage import binary_opening, binary_closing
from cv2 import bilateralFilter


tasks.register_task('segmentation', SegmentationTask)

# turn on cuda if GPU is available
use_cuda = torch.cuda.is_available()


# In[2]:


# Load pretrained ckpt & config
ckpt = './experiment_outputs/aero/checkpoint.best_mIoU_0.0020.pt'

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


# In[3]:


from einops import rearrange
from mmseg.ops import resize
import matplotlib.pyplot as plt
from crf import rgb_dense_crf
import torchvision.transforms.functional as VF
import time

def read_class_colors(image):
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
    - file_path: Path to the file containing class names and colors
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

def _remove_axes(ax):
    ax.xaxis.set_major_formatter(plt.NullFormatter())
    ax.yaxis.set_major_formatter(plt.NullFormatter())
    ax.set_xticks([])
    ax.set_yticks([])

def remove_axes(axes):
    if len(axes.shape) == 2:
        for ax1 in axes:
            for ax in ax1:
                _remove_axes(ax)
    else:
        for ax in axes:
            _remove_axes(ax)

def encode_text(text):
    line = [task.bpe.encode(' {}'.format(word.strip())) for word in text.strip().split()]
    line = ' '.join(line)

    s = task.tgt_dict.encode_line(
        line=line,
        add_if_not_exist=False,
        append_eos=False
    ).long()
    return s


# In[4]:


def pixel_color(pixColor):
    count = 0
    filename = open('classColor.txt', 'r')
    colorlist =[]

    while True:
        count += 1
        line = filename.readline()

        if not line:
           break
        x = line[:-1]
        x = x.split(",",1)
        name = x[0]
        color = x[1]
        pixColor = str(pixColor)

        if pixColor == color:
            print('pixColor is', pixColor, 'color is ', color)

            return name
            break

    filename.close()
    return name


# In[5]:


def merge_dictionaries(dict1, dict2):
    merged_dictionary = {}

    for key, value in dict1.items():
        new_value = dict1[key] + dict2[key]
        merged_dictionary[key] = new_value

    return merged_dictionary


# In[6]:


import math

def tile_image(img):

    tiles = {}
    img = cv2.imread(img)

    #default shape is 256, 400
    img_shape = img.shape
    img_height, img_width, _ = img_shape

    tile_width = int(img_width / 2)
    tile_height = int(img_height / 2)

    tile_size = (tile_width, tile_height)
    offset = (tile_width, tile_height)

    for i in range(int(math.ceil(img_shape[0]/(offset[1] * 1.0)))):
        for j in range(int(math.ceil(img_shape[1]/(offset[0] * 1.0)))):
            tiled_img = img[offset[1]*i:min(offset[1]*i+tile_size[1], img_shape[0]), offset[0]*j:min(offset[0]*j+tile_size[0], img_shape[1])]
            label = (i, j)
            tiles[label] = tiled_img
            cv2.imwrite(f"tiled_{i}_{j}.jpg", tiled_img)
    return tiles


# In[8]:


import pandas as pd

# Function to write data to a DataFrame and save it to CSV
def save_to_dataframe(data):
    df = pd.DataFrame(data)
    file_path = './aeroscapes/results/results.csv'
    try:
        with open(file_path, 'r') as f:
            headers_exist = True
    except FileNotFoundError:
        headers_exist = False

    # Append data to CSV
    df.to_csv(file_path, mode='a', header=not headers_exist, index=False)


# In[11]:


def create_weight_maps(tile_shape, overlap):
    h, w = tile_shape
    weight_map = np.ones((h, w), dtype=np.float32)

    if overlap > 0:
        horizontal_weights = np.linspace(1, 0, overlap)
        vertical_weights = np.linspace(1, 0, overlap)

        weight_map[:, :overlap] *= horizontal_weights  # Left edge
        weight_map[:, -overlap:] *= horizontal_weights[::-1]  # Right edge

        weight_map[:overlap, :] *= vertical_weights[:, np.newaxis]  # Top edge
        weight_map[-overlap:, :] *= vertical_weights[::-1][:, np.newaxis]  # Bottom edge

    return weight_map

def untile_single_category(tile_dict, img_shape, tile_size=(640, 360), overlap=5):
    h, w = img_shape
    reconstruction = np.zeros((h, w), dtype=np.float32)
    weight_map = np.zeros((h, w), dtype=np.float32)

    for (i, j), tile in tile_dict.items():
        y = i * (tile_size[1] - overlap)
        x = j * (tile_size[0] - overlap)

        end_y = min(y + tile_size[1], h)
        end_x = min(x + tile_size[0], w)

        tile_height = end_y - y
        tile_width = end_x - x

        # Create weight map for current tile size
        tile_weight_map = create_weight_maps((tile_height, tile_width), overlap)

        weighted_tile = tile[:tile_height, :tile_width] * tile_weight_map

        reconstruction[y:end_y, x:end_x] += weighted_tile
        weight_map[y:end_y, x:end_x] += tile_weight_map

    weight_map[weight_map == 0] = 1
    reconstruction = (reconstruction / weight_map).astype(np.uint8)

    return reconstruction

# Combine untile functions for both dictionaries
def untile_images(tile_dict_segmented, tile_dict_gt_segmented, img_shape, tile_size=(640, 360), overlap=5):
    print("shape img_shape", img_shape)
    reconstruction_segmented = untile_single_category(tile_dict_segmented, img_shape, tile_size, overlap)
    reconstruction_gt_segmented = untile_single_category(tile_dict_gt_segmented, img_shape, tile_size, overlap)

    return {
        'segmented_img': reconstruction_segmented,
        'gt_segmented_img': reconstruction_gt_segmented
    }


# In[12]:


from scipy.ndimage import label as label_nd, find_objects
from PIL import ImageDraw, ImageFont
from scipy.ndimage import label as label_nd

def ensure_uint8_rgb(image):
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)

    if len(image.shape) == 2 or image.shape[2] == 1:
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    return image

def place_labels(colored_image, segmented_image, classes, min_distance=10, save_file=None):
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

def on_draw(event):
    global ax, original_labels
    ax.cla()  # Clear the axis

    # Calculate the zoom factor
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    zoom_factor_x = (xlim[1] - xlim[0]) / labeled_image_np.shape[1]
    zoom_factor_y = (ylim[1] - ylim[0]) / labeled_image_np.shape[0]
    zoom_factor = max(zoom_factor_x, zoom_factor_y)

    # Adjust text size based on zoom factor
    for (centroid, class_name) in original_labels:
        ax.text(centroid[1], centroid[0], class_name, fontsize=18 / zoom_factor,
                color='black', ha='center', va='center')

    fig.canvas.draw_idle()  # Force a redraw


# In[13]:


import re
from collections import defaultdict

class UAVidColorTransformer:

    def __init__(self, color_filename='classColorGTaero.txt', mapping_filename='classColorGTaero.txt'):
        self.clr_tab = self.createColorTable()
        self.id_tab = {k: self.clr2id(v) for k, v in self.clr_tab.items()}
        self.id_reverse_tab = {v: k for k, v in self.id_tab.items()}
        self.class_mapping = self.loadClassMapping(mapping_filename)

    def createColorTable(self):
        return {
            'Vegetation': [0, 64, 0],
            'Road': [128, 128, 0],
            'Person': [192, 128, 128],
            'Obstacle':[192, 0, 0],
            'Construction':[192, 128, 0],
            'Bike': [0, 128, 0],
            'Car': [128, 128, 128],
            'Sky': [0, 128, 128],
            'Drone': [128, 0, 0],
            'Animal': [192, 0, 128],
            'Boat': [0, 0, 128],
            'Background':[0, 0, 0]
        }

    def color_to_index(self, color_str):
        color_value = list(map(int, color_str.strip('[]').split()))
        color_table = self.createColorTable()
        color_list = list(color_table.values())
        if color_value in color_list:
            return color_list.index(color_value)
        else:
            return -1  #Return -1 if the color is not found

    def index_to_class_name(self, index):
        color_table = self.createColorTable()
        class_names = list(color_table.keys())
        return class_names[index]

    def clr2id(self, clr):
        return clr[0] + clr[1] * 255 + clr[2] * 255 * 255

    def loadClassMapping(self, filename):
        mapping = defaultdict()
        with open(filename, 'r') as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) == 2:
                    original_class = int(parts[0])
                    target_class_name = parts[1]
                    mapping[original_class] = target_class_name
        return mapping

    def transform(self, label, dtype=np.int32):
        height, width = label.shape[:2]
        newLabel = np.zeros((height, width), dtype=dtype)
        id_label = label.astype(np.int64)
        id_label = id_label[:, :, 0] + id_label[:, :, 1] * 255 + id_label[:, :, 2] * 255 * 255
        print("label is ", label, "id_label is ", id_label)
        for tid, val in enumerate(self.id_tab.values()):
            mask = (id_label == val)
            newLabel[mask] = tid
        return newLabel

    def map_classes(self, original_labels):
        counts = {
            'Vegetation': 0,
            'Road': 0,
            'Person': 0,
            'Obstacle':0,
            'Construction':0,
            'Bike': 0,
            'Car': 0,
            'Sky': 0,
            'Drone': 0,
            'Animal': 0,
            'Boat': 0,
            'Background':0
            }
        height, width = original_labels.shape
        arr_size = height*width
        new_labels = np.zeros((height, width), dtype=np.int32)
        for i in range(original_labels.shape[0]):
            for j in range(original_labels.shape[1]):
                for original_class_id, target_class_name in self.class_mapping.items():
                    target_class_id = list(self.clr_tab.keys()).index(target_class_name)
                    if original_labels[i][j] == original_class_id:
                        new_labels[i][j] = target_class_id
                        self.increment_counts(target_class_name, counts)

        final_counts = [counts['Vegetation'], counts['Road'], counts['Person'], counts['Obstacle'], counts['Construction'],
                        counts['Bike'], counts['Car'], counts['Sky'], counts['Drone'], counts['Animal'],
                        counts['Boat'], counts['Background']]
        return final_counts, new_labels

    def map_to_colors(self, class_indices):
        height, width = class_indices.shape
        colored_image = np.zeros((height, width, 3), dtype=np.uint8)
        for tid in range(len(self.clr_tab)):
            mask = (class_indices == tid)
            class_name = list(self.clr_tab.keys())[tid]
            rgb = self.clr_tab[class_name]
            colored_image[mask] = rgb
        return colored_image

    def rgb_to_class_name(self, rgb):
        for class_name, color in self.clr_tab.items():
            if np.array_equal(rgb, color):
                return class_name

    def generalized_inverse_transform(self, label, labelSEG=None, mode="inverse"):
        counts = {
            'Vegetation': 0,
            'Road': 0,
            'Person': 0,
            'Obstacle':0,
            'Construction':0,
            'Bike': 0,
            'Car': 0,
            'Sky': 0,
            'Drone': 0,
            'Animal': 0,
            'Boat': 0,
            'Background':0
        }

        for i in range(label.shape[0]):
            for j in range(label.shape[1]):
                class_color = label[i, j]
                class_name = self.rgb_to_class_name(class_color)  # Convert RGB to class name
                class_color = str(class_color)
                gt_idx = self.color_to_index(class_color)

                if labelSEG is not None:
                    seg_idx = labelSEG[i,j]

                if gt_idx != -1:
                    if mode == "inverse":
                        self.increment_counts(class_name, counts)
                    elif mode == "tp" and gt_idx ==seg_idx:
                        self.increment_counts(class_name, counts)
                    elif mode == "fn" and gt_idx != seg_idx and class_name != "Background":
                        self.increment_counts(class_name, counts)
                    elif mode == "fp" and gt_idx != seg_idx:
                            self.increment_counts(self.index_to_class_name(seg_idx), counts)
        final_counts = [counts['Vegetation'], counts['Road'], counts['Person'], counts['Obstacle'], counts['Construction'],
                        counts['Bike'], counts['Car'], counts['Sky'], counts['Drone'], counts['Animal'],
                        counts['Boat'], counts['Background']]
        return final_counts

    def increment_counts(self, class_name, counts):
        counts[class_name] += 1

    def inverse_transform(self, label):
        return self.generalized_inverse_transform(label)

    def inverse_transform_tp(self, label, labelSEG):
        return self.generalized_inverse_transform(label, labelSEG, mode="tp")

    def inverse_transform_fn(self, label, labelSEG):
        return self.generalized_inverse_transform(label, labelSEG, mode="fn")

    def inverse_transform_fp(self, label, labelSEG):
        return self.generalized_inverse_transform(label, labelSEG, mode="fp")


# In[15]: CLASS WEIGHTING FUNCTIONS

from IPython.display import display, Image as IPyImage, Audio
import base64
import time
from openai import OpenAI

# Qwen client for class identification (replacing GPT-4o)
client = OpenAI(
    api_key="sk-2b9d06aafce14f70a0070e22f7e0d6d3",
    base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
)

def weight_booster(boosted_classes, classes, boost=100, neutral=1, override_weights=None):
    """Generate weight vector from boosted classes."""
    override_weights = override_weights or {}
    print("in weight_booster, override_weights are", override_weights)
    boosted_set = set(boosted_classes)
    weights = []
    for cls in classes:
        if cls in override_weights:
            weights.append(override_weights[cls])
        elif cls in boosted_set:
            weights.append(boost)
        else:
            weights.append(neutral)
    return weights

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def get_classes(f):
    """Query Qwen-VL to identify significant classes in the image."""
    base64_image = encode_image(f)

    completion = client.chat.completions.create(
        model="qwen-vl-max",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text":
    """
    You will serve as an agent for language-based image segmen-
    tation model. During each inference, your task is to describe
    a given image using chain of thought. You need to provide as
    many details as possible to help the segmentation model under-
    stand the image better. The target objects may contain multiple
    layers, be blocked by other objects, or be seamlessly embedded
    in their surroundings. Pay special attention to things such as water, sky, clouds, walls, etc. The
    first prompt can be related to the overall style or background of the image, then working on down to c
    very fine detail: for example in an image with people, and devices, satchels, or other items they are
    holding. Make sure to make a distinction between trees and other types of vegetation. Include anything that
    is visible in the image, even if it is not prominent. For the output, you need to follow the format:
    - <Question 1>: <Answer 1>.
    - <Question 2>: <Answer 2>.
    ..., etc, where each pair of prompt and answer implies the chain
    of thought, i.e., different levels or different part of the image
    understanding. Following the first chain of thought, review the image again, following a second chain of
    thought to discover items you may have missed or reassess your original evaluation. Then add a second output.

    classes must be selected from the following list.:
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

               Return as a list of every noun that you see in the total class list above, that you discover in
               the attached image. Go over it carefully with the image a make a careful assessment of the percentage of each object/thing
               in the list. If something is not on the list because it is unidentifiable or not on the main class list,
               just call it "other" and provide a percentage of that too. Do **not** skip signs, posts, or other markers. But be sure to include them
               when you observe them.

            Strict output rules:
            " - The dictionary override_weights in Python syntax as below.\n"
            " - Beneath: one bullet per changed class with concise, explanation.\n"
            " - Never add/remove/relabel keys. Order/length must match boosted_classes.\n"
            Elements in the returned class list (boosted_classes) must be exactly the same as those represented in the override_weights.
            The returned list of found classes must be in the format shown in Example Output1.
            
            **all override weights must be set to 100, aleays**
            
            "EXAMPLE OUTPUT1:\n"
            ```python\n
            boosted_classes = ['vegetation, person, ...']\n
            ```\n
            EXAMPLE OUTPUT2: \n
            "```python\n"
            "override_weights = {'vegetation': 100, 'person': 100, 'railroad' = 100 ...}\n"
            "```\n"
            NOTE: the classes represented in override_weights and boosted_classes must be in the *same* order!

            NOTE: class names in overright_weights *must* be LOWER-CASE.
            "REMINDER: No output is complete unless it includes a Python dictionary in the format
            'override_weights = {...}'. Never summarize, never skip, always output the dictionary—even if your
            answer has no other content."
        )

    """
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

    response_text = completion.choices[0].message.content
    print(response_text)

    pattern = r'override_weights\s*[:=]?\s*(\{.*?\})'
    match = re.search(pattern, response_text, re.DOTALL)
    dict_str1 = match.group(1)
    overrideWeights = eval(dict_str1)

    pattern = r'boosted_classes\s*[=]?\s(\[.*?\])'
    match = re.search(pattern, response_text, re.DOTALL)
    dict_str2 = match.group(1)
    boostedClasses = eval(dict_str2)

    print("in lm response, override_weights, boosted_classes are ", boostedClasses, overrideWeights)

    return boostedClasses, overrideWeights


def get_bounding_boxes(f, target_classes=None):
    """
    Query Qwen-VL to get bounding boxes for bounded objects.

    Args:
        f: Image file path
        target_classes: Not used (kept for compatibility)

    Returns:
        tuple: (bounding_boxes, detected_classes)
            - bounding_boxes: dict {class_name: [(x1,y1,x2,y2), ...]}
            - detected_classes: list of class names found
    """
    # Get image dimensions
    img = Image.open(f)
    img_w, img_h = img.size

    # Create Qwen client
    qwen_client = OpenAI(
        api_key="sk-2b9d06aafce14f70a0070e22f7e0d6d3",
        base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    )

    base64_image = encode_image(f)

    completion = qwen_client.chat.completions.create(
        model="qwen-vl-max",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"""
You are analyzing an aerial image that is {img_w} × {img_h} pixels.

Identify BOUNDED objects (objects with clear spatial boundaries) and provide bounding boxes.

**CRITICAL RULES for applying bounding boxes:**

  1. Must be a Discrete instance - The bounding box contains 99% of a single object/class type AND fulfills all other requirements below
  2. Must fill >80% of bounding box area - The object occupies at least 70% of the rectangular bounding box area (not sparse, elongated, or scattered)
  3. Must have clearly identifiable boundaries - The object has edges that visually separate it from surroundings
  4. Must be localized - Occupies a contiguous spatial area, not scattered across the image (CRITICAL)

1. **LIMIT: Return ONLY the 10 bounded objects that most closely fill the above criteria**

   
For each bounded object, provide:
1. Class name (must be from the list below, lowercase)
2. Bounding box coordinates: (x1, y1, x2, y2) in pixels
   - x1, y1 = top-left corner
   - x2, y2 = bottom-right corner

**Class list (use ONLY these names, lowercase):**
{classes}

**Output format:**
```python
boosted_classes = ['person', 'car', 'building', ...]
bounding_boxes = {{
    'person': [(x1, y1, x2, y2)],
    'car': [(x1, y1, x2, y2)],
    'building': [(x1, y1, x2, y2)]
}}
```

**Important:**
- Only include objects you are CONFIDENT about
- Coordinates must be integers within bounds (0 to {img_w}, 0 to {img_h})
- If no bounded objects found, return empty: `boosted_classes = []` and `bounding_boxes = {{}}`

Return ONLY these two Python variables.
"""
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

    response_text = completion.choices[0].message.content
    print("=== BOUNDING BOX RESPONSE ===")
    print(response_text)
    print("=" * 60)

    # Parse boosted_classes
    import re
    pattern = r'boosted_classes\s*[=]?\s*(\[.*?\])'
    match = re.search(pattern, response_text, re.DOTALL)
    detected_classes = eval(match.group(1)) if match else []

    # Parse bounding_boxes
    pattern = r'bounding_boxes\s*[:=]?\s*(\{.*?\})'
    match = re.search(pattern, response_text, re.DOTALL)
    bboxes = eval(match.group(1)) if match else {}

    # Handle malformed bbox format
    for class_name, box_list in bboxes.items():
        if box_list and isinstance(box_list[0], int):
            # Convert flat list to tuple: [x1,y1,x2,y2] -> [(x1,y1,x2,y2)]
            if len(box_list) == 4:
                bboxes[class_name] = [tuple(box_list)]
                print(f"  WARNING: Fixed malformed bbox for '{class_name}': {bboxes[class_name]}")

    print(f"Detected bounding boxes: {bboxes}")
    print(f"Detected classes: {detected_classes}")

    return bboxes, detected_classes


def filter_overlapping_bboxes(bboxes, img_w, img_h, max_area_pct=40):
    """
    Filter out problematic bounding boxes:
    1. Remove boxes that are too large (>max_area_pct of image)
    2. When boxes overlap, keep smaller box and remove larger one

    Args:
        bboxes: dict {class_name: [(x1,y1,x2,y2), ...]}
        img_w, img_h: Image dimensions
        max_area_pct: Maximum allowed box area as percentage of image

    Returns:
        filtered_bboxes: dict with problematic boxes removed
    """
    print("\n" + "="*60)
    print("FILTERING OVERLAPPING/OVERSIZED BBOXES")
    print("="*60)

    img_area = img_w * img_h
    max_box_area = img_area * (max_area_pct / 100)

    # Flatten to list of (class_name, bbox, area) for easier processing
    all_boxes = []
    for class_name, box_list in bboxes.items():
        for bbox in box_list:
            x1, y1, x2, y2 = bbox
            area = (x2 - x1) * (y2 - y1)
            all_boxes.append((class_name, bbox, area))

    # Filter 1: Remove oversized boxes
    filtered_boxes = []
    for class_name, bbox, area in all_boxes:
        area_pct = 100 * area / img_area
        if area > max_box_area:
            print(f"  REMOVED '{class_name}': too large ({area_pct:.1f}% of image)")
        else:
            filtered_boxes.append((class_name, bbox, area))

    # Filter 2: Remove boxes that overlap with smaller boxes
    def boxes_overlap(box1, box2):
        """Check if two boxes overlap"""
        x1a, y1a, x2a, y2a = box1
        x1b, y1b, x2b, y2b = box2

        # No overlap if one box is completely to the left/right/above/below the other
        if x2a < x1b or x2b < x1a or y2a < y1b or y2b < y1a:
            return False
        return True

    keep_boxes = []
    for i, (class_name_i, bbox_i, area_i) in enumerate(filtered_boxes):
        should_keep = True

        # Check against all other boxes
        for j, (class_name_j, bbox_j, area_j) in enumerate(filtered_boxes):
            if i == j:
                continue

            # If boxes overlap and this box is larger, remove it
            if boxes_overlap(bbox_i, bbox_j) and area_i > area_j:
                print(f"  REMOVED '{class_name_i}': overlaps with smaller box '{class_name_j}'")
                should_keep = False
                break

        if should_keep:
            keep_boxes.append((class_name_i, bbox_i, area_i))

    # Reconstruct dict format
    filtered_bboxes = {}
    for class_name, bbox, area in keep_boxes:
        if class_name not in filtered_bboxes:
            filtered_bboxes[class_name] = []
        filtered_bboxes[class_name].append(bbox)

    print(f"\nFiltering complete:")
    print(f"  Original: {len(all_boxes)} boxes")
    print(f"  Kept: {len(keep_boxes)} boxes")
    print(f"  Removed: {len(all_boxes) - len(keep_boxes)} boxes")

    return filtered_bboxes


def merge_segmentations(seg_baseline, bboxes, classes):
    """
    Merge baseline segmentation with bbox detections.

    Simply paints bbox regions with their detected class IDs, overriding the baseline.
    No model inference needed - we trust Qwen's bboxes completely.

    Args:
        seg_baseline: Baseline segmentation map [H, W] with class indices
        bboxes: dict {class_name: [(x1,y1,x2,y2), ...]}
        classes: Full list of 150 class names

    Returns:
        merged_seg: Final segmentation map [H, W] with bbox regions overlaid
    """
    print("\n" + "="*60)
    print("MERGING BASELINE + BBOX SEGMENTATIONS")
    print("="*60)

    # Start with baseline
    merged_seg = seg_baseline.copy()
    img_h, img_w = seg_baseline.shape

    if not bboxes:
        print("No bboxes to merge - returning baseline segmentation")
        return merged_seg

    # Paint each bbox region with its class
    for class_name, box_list in bboxes.items():
        if class_name in classes:
            class_idx = classes.index(class_name)

            for (x1, y1, x2, y2) in box_list:
                # Clamp to image bounds
                x1, x2 = max(0, min(x1, img_w)), max(0, min(x2, img_w))
                y1, y2 = max(0, min(y1, img_h)), max(0, min(y2, img_h))

                # Paint bbox region with class ID
                merged_seg[y1:y2, x1:x2] = class_idx

                print(f"  Painted '{class_name}' (class {class_idx}) at bbox ({x1},{y1})-({x2},{y2})")
        else:
            print(f"  WARNING: '{class_name}' not in classes list - skipping bbox")

    # Count how many pixels were changed
    pixels_changed = np.sum(merged_seg != seg_baseline)
    total_pixels = img_h * img_w
    pct_changed = 100 * pixels_changed / total_pixels

    print(f"\nMerge complete:")
    print(f"  Pixels changed: {pixels_changed:,} / {total_pixels:,} ({pct_changed:.2f}%)")
    print(f"  Bbox classes painted: {list(bboxes.keys())}")

    return merged_seg


# In[16]:


import torch.nn.functional as F
from torchvision.transforms.functional import to_pil_image
from torchvision import transforms
from mmseg.datasets.pipelines import MultiScaleFlipAug
import shutil
import re

clrEnc = UAVidColorTransformer()
mean = std = [0.5, 0.5, 0.5]
image_normalize = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize(mean=mean, std=std)])

image_transform = MultiScaleFlipAug(img_scale=(512*4, 512),
                                    flip=False,
                                    transforms=[dict(type='Resize', keep_ratio=True),
                                                dict(type='RandomFlip')])
#set the test directory
directory = './aeroscapes/images/testing'
#set the GT directory
GTdirectory = './aeroscapes/annotations/testing'

gt_classes = 'Vegetation', 'Road', 'Person', 'Obstacle', 'Construction','Bike', 'Car', 'Sky', 'Drone', 'Animal','Boat', 'Background'
print("gt_classes are ", gt_classes)


classes = [
    'car', 'wall', 'building', 'sky', 'lawn', 'tree', 'mall', 'road', 'motorcycle', 'reflection',
           'grass', 'shadow', 'sidewalk', 'human', 'earth', 'pine', 'elm', 'mountain', 'plant', 'facade',
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

class_number = [0]*150
name_dict = dict(zip(classes, class_number))

import glob

mIoU = 0
all_data_real = []
all_data_gt = []
all_mIoUs = []

k=1 #how many files to segment on this run

GTfiles = sorted(glob.glob(os.path.join(GTdirectory, "*.png")))
SEGfiles = sorted(glob.glob(os.path.join(directory, "*.jpg")))
full_list = list(zip(GTfiles, SEGfiles))


# ========== OBJECTIVE FUNCTION ==========
def objective(GTfile, file, weights, classes, gt_classes):
    """
    Run segmentation on a single image with given class weights.

    Args:
        GTfile: Ground truth image path
        file: Input image path
        weights: List of class weights (length = num_classes)
        classes: List of all class names
        gt_classes: List of ground truth class names

    Returns:
        overall_mIoU
    """
    print("weights going in", weights)

    GTf = GTfile
    f = file
    untiled_imgs, hold1, hold2, to_untile_combined = {}, {}, {}, {}

    C = 12  # Number of GT classes
    sum_ious = [0.0] * C

    GTfilename = os.path.basename(GTf)
    filename = os.path.basename(f)
    print("GTfilename is ", GTfilename, "filename is", filename)

    orig_img = Image.open(f)
    img_w = orig_img.width
    img_h = orig_img.height
    orig_img = np.asarray(orig_img)

    iou_counts = {
            'gtValues': 0,
            'segValues_gtmasks': 0,
            'tp': 0,
            'fn': 0,
            'fp': 0
        }
    tiled_imgs = {}

    pil_image = Image.open(f)
    gt_image = Image.open(GTf)

    image_arr = np.asarray(pil_image).copy()
    image_arr_bgr = image_arr[:, :, ::-1].copy() # to BGR
    ori_shape = image_arr_bgr.shape

    results = {}
    results['img'] = image_arr_bgr
    results['img_shape'] = ori_shape
    results['scale_factor'] = 1.0

    img_dict = image_transform(results)
    img = img_dict.pop('img')[0]
    img = img[:, :, ::-1].copy() # to RGB
    img = image_normalize(img)

    bos_item = torch.tensor([0])
    eos_item = torch.tensor([2])
    prompt = encode_text(' what is the segmentation map of the image? object:')
    id2text = [encode_text(f" {x}") for x in classes]

    src_text = torch.cat([bos_item, prompt, *id2text, eos_item], dim=0)
    net_input =  {
        "src_tokens": src_text.unsqueeze(0),
        "src_lengths": torch.tensor([len(src_text)]),
        "patch_images": img.unsqueeze(0),
        "patch_masks": torch.tensor([True]),
        "prev_output_tokens": bos_item.unsqueeze(0)
        }

    resnet_topk = 5
    resnet_iters = 25
    crf_iters = 10

    net_input = utils.move_to_cuda(net_input) if use_cuda else net_input

    with torch.inference_mode():
        start_time = time.time()

        batch_logits, extra = model(**net_input)
        batch_logits = batch_logits[:, :-1, :]

        B = batch_logits.size(0)
        H, W = extra['encoder_returns']['image_embed_shape'][0]
        C = batch_logits.size(2)

        # ========== APPLY CLASS WEIGHTS TO LOGITS ==========
        # Reshape logits for weighting [B, H*W, C] -> [B, C, H, W]
        batch_logits_reshaped = rearrange(batch_logits, 'b (h w) d -> b d h w', h=H, w=W)

        # Build class weights tensor [1, C, 1, 1]
        weights_tensor = torch.tensor(weights, dtype=batch_logits.dtype, device=batch_logits.device).view(1, C, 1, 1)

        # Apply weights
        batch_logits_weighted = batch_logits_reshaped * weights_tensor

        # Flatten back to [B, H*W, C]
        batch_logits = rearrange(batch_logits_weighted, 'b d h w -> b (h w) d')
        # ===================================================

        # Compute probabilities
        batch_probs = batch_logits.softmax(-1)

        # ========== RESNET FEATURE REFINEMENT ==========
        batch_probs_resnet = batch_probs.clone()
        resnet_feature = extra['encoder_returns']['image_embed_before_proj'][0]
        resnet_feature_norm = F.normalize(resnet_feature, dim=-1)
        cosine_sim = resnet_feature_norm @ resnet_feature_norm.transpose(-1, -2)
        _, topk_ind = torch.topk(cosine_sim, k=resnet_topk, dim=-1)
        batch_ind = torch.arange(B).unsqueeze(-1).unsqueeze(-1).expand(-1, H*W, resnet_topk)

        for _ in range(resnet_iters):
            batch_probs_resnet_topk = batch_probs_resnet[batch_ind, topk_ind]
            batch_probs_resnet = batch_probs_resnet_topk.mean(dim=-2)
        # ===============================================

        batch_probs = rearrange(batch_probs, 'b (h w) d -> b d h w', h=H, w=W)
        batch_probs = F.interpolate(batch_probs, size=ori_shape[:2], mode="bilinear", align_corners=False)

        batch_probs_resnet = rearrange(batch_probs_resnet, 'b (h w) d -> b d h w', h=H, w=W)
        batch_probs_resnet = F.interpolate(batch_probs_resnet, size=ori_shape[:2], mode="bilinear", align_corners=False)

        # CRF refinement
        batch_prob_rgb_crf = rgb_dense_crf(image_arr_bgr, batch_probs_resnet[0].detach().cpu().numpy(), max_iter=crf_iters)
        batch_pred_rgb_crf = batch_prob_rgb_crf.argmax(0)

        batch_pred_resnet = batch_probs_resnet[0].argmax(0).cpu()
        batch_pred = batch_probs[0].argmax(0).cpu()
        print("--- %s seconds ---" % (time.time() - start_time))

    arr1 = np.array(gt_image, dtype='uint8')
    arr3 = np.array(batch_pred_rgb_crf, dtype='uint8')

    iou_counts = {
        'gtValues': np.zeros(12, dtype=np.int32),
        'segValues_gtmasks': np.zeros(12, dtype=np.int32),
        'tp': np.zeros(12, dtype=np.int32),
        'fn': np.zeros(12, dtype=np.int32),
        'fp': np.zeros(12, dtype=np.int32),
    }

    gtValues= clrEnc.inverse_transform(arr1)

    print("Categories are: [Vegetation, Road, Person, Obstacle, Construction, Bike, Car, Sky, Drone, Animal, Boat, Background")
    print("Grount Truth Values ", gtValues)

    # Apply class mapping to transform the 150-class segmented image to 12-class indices
    mapped_seg_classCounts, arr3_new_ids = clrEnc.map_classes(arr3)

    segValues_gtmasks= mapped_seg_classCounts
    print("Segmented Image with Ground Truth-class Masks, Values ", segValues_gtmasks)

    tp= clrEnc.inverse_transform_tp(arr1, arr3_new_ids)

    fn= clrEnc.inverse_transform_fn(arr1, arr3_new_ids)

    fp= clrEnc.inverse_transform_fp(arr1, arr3_new_ids)

    denominator = [x + y + z for x,y,z in zip(tp, fn, fp)]
    denominator = np.array(denominator)
    tp = np.array(tp)

    print("tp is, denominator is", tp, denominator)

    tp[denominator==0] = 1
    denominator[denominator==0] = 1

    print(denominator)

    # Safely calculate IoU
    iou_per_class = np.divide(tp, denominator)
    print("iou_per_class is",iou_per_class)

    for i in range(len(iou_per_class)):
        sum_ious[i] += iou_per_class[i]

    img_shape_segmented = (img_h, img_w)

    mapped_labels = clrEnc.map_classes(batch_pred_rgb_crf)
    gt_segmented_img = clrEnc.map_to_colors(mapped_labels[1])
    segmented_img, coldict = dataset_cmap(batch_pred_rgb_crf)

    colkeys = np.array(list(coldict.keys()))
    colvalues = np.array(list(coldict.values()))

    segmented_keys = list(colkeys[colvalues!=0])
    segmented_values = list(colvalues[colvalues!=0])

    labeled_image_pil, original_labels = place_labels(segmented_img, batch_pred_rgb_crf, classes, min_distance=15, save_file='labeled_image.png')
    labeled_image_pil.show()

    labeled_image_np = np.array(labeled_image_pil)
    segmented_img = ensure_uint8_rgb(segmented_img)
    segmented_img_pil = Image.fromarray(segmented_img)
    reconstructed_img_gt_pil = Image.fromarray(gt_segmented_img)

    opacity = 0.5
    overlap_img = orig_img.copy() * (1 - opacity) + gt_segmented_img* opacity
    overlap_img = overlap_img.astype(np.uint8)
    overlap_img_pil = VF.to_pil_image(overlap_img)

    # Saving the reconstructed images
    print("filename before sending to paths is ", filename)
    filename = os.path.splitext(filename)[0]
    labeled_image_pil.save('./output/segmented/' + filename + 'UNtiledSeg.jpg')
    reconstructed_img_gt_pil.save('./output/gtsegmented/' + filename + 'UNtiledgtSeg.jpg')

    data_real = {
        'Segmented Keys': segmented_keys,
        'Segmented Values': segmented_values
        }

    df2_current = pd.DataFrame(data_real)
    df2_current['Filename'] = filename  # Adding filename as a column for reference

    all_data_real.append(df2_current)

    data_gt = {
        'Class Name': gt_classes,
        'Ground Truth Values': gtValues,
        'Segmented GT Values': segValues_gtmasks,
        'True Positives': tp,
        'False Negatives': fn,
        'False Positives': fp,
        'IoU': iou_per_class,
        }

    print("data_gt is", data_gt)
    df1_current = pd.DataFrame(data_gt)
    df1_current['Filename'] = filename  # Adding filename as a column for reference
    df1_current['GT Filename'] = GTfilename  # Adding gt filename as a column for reference

    all_data_gt.append(df1_current)

    # Division by 1 for untiled images
    num_tiled_images = 1
    mIoU_per_class = [iou / num_tiled_images for iou in sum_ious]

    print("mIoU_per_class:", mIoU_per_class)
    num_classes = len(mIoU_per_class)
    overall_mIoU = sum(mIoU_per_class) / num_classes
    print("mIoU for this file", overall_mIoU)

    mIOU = {
            'File Name': filename,
            'mIoU': [overall_mIoU],
                }
    df3_current = pd.DataFrame(mIOU)
    all_mIoUs.append(df3_current)

    # Concatenate all individual DataFrames
    final_df1 = pd.concat(all_data_gt, ignore_index=True)
    final_df2 = pd.concat(all_data_real, ignore_index=True)
    final_df3 = pd.concat(all_mIoUs, ignore_index=True)

    # Save or display the final DataFrames
    final_df1.to_csv('./output/results/UNgt_results.csv', index=False)
    final_df2.to_csv('./output/results/UNreal_results.csv', index=False)
    final_df3.to_csv('./output/results/UNmIoU_results.csv', index=False)

    print("\nFinal DataFrame 2: mIoU for GT Correlated Segmentation")
    print(final_df3)

    return overall_mIoU, batch_pred_rgb_crf


for GTf, f in zip(GTfiles, SEGfiles):

    #run segmentation
    bc, ow = get_classes(f)

    override_weights = ow
    boosted_classes = bc


    print("after override_weights assignment, boosted_classes\nn override_weights", boosted_classes, override_weights)
    weights = weight_booster(boosted_classes, classes, boost=0, neutral=0, override_weights=override_weights)

    print('weights are ', weights)

    # Pass 1: Baseline segmentation with LM-identified classes
    print("\n" + "="*60)
    print("PASS 1: BASELINE SEGMENTATION")
    print("="*60)
    baseline_mIoU, seg_baseline = objective(GTf, f, weights, classes, gt_classes)
    print(f"Baseline mIoU: {baseline_mIoU:.4f}")

    # Get bounding boxes - let Qwen find whatever bounded objects it wants
    print("\n" + "="*60)
    print("GETTING BOUNDING BOXES")
    print("="*60)
    bboxes, detected_classes = get_bounding_boxes(f, target_classes=None)

    # Filter out oversized and overlapping boxes
    if bboxes:
        img = Image.open(f)
        img_w, img_h = img.size
        bboxes = filter_overlapping_bboxes(bboxes, img_w, img_h, max_area_pct=40)

    # Visualize bounding boxes on original image before merging
    if bboxes:
        print("\n" + "="*60)
        print("VISUALIZING BOUNDING BOXES ON ORIGINAL IMAGE")
        print("="*60)

        # Load original image and create copy for drawing
        orig_image = Image.open(f)
        bbox_vis_img = orig_image.copy()
        draw = ImageDraw.Draw(bbox_vis_img)

        # Try to load a font, fall back to default if not available
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        except:
            font = ImageFont.load_default()

        # Draw each bounding box
        for class_name, box_list in bboxes.items():
            for (x1, y1, x2, y2) in box_list:
                # Draw rectangle outline only (not filled)
                draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

                # Draw label with background
                text_bbox = draw.textbbox((x1, y1 - 25), class_name, font=font)
                draw.rectangle(text_bbox, fill="red")
                draw.text((x1, y1 - 25), class_name, fill="white", font=font)

                print(f"  Drew bbox for '{class_name}' at ({x1},{y1})-({x2},{y2})")

        # Save visualization
        filename = os.path.basename(f)
        filename_base = os.path.splitext(filename)[0]
        os.makedirs('./output/bboxes', exist_ok=True)
        bbox_output_path = f'./output/bboxes/{filename_base}_bboxes.jpg'
        bbox_vis_img.save(bbox_output_path)
        print(f"Saved bbox visualization to: {bbox_output_path}")

    # Merge: Paint bbox regions onto baseline segmentation
    merged_seg = merge_segmentations(seg_baseline, bboxes, classes)

    # Save merged segmentation visualization
    filename = os.path.basename(f)
    filename_base = os.path.splitext(filename)[0]

    # Convert merged seg to colored image
    segmented_img_merged, coldict_merged = dataset_cmap(merged_seg)
    segmented_img_merged = ensure_uint8_rgb(segmented_img_merged)

    # Place labels on merged result
    labeled_image_merged, _ = place_labels(
        segmented_img_merged, merged_seg, classes,
        min_distance=15, save_file=f'./output/segmented/{filename_base}_MERGED.png'
    )

    # Save
    labeled_image_merged.save(f'./output/segmented/{filename_base}_MERGED.jpg')
    print(f"\n✓ Saved merged segmentation to: ./output/segmented/{filename_base}_MERGED.jpg")

    # TODO: Evaluate merged segmentation mIoU


    if k == 1:
        break
    else:
        k+=1


# In[ ]:
