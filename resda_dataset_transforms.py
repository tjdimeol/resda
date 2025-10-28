#!/usr/bin/env python
# coding: utf-8
"""
Dataset-specific transformer classes for ReSDA pipeline.
Each transformer handles mapping between 150-class segmentation and dataset-specific ground truth.
"""

import numpy as np
from collections import defaultdict


class AeroscapesTransformer:
    """
    Aeroscapes dataset transformer.
    Maps 150-class segmentation to 12 ground truth classes.
    """

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


class UAVidTransformer:
    """
    UAVid dataset transformer (STUB).
    TODO: Implement mapping for UAVid ground truth classes.
    """
    def __init__(self, color_filename='classColorGT_uavid.txt', mapping_filename='classColorGT_uavid.txt'):
        raise NotImplementedError("UAVidTransformer not yet implemented")


class DroneSegTransformer:
    """
    DroneSeg dataset transformer (STUB).
    TODO: Implement mapping for DroneSeg ground truth classes.
    """
    def __init__(self, color_filename='classColorGT_droneSeg.txt', mapping_filename='classColorGT_droneSeg.txt'):
        raise NotImplementedError("DroneSegTransformer not yet implemented")


class UDD6Transformer:
    """
    UDD6 dataset transformer (STUB).
    TODO: Implement mapping for UDD6 ground truth classes.
    """
    def __init__(self, color_filename='classColorGT_udd6.txt', mapping_filename='classColorGT_udd6.txt'):
        raise NotImplementedError("UDD6Transformer not yet implemented")


def get_transformer(dataset_name):
    """
    Factory function to get the appropriate transformer for a dataset.

    Args:
        dataset_name: String identifier for dataset (e.g., 'aero', 'uavid', 'droneseg', 'udd6')

    Returns:
        Transformer instance for the specified dataset

    Example:
        >>> ckpt = './experiment_outputs/aero/checkpoint.best_mIoU_0.0020.pt'
        >>> dataset = ckpt.split('/')[-2]  # 'aero'
        >>> transformer = get_transformer(dataset)
    """
    transformers = {
        'aero': AeroscapesTransformer,
        'aeroscapes': AeroscapesTransformer,
        'uavid': UAVidTransformer,
        'droneseg': DroneSegTransformer,
        'udd6': UDD6Transformer,
    }

    dataset_lower = dataset_name.lower()
    if dataset_lower not in transformers:
        raise ValueError(f"Unknown dataset: {dataset_name}. Supported: {list(transformers.keys())}")

    return transformers[dataset_lower]()
