import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
from utils.data_us import (
    JointTransform2D, correct_dims,
    random_click, fixed_click, random_bbox, fixed_bbox,
)


def _parse_config_yaml(config_path):
    """Parse pixel-value mapping from a config.yaml file.

    Returns a list of (label_index, pixel_value) tuples.
    """
    mapping = []
    with open(config_path, 'r') as f:
        for line in f:
            parts = line.strip().split(':')
            if len(parts) >= 3:
                mapping.append((int(parts[0]), int(parts[2])))
    return mapping


def _apply_label_mapping(mask, mapping):
    """Map raw pixel values in *mask* to label indices, then binarise."""
    remapped = np.zeros_like(mask)
    for label_index, pixel_value in mapping:
        remapped[mask == pixel_value] = label_index
    remapped[remapped > 1] = 1
    return remapped


def _discover_seg_datasets(data_root, split):
    """Return sorted list of dataset names that contain *split*.txt."""
    seg_root = os.path.join(data_root, 'segmentation')
    if not os.path.isdir(seg_root):
        return []
    names = []
    for name in sorted(os.listdir(seg_root)):
        d = os.path.join(seg_root, name)
        if os.path.isdir(d) and os.path.exists(os.path.join(d, f'{split}.txt')):
            names.append(name)
    return names


class XpertUSDataset(Dataset):
    """Merges all segmentation datasets under XpertUS/data for training.

    Directory layout expected::

        data_root/
          segmentation/
            DatasetA/
              imgs/   masks/   train.txt   val.txt   test.txt   config.yaml
            DatasetB/
              ...
    """

    def __init__(self, data_root, split='train', joint_transform=None,
                 img_size=256, prompt='click'):
        super().__init__()
        self.data_root = data_root
        self.split = split
        self.prompt = prompt
        self.img_size = img_size
        self.joint_transform = joint_transform

        self.samples = []          # (img_path, mask_path, config_path, dataset_name)
        self.dataset_names = _discover_seg_datasets(data_root, split)

        for ds_name in self.dataset_names:
            ds_dir = os.path.join(data_root, 'segmentation', ds_name)
            config_path = os.path.join(ds_dir, 'config.yaml')
            split_file = os.path.join(ds_dir, f'{split}.txt')
            with open(split_file, 'r') as f:
                fnames = [l.strip() for l in f if l.strip()]
            for fname in fnames:
                self.samples.append((
                    os.path.join(ds_dir, 'imgs', fname),
                    os.path.join(ds_dir, 'masks', fname),
                    config_path,
                    ds_name,
                ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path, config_path, dataset_name = self.samples[idx]

        image = cv2.imread(img_path, 0)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        mapping = _parse_config_yaml(config_path)
        mask = _apply_label_mapping(mask, mapping)

        image, mask = correct_dims(image, mask)

        if self.joint_transform:
            image, mask, low_mask = self.joint_transform(image, mask)
        else:
            image = F.to_tensor(image)
            mask = F.to_tensor(mask)
            low_mask = mask.clone()

        class_id = 1
        if 'train' in self.split or 'val' in self.split:
            pt, point_label = random_click(np.array(mask), class_id)
            bbox = random_bbox(np.array(mask), class_id, self.img_size)
        else:
            pt, point_label = fixed_click(np.array(mask), class_id)
            bbox = fixed_bbox(np.array(mask), class_id, self.img_size)

        mask[mask != class_id] = 0
        mask[mask == class_id] = 1
        low_mask[low_mask != class_id] = 0
        low_mask[low_mask == class_id] = 1

        low_mask = low_mask.unsqueeze(0)
        mask = mask.unsqueeze(0)

        return {
            'image': image,
            'label': mask,
            'p_label': np.array(point_label),
            'pt': pt,
            'bbox': bbox,
            'low_mask': low_mask,
            'image_name': os.path.basename(img_path),
            'class_id': class_id,
            'dataset_name': dataset_name,
        }


class XpertUSSingleDataset(Dataset):
    """Loads a single segmentation dataset for per-dataset evaluation."""

    def __init__(self, data_root, dataset_name, split='test',
                 joint_transform=None, img_size=256, prompt='click'):
        super().__init__()
        self.dataset_name = dataset_name
        self.img_size = img_size
        self.prompt = prompt
        self.joint_transform = joint_transform

        ds_dir = os.path.join(data_root, 'segmentation', dataset_name)
        self.config_path = os.path.join(ds_dir, 'config.yaml')
        split_file = os.path.join(ds_dir, f'{split}.txt')
        with open(split_file, 'r') as f:
            fnames = [l.strip() for l in f if l.strip()]

        self.samples = []
        for fname in fnames:
            self.samples.append((
                os.path.join(ds_dir, 'imgs', fname),
                os.path.join(ds_dir, 'masks', fname),
            ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]

        image = cv2.imread(img_path, 0)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        mapping = _parse_config_yaml(self.config_path)
        mask = _apply_label_mapping(mask, mapping)

        image, mask = correct_dims(image, mask)

        if self.joint_transform:
            image, mask, low_mask = self.joint_transform(image, mask)
        else:
            image = F.to_tensor(image)
            mask = F.to_tensor(mask)
            low_mask = mask.clone()

        class_id = 1
        pt, point_label = fixed_click(np.array(mask), class_id)
        bbox = fixed_bbox(np.array(mask), class_id, self.img_size)

        mask[mask != class_id] = 0
        mask[mask == class_id] = 1
        low_mask[low_mask != class_id] = 0
        low_mask[low_mask == class_id] = 1

        low_mask = low_mask.unsqueeze(0)
        mask = mask.unsqueeze(0)

        return {
            'image': image,
            'label': mask,
            'p_label': np.array(point_label),
            'pt': pt,
            'bbox': bbox,
            'low_mask': low_mask,
            'image_name': os.path.basename(img_path),
            'class_id': class_id,
            'dataset_name': self.dataset_name,
        }
