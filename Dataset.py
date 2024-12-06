# Dataset.py

import os
import torch
from torch.utils.data import Dataset
import numpy as np
import nibabel as nib
from glob import glob

class MRIDataset(Dataset):
    """
    A PyTorch Dataset class to handle MRI data in Analyze format.
    """

    def __init__(self, root_dir, transform=None, mode='train', label_mapping=None):
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode  # 'train', 'val', or 'test'
        self.data_list = self._create_file_list()
        if self.mode == 'train':
            # Build label mapping
            self.label_mapping, self.reverse_label_mapping, self.num_classes = self._build_label_mapping()
        else:
            # Use label mapping from training dataset
            if label_mapping is None:
                raise ValueError("For validation and test datasets, label_mapping must be provided.")
            self.label_mapping = label_mapping
            self.num_classes = len(self.label_mapping)
            # Build reverse mapping for possible use in test
            self.reverse_label_mapping = {v: k for k, v in self.label_mapping.items()}

    def _create_file_list(self):
        data_list = []
        if self.mode == 'train' or self.mode == 'val':
            if self.mode == 'train':
                data_dir = os.path.join(self.root_dir, (self.mode.capitalize() + 'ing'))
            else :
                data_dir = os.path.join(self.root_dir, (self.mode.capitalize() + 'idation'))
            # print(data_dir)
            subjects = os.listdir(data_dir)
            for subj in subjects:
                img_path = os.path.join(data_dir, subj, 'images', 'analyze')
                seg_path = os.path.join(data_dir, subj, 'segmentation', 'analyze')
                img_file = glob(os.path.join(img_path, '*.img'))[0]
                seg_file = glob(os.path.join(seg_path, '*.img'))[0]
                data_list.append((img_file, seg_file))
        elif self.mode == 'test':
            data_dir = os.path.join(self.root_dir, 'Testing')
            subjects = os.listdir(data_dir)
            for subj in subjects:
                img_path = os.path.join(data_dir, subj, 'images', 'analyze')
                img_file = glob(os.path.join(img_path, '*.img'))[0]
                data_list.append(img_file)
        else:
            raise ValueError("Mode should be 'train', 'val', or 'test'")
        return data_list

    def _build_label_mapping(self):
        unique_labels = set()
        for idx in range(len(self.data_list)):
            _, label_file = self.data_list[idx]
            label = nib.load(label_file).get_fdata()
            label = np.squeeze(label)
            unique_labels.update(np.unique(label).tolist())
        unique_labels = sorted(unique_labels)
        label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
        reverse_label_mapping = {new_label: old_label for new_label, old_label in enumerate(unique_labels)}
        num_classes = len(unique_labels)
        return label_mapping, reverse_label_mapping, num_classes

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if self.mode == 'train' or self.mode == 'val':
            img_file, seg_file = self.data_list[idx]
            image = nib.load(img_file).get_fdata()
            label = nib.load(seg_file).get_fdata()
            # Remove singleton dimensions
            image = np.squeeze(image)  # Shape: (H, W, D)
            label = np.squeeze(label)  # Shape: (H, W, D)
            # Remap labels using self.label_mapping, default to 0 for labels not in mapping
            def map_label(x):
                return self.label_mapping.get(x, 0)  # Map to 0 if label not found
            label = np.vectorize(map_label)(label)
            label = label.astype(np.int64)
            # Transpose to (D, H, W)
            image = np.transpose(image, (2, 0, 1))  # Now shape: (D, H, W)
            label = np.transpose(label, (2, 0, 1))  # Same shape
            # Add channel dimension to image only
            image = np.expand_dims(image, axis=0)  # Shape: (1, D, H, W)
            # Convert data types
            image = image.astype(np.float32)
            # Apply transformations if any
            if self.transform:
                image = self.transform(image)
                label = self.transform(label)
            return torch.from_numpy(image), torch.from_numpy(label)
        elif self.mode == 'test':
            img_file = self.data_list[idx]
            image = nib.load(img_file).get_fdata()
            # Remove singleton dimensions
            image = np.squeeze(image)  # Shape: (H, W, D)
            # Transpose to (D, H, W)
            image = np.transpose(image, (2, 0, 1))  # Now shape: (D, H, W)
            # Add channel dimension
            image = np.expand_dims(image, axis=0)  # Now shape: (1, D, H, W)
            # Convert to float32
            image = image.astype(np.float32)
            if self.transform:
                image = self.transform(image)
            return torch.from_numpy(image)
