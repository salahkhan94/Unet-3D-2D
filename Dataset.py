# Dataset.py

import os
import torch
from torch.utils.data import Dataset
import numpy as np
import nibabel as nib  # Nibabel can handle Analyze format
from glob import glob

class MRIDataset(Dataset):
    """
    A PyTorch Dataset class to handle MRI data in Analyze format.
    """

    def __init__(self, root_dir, transform=None, mode='train'):
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode  # 'train', 'val', or 'test'
        self.data_list = self._create_file_list()

    def _create_file_list(self):
        data_list = []
        if self.mode == 'train' or self.mode == 'val':
            if self.mode == 'train':
                data_dir = os.path.join(self.root_dir, (self.mode.capitalize() + 'ing'))
            else :
                data_dir = os.path.join(self.root_dir, (self.mode.capitalize() + 'idation'))
            print(data_dir)
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

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if self.mode == 'train' or self.mode == 'val':
            img_file, seg_file = self.data_list[idx]
            image = nib.load(img_file).get_fdata()
            label = nib.load(seg_file).get_fdata()
            # Remove singleton dimensions
            image = np.squeeze(image)  # Shape: (256, 256, 128)
            label = np.squeeze(label)  # Shape: (256, 256, 128)
            # Transpose to (D, H, W)
            image = np.transpose(image, (2, 0, 1))  # Now shape: (128, 256, 256)
            label = np.transpose(label, (2, 0, 1))  # Same shape
            # Add channel dimension to image only
            image = np.expand_dims(image, axis=0)  # Shape: (1, 128, 256, 256)
            # Convert data types
            image = image.astype(np.float32)
            label = label.astype(np.int64)
            # Apply transformations if any
            if self.transform:
                image = self.transform(image)
                label = self.transform(label)
            return torch.from_numpy(image), torch.from_numpy(label)
        elif self.mode == 'test':
            img_file = self.data_list[idx]
            image = nib.load(img_file).get_fdata()
            # Remove singleton dimensions
            image = np.squeeze(image)  # Shape: (256, 256, 128)
            # Transpose to (D, H, W)
            image = np.transpose(image, (2, 0, 1))  # Now shape: (128, 256, 256)
            # Add channel dimension
            image = np.expand_dims(image, axis=0)  # Now shape: (1, 128, 256, 256)
            # Convert to float32
            image = image.astype(np.float32)
            if self.transform:
                image = self.transform(image)
            return torch.from_numpy(image)
