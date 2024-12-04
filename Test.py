# Test.py

import os
import torch
from UNet_3D import UNet3D
from Dataset import MRIDataset
from torch.utils.data import DataLoader
import nibabel as nib
import numpy as np

# Paths
root_dir = '/home/salahuddin/cornell/ADSP/Project/MRIdata/ForClass'
checkpoint_dir = '/home/salahuddin/cornell/ADSP/Project/Unet-3D-2D/model_checkpoint'
checkpoint_path = 'path/to/checkpoints/model_epoch_X.pth'  # Replace X with the desired epoch
output_dir = 'path/to/output_segmentations'
os.makedirs(output_dir, exist_ok=True)

# Hyperparameters
num_classes = 35
in_channels = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize model
model = UNet3D(in_channels=in_channels, out_channels=num_classes).to(device)
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

# Test dataset and dataloader
test_dataset = MRIDataset(root_dir=root_dir, mode='test')
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Inference
with torch.no_grad():
    for idx, images in enumerate(test_loader):
        images = images.to(device)
        outputs = model(images)
        # Get the predicted class for each voxel
        predictions = torch.argmax(outputs, dim=1)
        predictions = predictions.cpu().numpy().squeeze()
        # Save the segmentation as a NIfTI file
        affine = np.eye(4)  # Modify if you have the actual affine matrix
        segmentation_nifti = nib.Nifti1Image(predictions, affine)
        nib.save(segmentation_nifti, os.path.join(output_dir, f'segmentation_{idx}.nii.gz'))
