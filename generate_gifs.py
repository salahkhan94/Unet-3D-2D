# generate_gifs.py

import os
import torch
from torch.utils.data import DataLoader
from UNet_3D import UNet3D
from Dataset import MRIDataset
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import imageio
from tqdm import tqdm

def generate_gifs(model, dataset, dataset_name, output_dir, device):
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, dataset_name), exist_ok=True)
    
    # DataLoader
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # Set model to evaluation mode
    model.eval()
    
    with torch.no_grad():
        for idx, (images, labels) in enumerate(tqdm(loader, desc=f'Processing {dataset_name} dataset')):
            images = images.to(device)
            labels = labels.numpy().squeeze()  # Shape: (D, H, W)
            # Run model inference
            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1).cpu().numpy().squeeze()  # Shape: (D, H, W)
            # Prepare slices for GIFs
            num_slices = predictions.shape[0]  # Number of slices along the z-axis
            frames = []
            for i in range(num_slices):
                # Get the i-th slice for prediction and ground truth
                pred_slice = predictions[i, :, :]
                label_slice = labels[i, :, :]
                # Create a figure with side-by-side images
                fig, axes = plt.subplots(1, 2, figsize=(8, 4))
                axes[0].imshow(pred_slice, cmap='gray')
                axes[0].set_title('Model Output')
                axes[0].axis('off')
                axes[1].imshow(label_slice, cmap='gray')
                axes[1].set_title('Ground Truth')
                axes[1].axis('off')
                # Save the figure to a numpy array
                fig.canvas.draw()
                frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                frames.append(frame)
                plt.close(fig)
            # Save the frames as a GIF
            gif_path = os.path.join(output_dir, dataset_name, f'scan_{idx}.gif')
            imageio.mimsave(gif_path, frames, fps=5)
            print(f'Saved GIF for scan {idx} in {dataset_name} dataset at {gif_path}')

def main():
    # Paths (Adjust these paths accordingly)
    root_dir = '/path/to/MRIdata/ForClass'  # Root directory containing Training and Validation folders
    checkpoint_path = '/path/to/checkpoints/model_checkpoint.pth'  # Path to the trained model checkpoint
    output_dir = '/path/to/output_gifs'  # Directory to save the generated GIFs

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize training and validation datasets
    train_dataset = MRIDataset(root_dir=root_dir, mode='train')
    val_dataset = MRIDataset(root_dir=root_dir, mode='val', label_mapping=train_dataset.label_mapping)

    # Get num_classes from the training dataset
    num_classes = train_dataset.num_classes
    print(f"Number of classes: {num_classes}")

    # Initialize model
    in_channels = 1
    model = UNet3D(in_channels=in_channels, out_channels=num_classes).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Generate GIFs for training dataset
    generate_gifs(model, train_dataset, 'Training', output_dir, device)

    # Generate GIFs for validation dataset
    generate_gifs(model, val_dataset, 'Validation', output_dir, device)

if __name__ == '__main__':
    main()
