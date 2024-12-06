import numpy as np
from Dataset import MRIDataset
from torch.utils.data import DataLoader

all_unique_labels = set()
root_dir = '/home/salahuddin/cornell/ADSP/Project/MRIdata/ForClass'
batch_size = 1

train_dataset = MRIDataset(root_dir=root_dir, mode='train')
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = MRIDataset(root_dir=root_dir, mode='val')
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


# for images, labels in train_loader:
#     unique_labels = np.unique(labels)
#     all_unique_labels.update(unique_labels.tolist())


for images, labels in val_loader:
    unique_labels = np.unique(labels)
    all_unique_labels.update(unique_labels.tolist())

print(f"All unique labels in the dataset: {sorted(all_unique_labels)}")
print(f"Total number of unique classes: {len(all_unique_labels)}")
