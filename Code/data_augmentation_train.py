import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import random

from model_architectures import EnhancedAudioCNN

data = np.load("../../Data/development.npy")
data_csv = pd.read_csv("../../Data/metadata/development.csv")
idx_to_feature_name = pd.read_csv("../../Data/metadata/idx_to_feature_name.csv")

data_no_other = data[:40834, :, :]
data_csv_no_other = data_csv[:40834]

speakers = np.unique(data_csv_no_other['speaker_id'])
speakers_train = speakers[:137]
speakers_valid = speakers[137:]

train_indices = np.where(data_csv_no_other['speaker_id'].isin(speakers_train))[0].tolist()
valid_indices = np.where(data_csv_no_other['speaker_id'].isin(speakers_valid))[0].tolist()

data_train = data_no_other[train_indices]
data_valid = data_no_other[valid_indices]

labels_train = data_csv_no_other['word'][train_indices].tolist()
labels_valid = data_csv_no_other['word'][valid_indices].tolist()

labels = data_csv_no_other['word']
y = {}
i = 0
for label in np.unique(data_csv_no_other['word']).tolist():
    y[label] = i
    i += 1
labels_train_num = []
labels_valid_num = []

for label in labels_train:
    labels_train_num.append(y[label])
for label in labels_valid:
    labels_valid_num.append(y[label])

X_tensor_train = torch.tensor(data_train, dtype=torch.float32)
y_tensor_train = torch.tensor(labels_train_num, dtype=torch.long)
X_tensor_valid = torch.tensor(data_valid, dtype=torch.float32)
y_tensor_valid = torch.tensor(labels_valid_num, dtype=torch.long)

# Define fixed size
FIXED_SIZE = 44


# Define data augmentation functions
def time_stretch(audio, rate=1.0):
    rate = np.random.uniform(0.8, 1.2)
    indices = np.round(np.arange(0, audio.shape[1], rate)).astype(int)
    indices = indices[indices < audio.shape[1]]
    return audio[:, indices]


def pitch_shift(audio, n_steps):
    return np.roll(audio, n_steps, axis=1)


def add_noise(audio, noise_level=0.005):
    noise = np.random.randn(*audio.shape) * noise_level
    return audio + noise


def random_crop(audio, crop_size):
    start = random.randint(0, audio.shape[1] - crop_size)
    return audio[:, start:start + crop_size]


def pad_or_crop(audio, size):
    audio = torch.tensor(audio, dtype=torch.float32)
    if audio.shape[1] > size:
        return audio[:, :size]
    else:
        padding = size - audio.shape[1]
        return torch.nn.functional.pad(audio, (0, padding), 'constant', 0)


# Update CustomDataset class to include augmentations
class CustomDataset(Dataset):
    def __init__(self, features, labels, augment=False):
        self.features = features
        self.labels = labels
        self.augment = augment

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        feature = self.features[index]
        label = self.labels[index]

        if self.augment:
            feature = feature.numpy()
            if random.random() > 0.5:
                feature = time_stretch(feature)
            if random.random() > 0.5:
                feature = pitch_shift(feature, n_steps=random.randint(-5, 5))
            if random.random() > 0.5:
                feature = add_noise(feature)
            if random.random() > 0.5:
                feature = random_crop(feature, crop_size=feature.shape[1] // 2)
            feature = pad_or_crop(feature, FIXED_SIZE)

        return feature, label


# Train model with augmented data
model = EnhancedAudioCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

# Use augmented dataset for training
dataset_train = CustomDataset(X_tensor_train, y_tensor_train, augment=True)
dataloader_train = DataLoader(dataset_train, batch_size=64, shuffle=True, drop_last=True)

dataset_valid = CustomDataset(X_tensor_valid, y_tensor_valid, augment=False)
dataloader_valid = DataLoader(dataset_valid, batch_size=64, shuffle=False, drop_last=True)

use_mps = torch.backends.mps.is_available()
device = torch.device("mps" if use_mps else "cpu")
model.to(device)

# Example training loop
num_epochs = 50
losses_train = []
losses_valid = []

for epoch in range(num_epochs):
    model.train()
    for features, labels in dataloader_train:
        optimizer.zero_grad()
        features = features.unsqueeze(1)
        inputs = features.to(device)
        targets = labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    losses_train.append(loss.item())

    model.eval()
    with torch.no_grad():
        for features, labels in dataloader_valid:
            inputs = features.to(device)
            inputs = inputs.unsqueeze(1)
            targets = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        losses_valid.append(loss.item())
    if (losses_train[-1] < 0.5 and losses_valid[-1]) < 0.2:
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, f'model_epoch_{epoch}.pth')
    print(f'Epoch {epoch + 1}/{num_epochs}, Training loss: {losses_train[-1]}, Validation loss: {losses_valid[-1]}')
