import torch.nn as nn


class EnhancedAudioCNN(nn.Module):
    def __init__(self):
        super(EnhancedAudioCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(32, momentum=0.01)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(64, momentum=0.1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1)
        self.bn3 = nn.BatchNorm2d(128, momentum=0.1)
        self.conv4 = nn.Conv2d(128, 64, kernel_size=(3, 3), padding=1)
        self.bn4 = nn.BatchNorm2d(64, momentum=0.1)
        self.fc1 = nn.Linear(64 * 10 * 2, 32)
        self.relu = nn.ReLU()
        self.dropout2 = nn.Dropout(0.1)
        self.fc2 = nn.Linear(32, 20)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.dropout1(x)
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = self.pool(self.relu(self.bn4(self.conv4(x))))
        x = self.dropout2(x)
        x = x.view(-1, 64 * 10 * 2)  # Adjust the flattening to new layer dimensions
        x = self.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x


#  PAST MODELS

class AudioCNN(nn.Module):
    def __init__(self):
        super(AudioCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(3, 3), padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 3), padding=1)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(32 * 43 * 11, 64)
        self.fc2 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 21)
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 32 * 43 * 11)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.output(x)
        return x


class SimpleAudioCNN(nn.Module):
    def __init__(self):
        super(SimpleAudioCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=(3, 3), padding=1)  # Reduced the number of filters
        self.pool = nn.MaxPool2d(2, 2)
        # One Convolutional layer followed by pooling
        self.fc1 = nn.Linear(8 * 55 * 22, 50)  # Reduced number of neurons, and adjusted for new output size
        self.fc2 = nn.Linear(50, 21)  # Output layer remains the same, 21 classes
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = x.view(-1, 8 * 55 * 22)  # Flatten the output for the dense layer
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class MinimalAudioCNN(nn.Module):
    def __init__(self):
        super(MinimalAudioCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=(3, 3), padding=1)  # Only 4 filters
        self.pool = nn.MaxPool2d(2, 2)  # Pooling to reduce spatial dimensions
        self.fc1 = nn.Linear(4 * 87 * 22, 21)  # Directly connecting to output layer
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))  # Applying convolution then pooling
        x = x.view(-1, 4 * 87 * 22)  # Flatten the output for the dense layer
        x = self.fc1(x)  # Only one linear layer directly to outputs
        return x


class ModifiedAudioCNN(nn.Module):
    def __init__(self):
        super(ModifiedAudioCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=(3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(4)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(4 * 87 * 22, 21)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.dropout(x)
        x = x.view(-1, 4 * 87 * 22)
        x = self.fc1(x)
        return x
