import torch.nn as nn
import torch.nn.functional as F


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # Block 1 Input image 32 x 32; output image 8x8
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 2 Input image 8 x 8; output image 2x2
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=512, kernel_size=3, stride=2, padding=1)

        # Block 3 Input image 2 x 2; output image 1x1
        self.conv3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=1, stride=1)  # why do we have to do pooling.. at 1x1?

        # Block 4
        self.conv4 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1)
        # Block 5
        self.conv5 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(512, 10)

    def forward(self, x):
        # forward Block 1
        x = F.relu(self.pool(self.conv1(x)))
        # forward Block 2
        x = F.relu(self.pool(self.conv2(x)))
        # forward Block 3
        x = F.relu(self.pool2(self.conv3(x)))
        # forward Block 4
        x = F.relu(self.pool2(self.conv4(x)))
        # forward Block 5
        x = F.relu(self.pool2(self.conv5(x)))
        # map to output of 10
        x = self.flatten(x)
        x = self.fc1(x)
        return x
