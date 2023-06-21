import torch.nn as nn
import torch.nn.functional as F


class ConvNetBatchNorm(nn.Module):
    def __init__(self):
        super(ConvNetBatchNorm, self).__init__()
        # Block 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3, stride=1, padding=0)
        self.batch1 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 2
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=512, kernel_size=3, stride=1, padding=0)
        self.batch2 = nn.BatchNorm2d(512)
        # Block 3
        self.conv3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=0)
        # Block 4
        self.conv4 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)

        # Block 5
        self.conv5 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=2)
        self.flatten = nn.Flatten()
        self.out = nn.Linear(512, 10)

    def forward(self, x):
        # for each block: convolution -> batch -> maxpooling -> relu
        # forward Block 1
        x = F.relu(self.pool(self.batch1(self.conv1(x))))
        # forward Block 2
        x = F.relu(self.pool(self.batch2(self.conv2(x))))
        # forward Block 2
        x = F.relu(self.pool(self.batch2(self.conv3(x))))
        # forward Block 2
        x = F.relu(self.pool(self.batch2(self.conv4(x))))
        # forward Block 2
        x = F.relu(self.pool(self.batch2(self.conv5(x))))
        # map to output of 10
        x = self.flatten(x)
        x = self.out(x)

        return x
