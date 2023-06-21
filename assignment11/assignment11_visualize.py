import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch

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

def plot_kernels(tensor, num_cols=16):
    num_filters = tensor.shape[0]       #128 filter
    num_rows = 1+ num_filters // num_cols   
    fig = plt.figure(figsize=(num_cols,num_rows))   #8x16 grid
    for i in range(tensor.shape[0]):
        ax1 = fig.add_subplot(num_rows,num_cols,i+1)
        ax1.imshow(tensor[i])
        ax1.axis('off')
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])

    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.savefig("./visualize.png")
    plt.show()

if __name__ == '__main__':
    net = ConvNet()
    """
    #UNTRAINED CONVNET
    tensor = net.conv1.weight.data.numpy()*5    #multiply to get more brightness..
    plot_kernels(tensor)
    """

    #TRAINED CONVNET
    PATH = './assignment10_Convnet_cifar_net.pth'
    trained_net = ConvNet()
    trained_net.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))
    tensor = net.conv1.weight.data.numpy()*5
    plot_kernels(tensor)

# missing: compare filters, patterns?
# 3p