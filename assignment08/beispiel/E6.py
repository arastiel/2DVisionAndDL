import torch
import math
import time
import torch.utils
from torch.utils.tensorboard import SummaryWriter
from torch.utils import data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torchvision.datasets.mnist as mnist
import torchvision
import torchvision.transforms as transforms
import numpy as np

# From https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html?highlight=mnist
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

batch_size = 8

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5), (0.5))])


trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=0)

classes = tuple(trainloader.dataset.classes)

def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# show images
# imshow(torchvision.utils.make_grid(images))
# print labels
# print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28*28, 10*10)      # mapping from image to hidden layer
        self.bn1 = nn.BatchNorm1d(100)          # batch normalization
        self.d1 = nn.Dropout()                  # dropout
        self.fc2 = nn.Linear(10 * 10, 10 * 10)  # first hidden to second hidden layer
        self.bn2 = nn.BatchNorm1d(100)          # batch normalization
        self.d2 = nn.Dropout()
        self.fc3 = nn.Linear(10 * 10, 10 * 10)  # second hidden to third hidden layer
        self.bn3 = nn.BatchNorm1d(100)          # batch normalization
        self.d3 = nn.Dropout()
        self.fc4 = nn.Linear(10 * 10, 10 * 10)  # third hidden to fourth hidden layer
        self.bn4 = nn.BatchNorm1d(100)          # batch normalization
        self.d4 = nn.Dropout()
        self.fc5 = nn.Linear(10 * 10, 10 * 10)  # fourth hidden to fifth hidden layer
        self.bn5 = nn.BatchNorm1d(100)          # batch normalization
        self.d5 = nn.Dropout()
        self.fc6 = nn.Linear(10*10, 10)         # mapping from fifth hidden to output layer

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.d1(self.bn1(self.fc1(x))))
        x = F.relu(self.d1(self.bn2(self.fc2(x))))
        x = F.relu(self.d1(self.bn3(self.fc3(x))))
        x = F.relu(self.d1(self.bn4(self.fc4(x))))
        x = F.relu(self.d1(self.bn5(self.fc5(x))))
        x = F.relu(self.fc6(x))
        return x

net = Net()
net.to(device)

criterion = nn.CrossEntropyLoss()  # loss term supports mini-batches
# optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0)
# betas are decay rates
# weight_decay is L2 regularisation
optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

for epoch in range(2):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999: # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

PATH = './mnist_net.pth'
torch.save(net.state_dict(), PATH)

dataiter = iter(testloader)
images, labels = dataiter.next()

# print images
# imshow(torchvision.utils.make_grid(images))
print('GrounTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))

net = Net()
net.load_state_dict(torch.load(PATH))
outputs = net(images)

_, predicted = torch.max(outputs, 1)
print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(batch_size)))

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))