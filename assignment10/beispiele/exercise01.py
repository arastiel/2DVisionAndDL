import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
# https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
# wenn unter Windows ein Broken Pipe error auftritt, muss man im Modul "torch\utils\data\dataloader.py" num_workers auf 0 setzen

class ConvNet(nn.Module):
    """A Neural Network with 5 hidden layers"""

    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Conv2d(in_channels=3, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.layer2 = nn.Conv2d(in_channels=128, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.layer3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.layer4 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.layer5 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.output = nn.Linear(512, 10) # ich glaub hier fehlt ein layer, nn.Linear(512,512)
        self.pooling = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

    def forward(self, out):
        out = self.layer1(out)
        out = self.pooling(out)
        out = F.relu(out)
        out = self.layer2(out)
        out = self.pooling(out)
        out = F.relu(out)
        out = self.layer3(out)
        out = self.pooling(out)
        out = F.relu(out)
        out = self.layer4(out)
        out = self.pooling(out)
        out = F.relu(out)
        out = self.layer5(out)
        out = self.pooling(out)
        out = out.view(-1, self.num_flat_features(out))
        out = F.relu(out)
        out = self.output(out)
        return out

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def PrintModelSize(self):
        i = 0
        num_params = 0
        my_iter = self.parameters()
        for i in my_iter:
            num_params += np.prod(i.shape)

        print(num_params)
        return num_params


# Hyperparameters
num_epochs = 2
momentum = 0.9
learning_rate = 0.001
batchsize = 10

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=5000,
                                           shuffle=True, num_workers=1)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(testset, batch_size=batchsize,
                                          shuffle=True, num_workers=1)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

model = ConvNet()
model.PrintModelSize()

criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def save_checkpoint(state, filename):
    print("--Saving Checkpoint--")
    torch.save(state, filename)


def load_checkpoint(filename):
    print("-- loading checkpoint --")
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])


# get some random training images
dataiter = iter(train_loader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(batchsize)))

#load_checkpoint("cnn_model_CIFAR-10_epoch0.pth.tar")

for epoch in range(1):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        print("i:", i)
        if i % 2000 == 1999:  # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

checkpoint = {'state_dict' : model.state_dict(), 'optimizer':optimizer.state_dict()}
save_checkpoint(checkpoint, f"cnn_model_CIFAR-10_epoch{epoch}.pth.tar")

print('Finished Training')

dataiter = iter(test_loader)
images, labels = dataiter.next()

# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(batchsize)))

# outputs = model(images)

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        # print("predicted: ", predicted)
        # print("c: ", c)
        print("outputs: ", outputs)
        for i in range(batchsize):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))
