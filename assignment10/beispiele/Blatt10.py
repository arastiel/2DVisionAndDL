import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torch.optim as optim


#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
epochs = 3
batch_size = 4
learning_rate = 0.001


transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

#download and load datesets 
train_dataset = torchvision.datasets.CIFAR10(root='./data', train = True, download = True, transform = transform)

test_dataset = torchvision.datasets.CIFAR10(root='./data', train = False, download = True, transform = transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True)

test_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = False)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # first convolution layer, 3 color input channels, 128 filters, 3 kernel size
        self.conv1 = nn.Conv2d(3, 128, 3, padding=(1,1))
        # kernel size 2, stride 2
        self.pool = nn.MaxPool2d(2,2)
        # second convolution layer, 128 input channels, 512 output channels, 3 kernel size
        self.conv2 = nn.Conv2d(128,512, 3, padding=(1,1))
        # third convolution layer, 512 input channels, 512 output channels, 3 kernel size
        self.conv3 = nn.Conv2d(512,512, 3, padding=(1,1))
        self.conv4 = nn.Conv2d(512,512, 3, padding=(1,1))
        self.conv5 = nn.Conv2d(512,512, 3, padding=(1,1))
        #fully conacted layer, 512 1*1 pictures, 512 neurons        
        self.fc1 = nn.Linear(512*1*1, 512)
        self.fc2 = nn.Linear(512, 10)
        
    def forward(self, x):
        x = F.relu(self.pool(self.conv1(x)))
        x = F.relu(self.pool(self.conv2(x)))
        x = F.relu(self.pool(self.conv3(x)))
        x = F.relu(self.pool(self.conv4(x)))
        x = F.relu(self.pool(self.conv5(x)))
        # flattens the 512 1*1 pictures
        x = x.view(-1, 512*1*1)
        x = F.relu(self.fc1(x))
        # softmax is included in nn.CrossEntropyLoss() 
        x = self.fc2(x)
        return x


model = ConvNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr = learning_rate)
n_total_steps = len(train_loader)

for epoch in range(epochs):
    for i, (images,labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)  
        #forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        #backward and optimizer
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        if (i+1) % 2000 == 0:
            print (f'Epoch [{epoch+1}/{epochs}], Step[{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')
            
print('FINISHED Training')

with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct  =[0 for i in range(10)]
    n_class_samples  =[0 for i in range(10)]
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        
        _, predicted = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()
        
        for i in range(batch_size):
            label = labels[i]
            pred = predicted[i]
            if label == pred:
                n_class_correct[label] += 1
            n_class_samples[label] += 1
            
    acc = 100.0 * n_class_correct[i] / n_class_samples[i]
    
    for i in range(10):
        acc = 100.0 * n_class_correct[i] / n_class_samples[i]
        print(f'Accuracy of {classes[i]} : {acc} % ')


PATH = './cifar_net.pth'
torch.save(model.state_dict(), PATH)