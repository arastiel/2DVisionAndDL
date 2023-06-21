import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import ConvNet as cn
import ConvNetBatchNorm as cnb


def prep_training_data():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                             shuffle=False, num_workers=2)
    return trainloader, testloader


if __name__ == '__main__':
    print(" Choose wich Net to train ")
    print(" 1) ConvNet ")
    print(" 2) ConvNetBatchNorm ")
    print(" Please input corresponding number:")
    num = input()
    # Here is Logic for training
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # init model with available device
    if num == "1":
        net = cn.ConvNet()
    elif num == "2":
        net = cnb.ConvNetBatchNorm()
    else:
        print(" incorrect input")
    net.to(device)

    print(net)

    # prep trainingdata
    trainloader, _ = prep_training_data()

    # Define a Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(2):  # loop over the dataset multiple times

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
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')
    if num == "1":
        PATH = './Convnet_cifar_net.pth'
    else:
        PATH = './ConvNetBatchNorm_cifar_net.pth'
    torch.save(net.state_dict(), PATH)
