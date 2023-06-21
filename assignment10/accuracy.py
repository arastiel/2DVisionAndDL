import ConvNet as cn
import ConvNetBatchNorm as cnb
import training as tr
import torch


# ConvNet has a higher loss compared to ConvNetBatchNorm and the accuracy of ConvNetBathedNorm is higher

def printModelSize(model):  # ???? idk
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    print("Modelsize is: ", pp)


if __name__ == '__main__':
    print(" Choose wich Net to print accuracy ")
    print(" 1) ConvNet ")
    print(" 2) ConvNetBatchNorm ")
    print(" Please input corresponding number:")
    num = input()

    if num == "1":
        PATH = './Convnet_cifar_net.pth'
        net = cn.ConvNet()
    elif num == "2":
        PATH = './ConvNetBatchNorm_cifar_net.pth'
        net = cnb.ConvNetBatchNorm()
    else:
        print(" incorrect input")

    # init saved network
    net.load_state_dict(torch.load(PATH))

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    _, testloader = tr.prep_training_data()

    # Accuracy of the network on the 10000 test images and by class
    correct = 0
    total = 0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))
    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))
    printModelSize(net)
