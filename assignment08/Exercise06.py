import numpy as np
import torch
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn

# Resources:
# https://pytorch.org/tutorials/beginner/pytorch_with_examples.html
# https://towardsdatascience.com/handwritten-digit-mnist-pytorch-977b5338e627

# Convert image to PyTorch Tensor and normalize
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)), ])

# Download, shuffle and transform data sets
trainset = datasets.MNIST('train_set_path', download=True, train=True, transform=transform)
valset = datasets.MNIST('test_set_path', download=True, train=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True)

dataiter = iter(trainloader)
images, labels = dataiter.next()


def view_classify(img, ps):
    '''
    Function for viewing an image and it's predicted classes.
    '''
    ps = ps.data.numpy().squeeze()

    fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)
    ax1.imshow(img.resize_(1, 28, 28).numpy().squeeze())
    ax1.axis('off')
    ax2.barh(np.arange(10), ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(10))
    ax2.set_yticklabels(np.arange(10))
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)
    plt.tight_layout()
    plt.show()

figure = plt.figure()
num_of_images = 60
for index in range(1, num_of_images + 1):
    plt.subplot(6, 10, index)
    plt.axis('off')
    plt.imshow(images[index].numpy().squeeze(), cmap='gray_r')


# Input size = 784 = 28 x 28 Pixels
input_size = 784
# Number of neurons on the hidden layer(?) # yes!
hidden_sizes = [30, 25, 20, 15, 10]
# Output layer consists of probabilities for each of the 10 classes (0, 1, 2, ..., 9)
output_size = 10


linear1 = nn.Linear(input_size, hidden_sizes[0])
linear2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
linear3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
linear4 = nn.Linear(hidden_sizes[2], hidden_sizes[3])
linear5 = nn.Linear(hidden_sizes[3], hidden_sizes[4])
linear6 = nn.Linear(hidden_sizes[4], output_size)

layers = [linear1, linear2, linear3, linear4, linear5, linear6]

# Task 1 | Xavier initialization, see: ---------------------------------------------------------------------------------
# https://discuss.pytorch.org/t/how-to-initialize-the-conv-layers-with-xavier-weights-initialization/8419
for i in range(len(layers)):
    nn.init.xavier_uniform_(layers[i].weight)

# Task 3 | Added Batch Normalization -----------------------------------------------------------------------------------
# Task 4 | Added dropout to every layer --------------------------------------------------------------------------------

model = nn.Sequential(nn.Dropout(p=0.1),
                      linear1,
                      #nn.BatchNorm1d(hidden_sizes[0]),
                      nn.ReLU(),
                      nn.Dropout(p=0.1),
                      linear2,
                      #nn.BatchNorm1d(hidden_sizes[1]),
                      nn.ReLU(),
                      nn.Dropout(p=0.1),
                      linear3,
                      #nn.BatchNorm1d(hidden_sizes[2]),
                      nn.ReLU(),
                      nn.Dropout(p=0.2),
                      linear4,
                      #nn.BatchNorm1d(hidden_sizes[3]),
                      nn.ReLU(),
                      nn.Dropout(p=0.2),
                      linear5,
                      #nn.BatchNorm1d(hidden_sizes[4]),
                      nn.ReLU(),
                      nn.Dropout(p=0.2),
                      linear6,
                      nn.LogSoftmax(dim=1))


# NLLLoss() => Negative log-likelihood loss
criterion = nn.NLLLoss()

images, labels = next(iter(trainloader))
images = images.view(images.shape[0], -1)

# log probabilities
logps = model(images)

# calculate the NLL loss
loss = criterion(logps, labels)

# perform gradient descent and update the weights by back-propagation
loss.backward()

# Task 1 | L2 regularization happens here, see: ------------------------------------------------------------------------
# https://towardsdatascience.com/l1-and-l2-regularization-methods-ce25e7fc831c
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

time0 = time()
epochs = 15
for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        # Flatten MNIST images into a 784 long vector
        images = images.view(images.shape[0], -1)

        # Training pass
        optimizer.zero_grad()

        output = model(images)
        loss = criterion(output, labels)

        # This is where the model learns by backpropagating
        loss.backward()

        # And optimizes its weights here
        optimizer.step()

        running_loss += loss.item()
    else:
        print("Epoch {} - Training loss: {}".format(e, running_loss / len(trainloader)))
        print("\nTraining Time (in minutes) =", (time() - time0) / 60)

        # Task 1 | Create a checkpoint to save the model ---------------------------------------------------------------
        torch.save({
            'epoch': e,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, "model.pt")

images, labels = next(iter(valloader))

img = images[0].view(1, 784)

with torch.no_grad():
    logps = model(img)

ps = torch.exp(logps)
probab = list(ps.numpy()[0])
print("Predicted Digit =", probab.index(max(probab)))

view_classify(img.view(1, 28, 28), ps)

correct_count, all_count = 0, 0
for images, labels in valloader:
    for i in range(len(labels)):
        img = images[i].view(1, 784)
        with torch.no_grad():
            logps = model(img)

        ps = torch.exp(logps)
        probab = list(ps.numpy()[0])
        pred_label = probab.index(max(probab))
        true_label = labels.numpy()[i]
        if (true_label == pred_label):
            correct_count += 1
        all_count += 1

print("Number Of Images Tested =", all_count, "\n")
print("Model Accuracy =", (correct_count / all_count))

# Task 3:   The model's loss converges a lot faster than without Batch Normalization. The model appears to be more solid

# Task 4:   Adding dropout does not seem to help reduce over-fitting. It decreases the overall accuracy of the model.
#           Loss is converging faster though.
