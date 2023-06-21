import numpy as np
import torch
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim

# Resources:
# https://pytorch.org/tutorials/beginner/pytorch_with_examples.html
# https://towardsdatascience.com/handwritten-digit-mnist-pytorch-977b5338e627
# https://pytorch.org/docs/stable/optim.html

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
# Number of neurons on the hidden layer(?)
hidden_size = 128
# Output layer consists of probabilities for each of the 10 classes (0, 1, 2, ..., 9)
output_size = 10

# Use LogSoftmax activation function because it is a classification problem
model = nn.Sequential(nn.Linear(input_size, hidden_size),
                      nn.ReLU(),
                      nn.Linear(hidden_size, output_size),
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

# Optimizer-Dict to choose from
optimizers = {"SGD":        optim.SGD(model.parameters(),
                                      lr=1e-3,
                                      momentum=0.9),

              "Adadelta":   optim.Adadelta(model.parameters(),
                                           lr=1e-3,
                                           rho=0.9,
                                           eps=1e-06,
                                           weight_decay=0),

              "Adagrad":    optim.Adagrad(model.parameters(),
                                          lr=1e-3,
                                          lr_decay=0,
                                          weight_decay=0,
                                          initial_accumulator_value=0,
                                          eps=1e-06),

              "Adam":       optim.Adam(model.parameters(),
                                       lr=1e-3,
                                       weight_decay=0),

              "RMSprop":    optim.RMSprop(model.parameters(),
                                          lr=1e-3,
                                          alpha=0.99,
                                          eps=1e-06,
                                          weight_decay=0,
                                          momentum=0,
                                          centered=False),

              "LBFGS":      optim.LBFGS(model.parameters(),
                                        lr=1e-3)
              }

time0 = time()
epochs = 15

for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        # Flatten MNIST images into a 784 long vector
        images = images.view(images.shape[0], -1)

        # Picking an optimizer
        optimizer = optimizers["LBFGS"]

        # No closure needed in step()
        if optimizer != optimizers["LBFGS"]:
            optimizer.step()

            # Training pass
            optimizer.zero_grad()

            output = model(images)
            loss = criterion(output, labels)

            # This is where the model learns by backpropagating
            loss.backward()

        # If optimizer is LFBGS or Conjugate Gradient we need a closure to reevaluate
        else:
            def closure():
                optimizer.zero_grad()
                output = model(images)
                loss = criterion(output, labels)
                loss.backward()
                return loss

            optimizer.step(closure)

        # And optimizes its weights here
        running_loss += loss.item()

    else:
        print("Epoch {} - Training loss: {}".format(e, running_loss / len(trainloader)))
        print("\nTraining Time (in minutes) =", (time() - time0) / 60)

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
        if true_label == pred_label:
            correct_count += 1
        all_count += 1

print("Number Of Images Tested =", all_count, "\n")
print("Model Accuracy =", (correct_count / all_count))

# Average behavior and results after testing 10 * 10000 images:

#           Model Accuracy in % | Training Time in Minutes
#           --------------------|-------------------------
# SGD:      97.33               | 2.276
# Adadelta: 87.42               | 2.313
# Adagrad:  91.88               | 2.235
# Adam:     97.58               | 2.591
# RMSprop:  96.88               | 2.452
# LBFGS:    -/-                 | -/-

# SGD: -----------------------------------------------------------------------------------------------------------------
# Simply implements the stochastic gradient descent (optionally with momentum).

# see also:
# https://en.wikipedia.org/wiki/Stochastic_gradient_descent
# https://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf


# Adadelta: ------------------------------------------------------------------------------------------------------------
# The method dynamically adapts over time using only first order information and has minimal computational overhead
# beyond vanilla stochastic gradient descent. The method requires no manual tuning of a learning rate and appears robust
# to noisy gradient information, different model architecture choices, various data modalities and selection of
# hyperparameters.

# see also: https://arxiv.org/abs/1212.5701

# Adam: ----------------------------------------------------------------------------------------------------------------
# The method is straightforward to implement, is computationally efficient, has little memory requirements, is invariant
# to diagonal rescaling of the gradients, and is well suited for problems that are large in terms of data and/or
# parameters. The method is also appropriate for non-stationary objectives and problems with very noisy and/or sparse
# gradients. The hyper-parameters have intuitive interpretations and typically require little tuning.

# see also: https://arxiv.org/abs/1412.6980

# Adagrad: -------------------------------------------------------------------------------------------------------------
# Describes and analyzes an apparatus for adaptively modifying the proximal function, which significantly simplifies
# setting a learning rate and results in regret guarantees that are provably as good as the best proximal function that
# can be chosen in hindsight.

# see also: https://jmlr.org/papers/volume12/duchi11a/duchi11a.pdf

# RMSprop: -------------------------------------------------------------------------------------------------------------
# The implementation here takes the square root of the gradient average before adding epsilon (note that TensorFlow
# interchanges these two operations). The effective learning rate is thus α/(sqrt(v)+ϵ) where α is the scheduled
# learning rate and v is the weighted moving average of the squared gradient.

# see also: https://arxiv.org/pdf/1308.0850v5.pdf

# LBFGS: ---------------------------------------------------------------------------------------------------------------
# Some optimization algorithms such as Conjugate Gradient and LBFGS need to reevaluate the function multiple times, so
# you have to pass in a closure that allows them to recompute your model. The closure should clear the gradients,
# compute the loss, and return it.

# Heavily inspired by Matlab's "minFunc". minFunc is a Matlab function for unconstrained optimization of differentiable
# real-valued multivariate functions using line-search methods. It uses an interface very similar to the Matlab
# Optimization Toolbox function fminunc, and can be called as a replacement for this function.
# Further it can optimize problems with a much larger number of variables, and uses a line search that is robust to
# several common function pathologies.

# see also: https://www.cs.ubc.ca/~schmidtm/Software/minFunc.html
