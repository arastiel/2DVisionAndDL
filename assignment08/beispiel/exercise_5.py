from pathlib import Path
import requests
import pickle
import numpy as np
import gzip
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

# basiert auf https://pytorch.org/tutorials/beginner/nn_tutorial.html
# und https://pytorch.org/tutorials/recipes/recipes/defining_a_neural_network.html

class Mnist_Model(nn.Module):
    def __init__(self):
        super().__init__()
        # input layer to hidden layer, nutzen 50 neuronen im hidden layer
        self.l1 = nn.Linear(784, 50)
        # hidden layer to output (10 classes for nums 0 to 9), nutzen 50 neuronen im hidden layer
        self.l2 = nn.Linear(50, 10)
        # dropout to avoid overfitting
        self.dropout = nn.Dropout(0.5)  # not needed here

    def forward(self, xb):
        xb = self.l1(xb)
        xb = F.relu(xb)
        xb = self.dropout(xb)
        xb = self.l2(xb)

        return xb

def get_model():
    model = Mnist_Model()
    # using sgd as optimizer
    return model, optim.SGD(model.parameters(), lr=lr)

def loss_batch(model, loss_func, xb, yb, opt=None):
    loss = loss_func(model(xb), yb)
    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb)

def fit(epochs, model, loss_func, opt, train_dl, valid_dl):
    for epoch in range(epochs):
        print("Epoche:  ", str(epoch))

        model.train()
        losses, nums = zip(
            # .to(dev), um die eingabe- und ausgabedaten durch entweder cpu oder gpu nutzbar zu machen,
            # je nach dem was verfügbar ist
            *[loss_batch(model, loss_func, xb.to(dev), yb.to(dev), opt) for xb, yb in train_dl]
        )

        training_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
        print("Loss-Funktion (Training):    %.3f" % training_loss )

        model.eval()
        with torch.no_grad():
            losses, nums = zip(
                *[loss_batch(model, loss_func, xb.to(dev), yb.to(dev)) for xb, yb in valid_dl]
            )
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)

        print("Loss-Funktion (Validation):  %.3f" % val_loss)
        print("--------------------")


def get_data(train_ds, valid_ds, bs):
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True),
        DataLoader(valid_ds, batch_size=bs * 2),
    )

bs = 64         # batch size
lr = 0.5        # learning rate
epochs = 5      # how many epochs to train for

# wenn cuda device verfügbar,
# nutze es
dev = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")

# ----------------------
# MNIST Daten hineinladen
DATA_PATH = Path("data")
PATH = DATA_PATH / "mnist"

PATH.mkdir(parents=True, exist_ok=True)

URL = "https://github.com/pytorch/tutorials/raw/master/_static/"  # mnist is included in torchvision
FILENAME = "mnist.pkl.gz"

if not (PATH / FILENAME).exists():
        content = requests.get(URL + FILENAME).content
        (PATH / FILENAME).open("wb").write(content)

with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
        ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")

x_train, y_train, x_valid, y_valid = map(
    torch.tensor, (x_train, y_train, x_valid, y_valid)
)
# ----------------------

# using cross entropy loss function
loss_func = F.cross_entropy

train_ds = TensorDataset(x_train, y_train)
valid_ds = TensorDataset(x_valid, y_valid)
train_dl, valid_dl = get_data(train_ds, valid_ds, bs)
model, opt = get_model()
# model entweder durch cpu oder gpu nutzbar
model.to(dev)
fit(epochs, model, loss_func, opt, train_dl, valid_dl)
# 4p