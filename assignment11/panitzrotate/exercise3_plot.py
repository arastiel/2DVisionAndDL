import torch
import torch.nn as nn
from PIL import Image
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import random
import time
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

# general settings
BATCH_SIZE = 100

SCRATCH = True

# settings for Model 1: train a CNN from scratch
if SCRATCH:

    IMAGE_SIZE = (64,64,3)
    LRATE = None # FIXME
    # ...
    
# settings for Model 2: use transfer learning
else:

    IMAGE_SIZE = (224,224,3)
    LRATE = None # FIXME
    # ...



class PRDataset():
    ''' 
    This dataset reads jpg images from a given directory (e.g. 'imgs/train') 
    and returns batches of randomly rotated images. Each item in the batch is
    a triple

                      (img, x, t)

    where
    - img is a randomly picked image, rotated by 0,90,180 or 270 degrees,
          as a numpy array.
    - x   is a torch.tensor representation of the image  (which can be used as 
          input for a CNN. x is obtained from 'img' using a PyTorch DataTransform.
    - t   is the image's true orientation (0=0 degrees, 1=90 degrees, 2=180 degrees,
          3=270 degrees).

    The dataset also offers a method to rotate an image to a given orientation and back,
    which you can use for visualizing your results.
    '''

    def __init__(self, directory, image_size, transform=None):
        ''' the constructor.

        @type directory: str
        @param directory: the input folder with image (e.g., 'imgs/train')
        @type image_size: tuple
        @param image_size: a tuple (W,H,C), where W,H are the image dimensions
                           and C is the number of channels. For Exercise 1, use (64,64,3).
        @type transform: a PyTorch DataTransform
        @param transform: see torchvision.transforms. Using ToTensor() can be used
                          to transform a (W,H,3) numpy image into a (3,W,H) tensor
                          as input for a CNN and also scale values from 0,...,255 to [0,1]
        '''
        self.imgs = self._read_imagefolder(directory, image_size)
        self.transform = transform


    def _read_imagefolder(self, directory, image_size):
        ''' given a folder, read all jpegs in it and scale them to the desired size.

        @type directory: str
        @param directory: the input folder with image (e.g., 'imgs/train')
        @type image_size: tuple
        @param image_size: a tuple (W,H,C), where W,H are the image dimensions
                           and C is the number of channels. For Exercise 1, use (64,64,3).
        @rtype: np.array
        @returns: if the folder contains N images, a numpy array of shape (N,W,H,C) is returned.
        '''

        def _read_jpg(path, image_size):
            '''reads a JPEG and returns a numpy array of size 'size'.'''
            img = Image.open(path)
            w,h,_ = image_size
            img = img.resize((w,h), Image.ANTIALIAS)
            img = np.asarray(img)
            return img

        paths = [directory + os.sep + filename
                 for filename in os.listdir(directory)
                 if filename.endswith('.jpg')]
    
        return np.asarray([_read_jpg(path, image_size) for path in paths])

    
    def rotate_img(self, img, ori, forward=True):
        ''' given an image, rotate it according to 'ori' by 0,90,180 or 270 degrees.

        @type img: np.array
        @param img: the image to rotate.
        @type ori: int
        @param ori: can be 0 : rotate by 0 degrees,
                           1 : rotate by 90 degrees,
                           2 : rotate by 180 degrees, or
                           3 : rotate by 270 degrees
        @type forward: bool
        @param forward: if forward=False, the image is rotated backward instead
                        of forward.
        @rtype: np.array
        @returns: the rotated image.
        '''
        if ori==0:
            pass
        elif ori==1:
            img = np.rot90(img, k=(1 if forward else 3), axes=[0,1])
        elif ori==2:
            img = np.rot90(img, k=2, axes=[0,1])
        elif ori==3:
            img = np.rot90(img, k=(3 if forward else 1), axes=[0,1])

        return img

    
    def _sample(self):
        ''' internal method. returns a random image rotated to a random orientation. '''
        img = random.choice(self.imgs)
        ori = random.randint(0,3)
        img = self.rotate_img(img, ori).copy()
        return img, ori

    
    def __getitem__(self, index):
        ''' 
        @rtype: tuple(np.array, torch.tensor, int)
        @returns: a tuple (img,x,t), where img is a randomly picked (and randomly
            rotated) image. x is the corresponding tensor (which can be used
            as input for PyTorch CNN models), and t is the target (i.e., the image's
            true orientation).
        '''
        img,t = self._sample()
        x = self.transform(img) if self.transform else None
        return img,x,t

    def __len__(self):
        ''' since we draw random samples from self.imgs, this parameter 
            is kinda arbitrary. '''
        return 10 * len(self.imgs)




class PRNet(nn.Module):
    '''
    Exercise 1: implement your own CNN here. 
    '''
    
    def __init__(self, _F=64):
        '''
        the class constructor. 

        @type F: int
        @param F: the base number of channels (see exercise sheet).
        '''
        super(PRNet, self).__init__()

        self.F = _F

        # Block 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.F, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 2
        self.conv2 = nn.Conv2d(in_channels=self.F, out_channels=self.F * 2, kernel_size=3, stride=2, padding=1)

        # Block 3
        self.conv3 = nn.Conv2d(in_channels=self.F * 2, out_channels=self.F * 4, kernel_size=3, stride=2, padding=1)


        self.flatten = nn.Flatten()
        self.out = nn.Linear(self.F * 4, 4)
        self.softmax = nn.Softmax()

        #raise NotImplementedError()
    
        
    def forward(self, x):
        '''
        the forward pass. takes a batch of tensors and outputs probabilities
        for the 4 orientations.

        @type F: x
        @param F: a batch of images, as a torch.tensor of shape (BATCH_SIZE,3,W,H).
        @rtype: torch.tensor
        @param F: for each image, the probability of the 4 orientations, in form of 
                  a torch tensor of size (BATCH_SIZE,4).
        '''

        # for each block: convolution -> maxpooling -> relu
        # forward Block 1
        x = self.pool(F.relu(self.conv1(x)))
        # forward Block 2
        x = self.pool(F.relu(self.conv2(x)))
        # forward Block 
        x = self.pool(F.relu(self.conv3(x)))

        x = self.flatten(x)
        x = self.out(x)
        x = self.softmax(x)
        #print(x.size())
        return x

def plot(imgs, imgs_rot, path):
    ''' a help method to visualize your results: Give it two equally long lists 
        of rotated images, the first rotated according to your model, 
        the second according to the ground truth.
        The method will illustrate the images in a grid, with both versions
        of each image side-by-side. The plot will be stored to a file.

        !!! For plotting not to take too long, use lists of length <= 30. !!!

        @type imgs: list<np.array>
        @param imgs: a list of images, each a (W,H,3) numpy array.
        @type imgs_rot: list<np.array>
        @param imgs_rot: a list of images, each a (W,H,3) numpy array.
        @type path: str
        @param path: the path where to save the plot    
    '''        
        
    # create a NxN grid to plot the images.
    N = int(np.ceil(np.sqrt(2*len(imgs))))
    f, axarr = plt.subplots(nrows=N, ncols=N, figsize=(18,18))

    # insert images into the plot.
    for i in range(min(len(imgs),100)):
        axarr[2*i//N,2*i%N].imshow(imgs[i], 
                                   interpolation='nearest')
        axarr[(2*i+1)//N,(2*i+1)%N].imshow(imgs_rot[i], 
                                           interpolation='nearest')
            
    f.tight_layout()
    plt.savefig(path)
    plt.clf()
    plt.close()



def train(dataset, dataset_valid, model, optimizer):
    '''
    train your model.
    '''
    train_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=BATCH_SIZE)
    valid_loader = torch.utils.data.DataLoader(dataset=dataset_valid, batch_size=BATCH_SIZE)

    criterion = nn.CrossEntropyLoss()
    writer = SummaryWriter()        # for tensorboard - localhost:6006
    for epoch in range(2):  # loop over the dataset multiple times
        
        loss=0.0  
        correct=0.0  
        val_loss=0.0  
        val_correct=0.0 
        total = 0
        val_total = 0
        for i, data in enumerate(train_loader, 0):      #training part
            img, x, t = data
 
            outputs=model(x)  
            loss1=criterion(outputs, t)  
            optimizer.zero_grad()  
            loss1.backward()  
            optimizer.step()  
            _,preds=torch.max(outputs,1)
            total += t.size(0)
            loss+=loss1.item()  
            correct+=torch.sum(preds==t.data)
            writer.add_scalar("Loss/train", loss1, epoch)
            writer.add_scalar("Correct/train", correct, epoch)
            writer.add_scalar("Accuracy/Train", correct/total, epoch)

        else:  
            with torch.no_grad():  
                for i, data in enumerate(valid_loader, 0):      #validation part
                    img, x, t = data

                    val_outputs=model(x)  
                    val_loss1=criterion(val_outputs,t)   
                    _,val_preds=torch.max(val_outputs,1)  
                    val_loss+=val_loss1.item()  
                    val_correct+=torch.sum(val_preds==t.data) 
                    val_total += t.size(0)

                    writer.add_scalar("Loss/Validation", val_loss1, epoch)
                    writer.add_scalar("Correct/Validation", val_correct, epoch)
                    writer.add_scalar("Accuracy/Validation", val_correct/val_total, epoch)

    writer.flush()
    writer.close()
    print('Finished Training')
    #PATH = './prnet_test.pth'
    #torch.save(model.state_dict(), PATH)
    return
    #raise NotImplementedError() # FIXME
    
def rotate_truth(imgs):     # rotating according to ground truth
    return np.array([np.rot90(img, k=4-ori, axes=[0, 1]) for img, ori in imgs])


def rotate_prediction(imgs, model, transform):      # rotating according to model prediction
    rotated = []
    for data in imgs:
        img = data[0]
        x = transform(img) if transform else None
        outputs = model(x[None, ...])   #extend x with None (replacing batch size)
        _, pred = torch.max(outputs,1)
        ori = pred.item()

        img = np.rot90(img, k=4-ori, axes=[0, 1])
        rotated.append(img)

    return np.array(rotated)



if __name__ == '__main__':

    # Model 1

    model = PRNet()
    transform = transforms.ToTensor()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        
    train_directory = 'imgs' + os.sep + 'train'

    valid_directory = 'imgs' + os.sep + 'valid'

    dataset = PRDataset(train_directory, IMAGE_SIZE, transform)
    dataset_valid = PRDataset(valid_directory, IMAGE_SIZE, transform)

    train(dataset, dataset_valid, model, optimizer)

    #create some images to test..
    random_imgs = []
    only_imgs = []
    for i in range(12):
        img, ori = dataset_valid._sample()
        random_imgs.append([img, ori])
        only_imgs.append(img)

    random_imgs = np.array(random_imgs)
    only_imgs = np.array(only_imgs)
    
    truth_imgs = rotate_truth(random_imgs)
    pred_imgs = rotate_prediction(random_imgs, model, transform)
    
    path = "./comparison_truth_prediction.png"
    plot(pred_imgs, truth_imgs, path)