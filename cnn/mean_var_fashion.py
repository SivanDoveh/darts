import argparse
import torch
import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F
from torch.autograd import Variable

import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import torchvision.models as models

import os
import sys
import math
import numpy as np

data = dset.FashionMNIST(root='FashionMNIST', train=True, download=True,transform=transforms.ToTensor()).train_data

data=np.asarray(data)#.numpy()
data = data.astype(np.float32)/255.

means = []
stdevs = []
for i in range(1):
    pixels = data[:,:,:].ravel()
    #pixels = data
    means.append(np.mean(pixels))
    stdevs.append(np.std(pixels))

print("means: {}".format(means))
print("stdevs: {}".format(stdevs))
print('transforms.Normalize(mean = {}, std = {})'.format(means, stdevs))