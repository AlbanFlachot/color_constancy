#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 15:31:29 2018

@author: alban
"""
from __future__ import print_function, division
import os
import torch

from skimage import io, transform
import numpy as np
from torch.utils import data
from torchvision import transforms, utils
import pickle
#from my_classes import Dataset

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

import torch.nn as nn
import torch.nn.functional as F


class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5, stride = 2,padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3,padding=0)
        self.conv3 = nn.Conv2d(32, 64, 3,padding=0)
        self.fc1 = nn.Linear(64 * 14 * 14, 250)
        self.fc2 = nn.Linear(250, 250)
        self.fc3 = nn.Linear(250, 7)
        self.dropout = nn.Dropout2d(p=0.4)
	
    def forward(self, x):
        CONV1 = F.relu(self.conv1(x))
        POOL1 = self.pool(CONV1)
        CONV2 = F.relu(self.conv2(POOL1))
        POOL2 = self.pool(CONV2)
        CONV3 = F.relu(self.conv3(POOL2))
        POOL3 = self.pool(CONV3)
        POOL3_flat = POOL3.view(-1, 64 * 14 * 14)
        FC1 = F.relu(self.fc1(POOL3_flat))
        FC2 = F.relu(self.fc2(FC1))
        DROP = self.dropout(FC2)
        FC3 = self.fc3(DROP)     
        FEATURES = {'conv1': CONV1,
                    'conv2': CONV2,
                    'conv3': CONV3,
                    'pool3': POOL3,
                    'fc1': FC1,
                    'fc2': FC2,
                    'out': FC3}
        return FEATURES
    
class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 7, stride = 2,padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3,padding=0)
        self.conv3 = nn.Conv2d(32, 64, 3,padding=0)
        self.fc1 = nn.Linear(64 * 6 * 6, 250)
        self.fc2 = nn.Linear(250, 250)
        self.fc3 = nn.Linear(250, 7)
        self.dropout = nn.Dropout2d(p=0.4)
	
    def forward(self, x):
        CONV1 = F.relu(self.conv1(x))
        POOL1 = self.pool(CONV1)
        CONV2 = F.relu(self.conv2(POOL1))
        POOL2 = self.pool(CONV2)
        CONV3 = F.relu(self.conv3(POOL2))
        POOL3 = self.pool(CONV3)
        POOL3_flat = POOL3.view(-1, 64 * 6 * 6)
        FC1 = F.relu(self.fc1(POOL3_flat))
        FC2 = F.relu(self.fc2(FC1))
        DROP = self.dropout(FC2)
        FC3 = self.fc3(DROP)
        FEATURES = {'conv1': CONV1,
                    'conv2': CONV2,
                    'conv3': CONV3,
                    'pool3': POOL3,
                    'fc1': FC1,
                    'fc2': FC2,
                    'out': FC3}
        return FEATURES
        
class Net3(nn.Module):
    def __init__(self):
        super(Net3, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, stride = 2,padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 128, 3,padding=0)
        self.conv3 = nn.Conv2d(128, 256, 3,padding=0)
        self.fc1 = nn.Linear(256 * 6 * 6, 1000)
        self.fc2 = nn.Linear(1000, 1000)
        self.fc3 = nn.Linear(1000, 7)
        self.dropout = nn.Dropout2d(p=0.4)
	
    def forward(self, x):
        CONV1 = F.relu(self.conv1(x))
        POOL1 = self.pool(CONV1)
        CONV2 = F.relu(self.conv2(POOL1))
        POOL2 = self.pool(CONV2)
        CONV3 = F.relu(self.conv3(POOL2))
        POOL3 = self.pool(CONV3)
        POOL3_flat = POOL3.view(-1, 256 * 6 * 6)
        FC1 = F.relu(self.fc1(POOL3_flat))
        FC2 = F.relu(self.fc2(FC1))
        DROP = self.dropout(FC2)
        FC3 = self.fc3(DROP)
        FEATURES = {'conv1': CONV1,
                    'conv2': CONV2,
                    'conv3': CONV3,
                    'pool3': POOL3,
                    'fc1': FC1,
                    'fc2': FC2,
                    'out': FC3}
        return FEATURES

class Net4(nn.Module):
    def __init__(self):
        super(Net3, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, stride = 2,padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 128, 3,padding=0)
        self.conv3 = nn.Conv2d(128, 256, 3,padding=0)
        self.fc1 = nn.Linear(256 * 6 * 6, 1000)
        self.fc2 = nn.Linear(1000, 1000)
        self.fc3 = nn.Linear(1000, 7)
        self.dropout = nn.Dropout2d(p=0.4)
	
    def forward(self, x):
        CONV1 = F.relu(self.conv1(x))
        POOL1 = self.pool(CONV1)
        CONV2 = F.relu(self.conv2(POOL1))
        POOL2 = self.pool(CONV2)
        CONV3 = F.relu(self.conv3(POOL2))
        POOL3 = self.pool(CONV3)
        POOL3_flat = POOL3.view(-1, 256 * 6 * 6)
        FC1 = F.relu(self.fc1(POOL3_flat))
        FC2 = F.relu(self.fc2(FC1))
        DROP = self.dropout(FC2)
        FC3 = self.fc3(DROP)
        FEATURES = {'conv1': CONV1,
                    'conv2': CONV2,
                    'conv3': CONV3,
                    'pool3': POOL3,
                    'fc1': FC1,
                    'fc2': FC2,
                    'out': FC3}
        return FEATURES

class Net2_full_256(nn.Module):
    def __init__(self):
        super(Net2_full, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 11, stride = 2,padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3,padding=0)
        self.conv3 = nn.Conv2d(32, 64, 3,padding=0)
        self.fc1 = nn.Linear(64 * 14 * 14, 250)
        self.fc2 = nn.Linear(250, 250)
        self.fc3 = nn.Linear(250, 7)
        self.dropout = nn.Dropout2d(p=0.2)
	
    def forward(self, x):
        CONV1 = F.relu(self.conv1(x))
        POOL1 = self.pool(CONV1)
        CONV2 = F.relu(self.conv2(POOL1))
        POOL2 = self.pool(CONV2)
        CONV3 = F.relu(self.conv3(POOL2))
        POOL3 = self.pool(CONV3)
        POOL3_flat = POOL3.view(-1, 64 * 14 * 14)
        FC1 = F.relu(self.fc1(POOL3_flat))
        FC2 = F.relu(self.fc2(FC1))
        DROP = self.dropout(FC2)
        FC3 = self.fc3(DROP)
        FEATURES = {'conv1': CONV1,
                    'conv2': CONV2,
                    'conv3': CONV3,
                    'pool3': POOL3,
                    'fc1': FC1,
                    'fc2': FC2,
                    'out': FC3}
        return FEATURES

class Net3_full(nn.Module):
    def __init__(self):
        super(Net3_full, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 11, stride = 2,padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 128, 3,padding=0)
        self.conv3 = nn.Conv2d(128, 256, 3,padding=0)
        self.fc1 = nn.Linear(256 * 14 * 14, 1000)
        self.fc2 = nn.Linear(1000, 1000)
        self.fc3 = nn.Linear(1000, 7)
        self.dropout = nn.Dropout2d(p=0.4)
	
    def forward(self, x):
        CONV1 = F.relu(self.conv1(x))
        POOL1 = self.pool(CONV1)
        CONV2 = F.relu(self.conv2(POOL1))
        POOL2 = self.pool(CONV2)
        CONV3 = F.relu(self.conv3(POOL2))
        POOL3 = self.pool(CONV3)
        POOL3_flat = POOL3.view(-1, 256 * 14 * 14)
        FC1 = F.relu(self.fc1(POOL3_flat))
        FC2 = F.relu(self.fc2(FC1))
        DROP = self.dropout(FC2)
        FC3 = self.fc3(DROP)
        FEATURES = {'conv1': CONV1,
                    'conv2': CONV2,
                    'conv3': CONV3,
                    'pool3': POOL3,
                    'fc1': FC1,
                    'fc2': FC2,
                    'out': FC3}
        return FEATURES

class Net2_full2(nn.Module):
    def __init__(self):
        super(Net2_full2, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 11, stride = 2,padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3,padding=0)
        self.conv3 = nn.Conv2d(32, 64, 3,padding=0)
        self.fc1 = nn.Linear(64 * 12 * 12, 250)
        self.fc2 = nn.Linear(250, 250)
        self.fc3 = nn.Linear(250, 7)
        self.dropout = nn.Dropout2d(p=0.4)
	
    def forward(self, x):
        CONV1 = F.relu(self.conv1(x))
        POOL1 = self.pool(CONV1)
        CONV2 = F.relu(self.conv2(POOL1))
        POOL2 = self.pool(CONV2)
        CONV3 = F.relu(self.conv3(POOL2))
        POOL3 = self.pool(CONV3)
        POOL3_flat = POOL3.view(-1, 64 * 12 * 12)
        FC1 = F.relu(self.fc1(POOL3_flat))
        FC2 = F.relu(self.fc2(FC1))
        DROP = self.dropout(FC2)
        FC3 = self.fc3(DROP)
        FEATURES = {'conv1': CONV1,
                    'conv2': CONV2,
                    'conv3': CONV3,
                    'pool3': POOL3,
                    'fc1': FC1,
                    'fc2': FC2,
                    'out': FC3}
        return FEATURES


class Net2_4ter(nn.Module):
    def __init__(self):
        super(Net2_4ter, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5, stride = 1,padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3,padding=0)
        self.conv3 = nn.Conv2d(32, 64, 3,padding=0)
        self.fc1 = nn.Linear(64 * 14 * 14, 250)
        self.fc2 = nn.Linear(250, 250)
        self.fc3 = nn.Linear(250, 330)
        self.dropout = nn.Dropout2d(p=0.4)
	
    def forward(self, x):
        CONV1 = F.relu(self.conv1(x))
        POOL1 = self.pool(CONV1)
        CONV2 = F.relu(self.conv2(POOL1))
        POOL2 = self.pool(CONV2)
        CONV3 = F.relu(self.conv3(POOL2))
        POOL3 = self.pool(CONV3)
        POOL3_flat = POOL3.view(-1, 64 * 14 * 14)
        FC1 = F.relu(self.fc1(POOL3_flat))
        FC2 = F.relu(self.fc2(FC1))
        DROP = self.dropout(FC2)
        FC3 = self.fc3(DROP)
        FEATURES = {'conv1': CONV1,
                    'conv2': CONV2,
                    'conv3': CONV3,
                    'pool3': POOL3,
                    'fc1': FC1,
                    'fc2': FC2,
                    'out': FC3}
        return FEATURES
    
class Net2_all(nn.Module):
    def __init__(self):
        super(Net2_all, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5, stride = 1,padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3,padding=0)
        self.conv3 = nn.Conv2d(32, 64, 3,padding=0)
        self.fc1 = nn.Linear(64 * 14 * 14, 250)
        self.fc2 = nn.Linear(250, 250)
        self.fc3 = nn.Linear(250, 1600)
        self.dropout = nn.Dropout2d(p=0.4)
	
    def forward(self, x):
        CONV1 = F.relu(self.conv1(x))
        POOL1 = self.pool(CONV1)
        CONV2 = F.relu(self.conv2(POOL1))
        POOL2 = self.pool(CONV2)
        CONV3 = F.relu(self.conv3(POOL2))
        POOL3 = self.pool(CONV3)
        POOL3_flat = POOL3.view(-1, 64 * 14 * 14)
        FC1 = F.relu(self.fc1(POOL3_flat))
        FC2 = F.relu(self.fc2(FC1))
        DROP = self.dropout(FC2)
        FC3 = self.fc3(DROP)
        FEATURES = {'conv1': CONV1,
                    'conv2': CONV2,
                    'conv3': CONV3,
                    'pool3': POOL3,
                    'fc1': FC1,
                    'fc2': FC2,
                    'out': FC3}
        return FEATURES
        
        
class hyper_search_net(nn.Module, nb_conv, nb_fc, ss1=None, ss2=None, ss3 = None, nb_kerc1=None, nb_kerc2=None, nb_kerc3=None, nb_kerc4=None, nb_kerfc1=None, nb_kerfc2=None):
	    def __init__(self):
        super(hyper_search_net, self).__init__()
        if nb_conv > 0:	
        	self.conv1 = nn.Conv2d(3, 16, 5, stride = 1,padding=1)
        	
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3,padding=0)
        self.conv3 = nn.Conv2d(32, 64, 3,padding=0)
        self.fc1 = nn.Linear(64 * 14 * 14, 250)
        self.fc2 = nn.Linear(250, 250)
        self.fc3 = nn.Linear(250, 1600)
        self.dropout = nn.Dropout2d(p=0.4)
	
    def forward(self, x):
        CONV1 = F.relu(self.conv1(x))
        POOL1 = self.pool(CONV1)
        CONV2 = F.relu(self.conv2(POOL1))
        POOL2 = self.pool(CONV2)
        CONV3 = F.relu(self.conv3(POOL2))
        POOL3 = self.pool(CONV3)
        POOL3_flat = POOL3.view(-1, 64 * 14 * 14)
        FC1 = F.relu(self.fc1(POOL3_flat))
        FC2 = F.relu(self.fc2(FC1))
        DROP = self.dropout(FC2)
        FC3 = self.fc3(DROP)
        FEATURES = {'conv1': CONV1,
                    'conv2': CONV2,
                    'conv3': CONV3,
                    'pool3': POOL3,
                    'fc1': FC1,
                    'fc2': FC2,
                    'out': FC3}
        return FEATURES


