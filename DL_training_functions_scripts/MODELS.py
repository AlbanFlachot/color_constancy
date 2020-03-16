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

class Net2_full(nn.Module):
    def __init__(self):
        super(Net2_full, self).__init__()
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

class Net3_full(nn.Module):
    def __init__(self):
        super(Net3_full, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 11, stride = 2,padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 128, 3,padding=0)
        self.conv3 = nn.Conv2d(128, 256, 3,padding=0)
        self.fc1 = nn.Linear(256 * 12 * 12, 1000)
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
        POOL3_flat = POOL3.view(-1, 256 * 12 * 12)
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

class Net2_4ter_norm(nn.Module):
    def __init__(self):
        super(Net2_4ter_norm, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5, stride = 1,padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3,padding=0)
        self.conv3 = nn.Conv2d(32, 64, 3,padding=0)
        self.conv3_bn = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64 * 14 * 14, 250)
        self.fc2 = nn.Linear(250, 250)
        self.fc3 = nn.Linear(250, 1600)
        self.dropout = nn.Dropout2d(p=0.4)
	
    def forward(self, x):
        CONV1 = F.relu(self.conv1(x))
        POOL1 = self.pool(CONV1)
        CONV2 = F.relu(self.conv2(POOL1))
        POOL2 = self.pool(CONV2)
        CONV3 = F.relu(self.conv3_bn(self.conv3(POOL2)))
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

class Net_tristimulus(nn.Module):
    def __init__(self):
        super(Net_tristimulus, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5, stride = 1,padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3,padding=0)
        self.conv3 = nn.Conv2d(32, 64, 3,padding=0)
        self.fc1 = nn.Linear(64 * 14 * 14, 250)
        self.fc2 = nn.Linear(250, 250)
        self.fc3 = nn.Linear(250, 3)
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

class Readout_net_c1(nn.Module):
    def __init__(self):
        super(Readout_net_c1, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16*63*63, 1600)
	
    def forward(self, x):
        POOL = self.pool(x)
        POOL_flat = POOL.view(-1, 16*63*63)
        FC1 = self.fc1(POOL_flat)
        FEATURES = {'out': FC1}
        return FEATURES


class Readout_net_c2(nn.Module):
    def __init__(self):
        super(Readout_net_c2, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32*30*30, 1600)
	
    def forward(self, x):
        POOL = self.pool(x)
        POOL_flat = POOL.view(-1, 32*30*30)
        FC1 = self.fc1(POOL_flat)
        FEATURES = {'out': FC1}
        return FEATURES

class Readout_net_c3(nn.Module):
    def __init__(self):
        super(Readout_net_c3, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64*14*14, 1600)
	
    def forward(self, x):
        POOL = self.pool(x)
        POOL_flat = POOL.view(-1, 64*14*14)
        FC1 = self.fc1(POOL_flat)
        FEATURES = {'out': FC1}
        return FEATURES
        
class Readout_net_fc(nn.Module):
    def __init__(self):
        super(Readout_net_fc, self).__init__()
        self.fc1 = nn.Linear(250, 1600)
	
    def forward(self, x):
        X_flat = x.view(-1, 250)
        FC1 = self.fc1(X_flat)
        FEATURES = {'out': FC1}
        return FEATURES
        
class Net_tristimulus_norm(nn.Module):
    def __init__(self):
        super(Net_tristimulus_norm, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5, stride = 1,padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3,padding=0)
        self.conv3 = nn.Conv2d(32, 64, 3,padding=0)
        self.conv3_bn = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64 * 14 * 14, 250)
        self.fc2 = nn.Linear(250, 250)
        self.fc3 = nn.Linear(250, 3)
        self.dropout = nn.Dropout2d(p=0.4)
	
    def forward(self, x):
        CONV1 = F.relu(self.conv1(x))
        POOL1 = self.pool(CONV1)
        CONV2 = F.relu(self.conv2(POOL1))
        POOL2 = self.pool(CONV2)
        CONV3 = F.relu(self.conv3_bn(self.conv3(POOL2)))
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

class Ref(nn.Module):
    def __init__(self):
        super(Ref, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5, stride = 1,padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3,padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3,padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3,padding=1)
        self.conv3_bn = nn.BatchNorm2d(64)
        self.conv4_bn = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64 * 16 * 16, 250)
        self.fc2 = nn.Linear(250, 250)
        self.fc3 = nn.Linear(250, 1600)
        self.dropout = nn.Dropout2d(p=0.4)
	
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3_bn(self.conv3(x)))
        x = self.pool(x)
        x = F.relu(self.conv4_bn(self.conv4(x)))
        x = self.pool(x)
        x = POOL3.view(-1, 64 * 16 * 16)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        FEATURES = {'out': x}
        return FEATURES

class hyper_search_net(nn.Module):
	def __init__(self):
		super(hyper_search_net, self).__init__()
		self.conv1 = nn.Conv2d(3, nb_kerc1, ks1, stride = ss1, padding=ks1//2)
		self.pool = nn.MaxPool2d(2, 2)
		self.conv2 = nn.Conv2d(nb_kerc1, nb_kerc2, ks2, stride = ss2, padding=ks2//2)
		self.conv3 = nn.Conv2d(nb_kerc2, nb_kerc3, ks3, stride = ss3, padding=ks3//2)
		self.conv4 = nn.Conv2d(nb_kerc3, nb_kerc4, ks4, stride = 1, padding=ks4//2)
		self.fc1 = nn.Linear(nb_kerc4*input_fc_size*input_fc_size, nb_kerfc1)
		self.fc2 = nn.Linear(nb_kerfc1, nb_kerfc2)
		self.fc3 = nn.Linear(nb_kerfc2, nb_munsells)
		self.dropout = nn.Dropout2d(p=0.4)
		self.conv3_bn = nn.BatchNorm2d(nb_kerc3)
		self.conv4_bn = nn.BatchNorm2d(nb_kerc4)

	def forward(self, x):
		x = F.relu(self.conv1(x))
		if nb_conv > 1:
			x = self.pool(x)
			x = F.relu(self.conv2(x))
			if nb_conv >2:
				x = self.pool(x)
				x = F.relu(self.conv3_bn(self.conv3(x)))
				if nb_conv == 4:
					x = self.pool(x)
					x = F.relu(self.conv4_bn(self.conv4(x)))
		if nb_conv <4:
			x = self.pool(x)
		x = x.view(-1, nb_kerc4*input_fc_size*input_fc_size)
		x = F.relu(self.fc1(x))
		if nb_fc == 2:
			x = F.relu(self.fc2(x))
		x = self.dropout(x)
		x = self.fc3(x)
		FEATURES = {'out': x}
		return FEATURES
