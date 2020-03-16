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



class Dataset(data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, list_IDs, labels):
        'Initialization'
        self.labels = labels
        self.list_IDs = list_IDs

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]
        #Load data and get label
        if ID[-3:] == 'png':
            I = np.array(io.imread(ID)).astype(float)
            if np.amax(I) > 2:
            	I = (I/255)
            	#print((np.amax(I),np.amin(I)))
            else:
            	I = (I)
            #I = transform.resize(I,(224,224))
            I = np.moveaxis(I,-1,0)
            X = torch.from_numpy(I)
        elif ID[-3:] == 'npy':
        	I = np.load(ID).astype(float)
        	I = np.moveaxis(I,-1,0)
        	I = I*(0.75 + 0.5*np.random.random(1))
        	I = I - 3
        	X = torch.from_numpy(I)
        else:
            X = torch.load(ID)
        X = X.type(torch.FloatTensor)
        #X = torch.load(ID)
        #X = transforms.CenterCrop(224)
        y = self.labels[ID]
        return X, y
