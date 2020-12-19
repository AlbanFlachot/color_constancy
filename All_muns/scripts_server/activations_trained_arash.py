#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 16:18:07 2019

@author: alban
"""
from __future__ import print_function, division
import torch

import numpy as np
import pickle
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


import sys
sys.path.append('../../')
sys.path.append('/home/alban/DATA/MODELS/')
sys.path.append('/home/arash/Software/repositories/kernelphysiology/python/src/')

from DL_testing_functions_scripts import DL_testing_functions as DLtest
from utils_scripts import algos
from DL_training_functions_scripts import MODELS as M
from kernelphysiology.dl.pytorch.models import model_utils 


# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
#cudnn.benchmark = True


# In[9]: path to files

txt_dir_path = '../../txt_files/'
npy_dir_path = '../../npy_files/'
pickles_dir_path = '../pickles/'
figures_dir_path = '../figures/'

path = '/home/alban/DATA/MODELS/wcs_lms_1600/vgg11_bn/sgd/scratch/original/checkpoint.pth.tar'

model = torch.load(path)
model['state_dict'].keys()
net, tgsize = model_utils.which_network_classification(path%0, 1600)

layer_names_ResCC = ['layer1.2.conv3.weights','layer2.0.conv3.weights','layer3.1.conv3.weights']

path = '/home/alban/DATA/MODELS/wcs_lms_1600/vgg11_bn/sgd/scratch/original/checkpoint.pth.tar'
model = torch.load(path)
model['state_dict'].keys()
layer_names_vgg11 = [n  for n in model['state_dict'].keys() if 'weight' in n]


path = '/home/alban/DATA/MODELS/wcs_lms_1600/resnet_bottleneck_custom/sgd/scratch/original_b2354_k64/checkpoint.pth.tar'
model = torch.load(path)
model['state_dict'].keys()
layer_names_ResNet50 = ['layer1.1.conv3.weight', 'layer2.2.conv3.weight', 'layer3.4.conv3.weight', 'layer4.3.conv3.weight']

path = '/home/alban/DATA/MODELS/wcs_lms_1600/mobilenet_v2/sgd/scratch/original/checkpoint.pth.tar'
model = torch.load(path)
model['state_dict'].keys()
layer_names_MobileNet = ['features.1.conv.2.weight', 'features.2.conv.3.weight',  'features.3.conv.3.weight',  'features.4.conv.3.weight',  'features.5.conv.3.weight',  'features.6.conv.3.weight',  'features.7.conv.3.weight',  'features.8.conv.3.weight', 'features.9.conv.3.weight', 'features.10.conv.3.weight', 'features.11.conv.3.weight', 'features.12.conv.3.weight', 'features.13.conv.3.weight', 'features.14.conv.3.weight', 'features.15.conv.3.weight', 'features.16.conv.3.weight', 'features.17.conv.3.weight', 'features.18.1.weight']


#net2 = model_utils.LayerActivation(net, 'layer1.2.conv3.weights')
