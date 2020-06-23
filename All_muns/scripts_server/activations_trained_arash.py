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

path = '/home/alban/DATA/MODELS/ResNets_customs/reference_ResNet_rendLum_last/Ref%i_randLum_b3120_k16_b1024_e90/checkpoint.pth.tar'

model = torch.load(path%0)
net, tgsize = model_utils.which_network_classification(path%0, 1600)

layer_names = ['layer1.2.conv3.weights','layer2.0.conv3.weights','layer3.1.conv3.weights']


net2 = model_utils.LayerActivation(net, 'layer1.2.conv3.weights')
