import sys
import os

import torch
import torch.nn as nn
import torchvision.models as pmodels
import torchvision.models.segmentation as seg_models

sys.path.append('/home/alban/DATA/MODELS/')
sys.path.append('/home/arash/Software/repositories/kernelphysiology/python/src/')
sys.path.append('../../')


from DL_testing_functions_scripts import DL_testing_functions as DLtest
from utils_scripts import algos
from DL_training_functions_scripts import MODELS as M

from kernelphysiology.dl.pytorch.models import model_utils 


# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

net, tgsize = model_utils.which_network_classification('/home/alban/DATA/MODELS/wcs_lms_1600/resnet50/sgd/scratch/original_b256_e90/model_best.pth.tar', 1600)

#net = model_utils.LayerActivation(net, 'layer1.2.conv3.weights')

net.to(device)
net.eval()


test_addr = '/home/alban/DATA/IM_CC/masks_5illu/'

img = test_addr + 'object218/object218_illu_D_norm_4.npy' 
x, mask = DLtest.transform(img, 'normal', 'npy')
outputs = net(x)
