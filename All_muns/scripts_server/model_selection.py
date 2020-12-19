# -*- coding: utf-8 -*-
'''

Script to select the model with highest accuracy

'''

import numpy as np
import glob
import sys
import torch
import os

import sys
sys.path.append('../../')
sys.path.append('/home/alban/DATA/MODELS/')
sys.path.append('/home/arash/Software/repositories/kernelphysiology/python/src/')

from DL_testing_functions_scripts import DL_testing_functions as DLtest
from utils_scripts import algos
from DL_training_functions_scripts import MODELS as M
from kernelphysiology.dl.pytorch.models import model_utils 


#from models import utils
os.environ["CUDA_VISIBLE_DEVICES"]="2"
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

nb_epochs = 90

def info_parameters(state_dict):
    num_kernels = 0
    num_parameters = 0
    for layer in state_dict.keys():
        if '.weight' in layer:
            current_layer = state_dict[layer].cpu().numpy()
            #import pdb; pdb.set_trace()
            num_kernels += current_layer.shape[0]
            num_parameters += current_layer.size
    return num_kernels, num_parameters

'''
list_training_dir = glob.glob('*train_curve.npy')

for i in list_training_dir:
	if 'tristimulus' in i:
		list_training_dir.remove(i)
	if 'D65' in i:
		list_training_dir.remove(i)

for i in list_training_dir:
	if 'tristimulus' in i:
		list_training_dir.remove(i)
	if 'D65' in i:
		list_training_dir.remove(i)

TRAIN_CURVES = np.zeros((len(list_training_dir), nb_epochs))
DIFF_train_val = np.zeros((len(list_training_dir), nb_epochs))
NB_PARAM = np.zeros((len(list_training_dir), 2))

import re
models = [re.search('(.*?)train_curve.npy', addr).group(1) for addr in list_training_dir]

count = 0
for add in list_training_dir:
	tran_curv = np.load(add)
	TRAIN_CURVES[count] = tran_curv
	weights = 'INST/%s/epoch_%i.pt' %(models[count], np.argmax(TRAIN_CURVES[count])+1)
	state_dict = torch.load(weights)
	num_kernels, num_param = info_conv_parameters(state_dict)
	NB_PARAM[count, 0] = num_kernels
	NB_PARAM[count, 1] = num_param
	count +=1


INDX = np.unravel_index(np.argmax(TRAIN_CURVES), (len(list_training_dir), nb_epochs))

print('The model with the highest performance is: %s \nat epoch %i \nwith a performance of %d percents' %(list_training_dir[INDX[0]], INDX[1], TRAIN_CURVES[INDX]))

sort_IDX = np.argsort(-np.amax(TRAIN_CURVES, axis = 1))

np.amax(TRAIN_CURVES, axis = 1)[sort_IDX[:10]]
#list_training_dir[list(sort_IDX[:10])]
for i in sort_IDX[:10]:
	print(list_training_dir[i])

np.save('nb_param_convnet.npy',NB_PARAM)
np.save('train_curves_convnet.npy', TRAIN_CURVES)

'''


# In[9]: COMPUTE ACTIVATIONS DCC
import argparse

parser = argparse.ArgumentParser(description='Parsing variables for rendering images')

parser.add_argument('--gpu_id', default=0, type=int, metavar='N',
                    help='ID of Gpu to use')

parser.add_argument('--load_dir', default='', type=str, metavar='str',
                    help='dir where to load models, weights and training curves')

parser.add_argument('--save_dir', default='', type=str, metavar='str',
                    help='dir where to save activations, weights and training curves')

parser.add_argument('--model', default='Ref', type=str, metavar='str',
                    help='to distiguish between the different models')
                    
args = parser.parse_args()




## Xp set
####---------------------------------------------------------------------------------------------------------------------

DIR_LOAD = args.load_dir



#epochmax[[2,-3]] = 37

# In[9]:

if args.model == 'RefConv':
    path = '/home/alban/DATA/MODELS/state_dicts/inst1_CC_AlbanNet.pt'
elif args.model == 'Original':
    path = '/mnt/juggernaut/alban/project_color_constancy/PYTORCH/WCS/train_centered/All_muns/INST_CC/inst_1_CC/epoch_80.pt'
elif args.model == 'MobileNet':
    path = '/home/alban/DATA/MODELS/wcs_lms_1600/mobilenet_v2/sgd/scratch/original/checkpoint.pth.tar'
    layer_names = ['features.0.1.weight', 'features.2.conv.3.weight',  'features.4.conv.3.weight',   'features.6.conv.3.weight',  'features.8.conv.3.weight', 'features.10.conv.3.weight', 'features.12.conv.3.weight', 'features.14.conv.3.weight', 'features.16.conv.3.weight', 'features.18.1.weight']
elif args.model == 'AlbanNet':
   path = '/home/alban/DATA/MODELS/wcs_lms_1600/alban_net/sgd/scratch/original/checkpoint.pth.tar'
elif args.model == 'VGG11_bn':
    path = '/home/alban/DATA/MODELS/wcs_lms_1600/vgg11_bn/sgd/scratch/original/checkpoint.pth.tar'
    layer_names = ['features.0.weight',  'features.4.weight',  'features.8.weight',  'features.11.weight', 'features.15.weight', 'features.18.weight', 'features.22.weight',  'features.26.weight', 'classifier.0.weight', 'classifier.3.weight']
elif args.model == 'ResNet50':
    path = '/home/alban/DATA/MODELS/wcs_lms_1600/resnet_bottleneck_custom/sgd/scratch/original_b2354_k64/checkpoint.pth.tar'
    layer_names = ['layer1.1.conv3.weight', 'layer2.2.conv3.weight', 'layer3.4.conv3.weight', 'layer4.3.conv3.weight']
elif args.model == 'ResNet18':
    path = '/home/alban/DATA/MODELS/wcs_lms_1600/resnet_bottleneck_custom/sgd/scratch/original_b2222_k64/checkpoint.pth.tar'
elif args.model == 'ResNet11':
    path = '/home/alban/DATA/MODELS/wcs_lms_1600/resnet_bottleneck_custom/sgd/scratch/original_b1111_k64/checkpoint.pth.tar'
elif args.model == 'RefResNet':
    path = '/home/arash/Software/repositories/kernelphysiology/python/data/nets/pytorch/wcs/wcs_lms_1600/resnet_bottleneck_custom/sgd/scratch/Ref0_b3120_k16_b1024_e90/checkpoint.pth.tar'
    layer_names = ['layer1.2.conv3.weight','layer2.0.conv3.weight','layer3.1.conv3.weight']

net = torch.load(path)

num_ker, num_param = info_parameters(net)

np.save('PARAM_%s.npy'%args.model, {'nb_ker': num_ker, 'nb_param': num_param})

