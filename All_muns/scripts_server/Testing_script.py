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


# In[9]: COMPUTE ACTIVATIONS DCC
import argparse

parser = argparse.ArgumentParser(description='Parsing variables for rendering images')

parser.add_argument('--gpu_id', default=0, type=int, metavar='N',
                    help='ID of Gpu to use')

parser.add_argument('--load_dir', default='', type=str, metavar='str',
                    help='dir where to load models, weights and training curves')

parser.add_argument('--save_dir', default='', type=str, metavar='str',
                    help='dir where to save activations, weights and training curves')

parser.add_argument('--testing_dir', default='', type=str, metavar='str',
                    help='dir where to find models and training curves')

parser.add_argument('--training_set', default='CC', type=str, metavar='str',
                    help='to distiguish between CC and D65')

parser.add_argument('--testing_set', default='WCS', type=str, metavar='str',
                    help='to distiguish between WCS and all muns')

parser.add_argument('--testing_type', default='4illu', type=str, metavar='str',
                    help='to distiguish between the illuminant kind')

parser.add_argument('--testing_condition', default='normal', type=str, metavar='str',
                    help='to distiguish between the different conditions (nromal, no patch, wrong patch..)')

parser.add_argument('--model', default='Ref', type=str, metavar='str',
                    help='to distiguish between the different models')
                    
parser.add_argument('--focus', default='Munsells', type=str, metavar='str',
                    help='to distiguish between the different models')

parser.add_argument('--layer', default='', type=str, metavar='str',
                    help='to distiguish between the different layers to take activations from')

args = parser.parse_args()


if args.model == 'Original':
	nb_models = 10
elif args.model == 'RefResNet':
	nb_models = 6
else:
	nb_models = 1

if args.testing_set == 'WCS':
	nb_obj = 330
elif args.testing_set == 'all':
	nb_obj = 1600

if args.testing_type == '5illu':
	nb_illu = 5
	nb_ex = 5
	list_illus = ['A_norm', 'B_norm', 'C_norm', 'D_norm', 'D_65']
elif args.testing_type == '4illu':
	nb_illu = 4
	nb_ex = 10
elif args.testing_type == 'var':
    nb_illu = 28
    nb_ex = 1
elif args.testing_type == 'D65':
    nb_illu = 1
    nb_ex = 28
elif args.testing_type == 'D65_masks':
    nb_illu = 1
    nb_ex = 20


## Xp set
####---------------------------------------------------------------------------------------------------------------------

list_WCS_labels = algos.compute_WCS_Munsells_categories() # import idxes of WCS munsells among the 1600

DIR_LOAD = args.load_dir


#epochmax = np.zeros((nb_models))

Training_curv, epochmax = DLtest.training_curves(DIR_LOAD + 'INST_%s/'%(args.training_set),args.training_set, 90)


#epochmax[[2,-3]] = 37

# In[9]:

with open("/mnt/juggernaut/alban/project_color_constancy/PYTORCH/WCS/train_centered/WCS/ima_empty_scenes.txt", "rb") as fp:   # Unpickling
        val_im_empty_scenes = pickle.load(fp)

val_im_empty_scenes = ['/mnt/juggernaut' + i[5:] for i in val_im_empty_scenes]

if args.model == 'RefConv':
	net = M.Ref()
elif args.model == 'Original':
	net = M.Net2_4ter_norm()
elif args.model == 'MobileNet':
	path = '/home/alban/DATA/MODELS/wcs_lms_1600/mobilenet_v2/sgd/scratch/original/checkpoint.pth.tar'
	net, tgsize = model_utils.which_network_classification(path, 1600)
	layer_names = ['features.0.1.weight', 'features.2.conv.3.weight',  'features.4.conv.3.weight',   'features.6.conv.3.weight',  'features.8.conv.3.weight', 'features.10.conv.3.weight', 'features.12.conv.3.weight', 'features.14.conv.3.weight', 'features.16.conv.3.weight', 'features.18.1.weight']
elif args.model == 'AlbanNet':
	net, tgsize = model_utils.which_network_classification('/home/alban/DATA/MODELS/wcs_lms_1600/alban_net/sgd/scratch/original/checkpoint.pth.tar', 1600)
elif args.model == 'VGG11_bn':
	path = '/home/alban/DATA/MODELS/wcs_lms_1600/vgg11_bn/sgd/scratch/original/checkpoint.pth.tar'
	net, tgsize = model_utils.which_network_classification(path, 1600)
	layer_names = ['features.0.weight',  'features.4.weight',  'features.8.weight',  'features.11.weight', 'features.15.weight', 'features.18.weight', 'features.22.weight',  'features.26.weight', 'classifier.0.weight', 'classifier.3.weight']
elif args.model == 'ResNet50':
	path = '/home/alban/DATA/MODELS/wcs_lms_1600/resnet_bottleneck_custom/sgd/scratch/original_b2354_k64/checkpoint.pth.tar'
	net, tgsize = model_utils.which_network_classification(path, 1600)
	layer_names = ['layer1.1.conv3.weight', 'layer2.2.conv3.weight', 'layer3.4.conv3.weight', 'layer4.3.conv3.weight']
elif args.model == 'ResNet18':
	net, tgsize = model_utils.which_network_classification('/home/alban/DATA/MODELS/wcs_lms_1600/resnet_bottleneck_custom/sgd/scratch/original_b2222_k64/checkpoint.pth.tar', 1600)
elif args.model == 'ResNet11':
	net, tgsize = model_utils.which_network_classification('/home/alban/DATA/MODELS/wcs_lms_1600/resnet_bottleneck_custom/sgd/scratch/original_b1111_k64/checkpoint.pth.tar', 1600)
elif args.model == 'RefResNet':
	path = '/home/arash/Software/repositories/kernelphysiology/python/data/nets/pytorch/wcs/wcs_lms_1600/resnet_bottleneck_custom/sgd/scratch/Ref%i_b3120_k16_b1024_e90/checkpoint.pth.tar'
	layer_names = ['layer1.2.conv3.weight','layer2.0.conv3.weight','layer3.1.conv3.weight']

if args.testing_type == '4illu':
	test_addr = '/mnt/juggernaut/alban/project_color_constancy/TRAINING/DATA/PNG/WCS/Test_4_illu_centered/'
elif args.testing_type == 'D65':
	ILLU = np.load(npy_dir_path + 'ILLU.npy')
	test_addr = '/home/alban/project_color_constancy/TRAINING/DATA/PNG/WCS/validation_D65_centered/'
elif args.testing_type == '5illu':
	test_addr = '/home/alban/DATA/IM_CC/masks_5illu/'
elif args.testing_type == 'D65_masks':
	test_addr = '/home/alban/DATA/IM_CC/masks_D65/'

# name of layers for RefResNet


if args.model == 'Original':
	conv1 = np.zeros((nb_models,nb_obj,nb_illu,nb_ex, 16))
	conv2 = np.zeros((nb_models,nb_obj,nb_illu,nb_ex, 32))
	conv3 = np.zeros((nb_models,nb_obj,nb_illu,nb_ex, 64))
	fc1 = np.zeros((nb_models,nb_obj,nb_illu,nb_ex, 250))
	fc2 = np.zeros((nb_models,nb_obj,nb_illu,nb_ex, 250))

shape_out = tuple([1600])

if (args.model == 'RefResNet') & (len(args.layer) > 0):
	net, tgsize = model_utils.which_network_classification(path%0, 1600)
	net = model_utils.LayerActivation(net, layer_names[int(args.layer[-1]) -1])
	x, _ = DLtest.transform(val_im_empty_scenes[0], val_im_empty_scenes, preprocessing = 'arash')
	net.to(device)
	net.eval()
	output = net(x)
	shape_out = output[0].cpu().detach().numpy().shape
    
if (args.model =='ResNet50') & (len(args.layer) > 0):
	net = model_utils.LayerActivation(net, layer_names[int(args.layer[-1]) -1])
	x, _ = DLtest.transform(val_im_empty_scenes[0], val_im_empty_scenes, preprocessing = 'arash')
	net.to(device)
	net.eval()
	output = net(x)
	shape_out = output[0].cpu().detach().numpy().shape

if (args.model in ['MobileNet', 'VGG11_bn'] ) & (len(args.layer) > 0):
	#import pdb; pdb.set_trace()
	net = DLtest.IntermediateModel(net, layer_names[int(args.layer[-1]) -1])
	x, _ = DLtest.transform(val_im_empty_scenes[0], val_im_empty_scenes, preprocessing = 'arash')
	net.to(device)
	net.eval()
	output = net(x)
	shape_out = output[0].cpu().detach().numpy().shape

out = np.zeros((nb_models,nb_obj,nb_illu,nb_ex) + tuple([shape_out[0]]))
predictions = np.zeros((nb_models,nb_obj,nb_illu,nb_ex))
EVAL = np.zeros((nb_models,nb_obj,nb_illu))




list_WCS_labels = algos.compute_WCS_Munsells_categories() # import idxes of WCS munsells among the 1600
muns_alpha2num = sorted(range(1600), key = str)
muns_num2alpha = [muns_alpha2num.index(l) for l in range(1600)] # labels of muns in alpha order

WCS_num2alpha = [sorted(range(1600), key = str).index(l) for l in list_WCS_labels] # labels of WCS in alpha order
WCS_alpha2num = [muns_num2alpha[l] for l in list_WCS_labels] # indx of WCS in alpha order




for m in range(nb_models):
	print('Evaluation of model %i' %(m+1))
	if (args.model == 'RefConv') | (args.model == 'Original'):
		weights = DIR_LOAD +'INST_%s/inst_%d_%s/epoch_%i.pt' %((args.training_set,m,args.training_set,epochmax[m]))
		net.load_state_dict(torch.load(weights))
	elif args.model == 'RefResNet':
		net, tgsize = model_utils.which_network_classification(path%m, 1600)
		if len(args.layer) > 0:
                        net = model_utils.LayerActivation(net, layer_names[int(args.layer[-1]) -1])
	elif args.model == 'ResNet50':
		net, tgsize = model_utils.which_network_classification(path, 1600)
		if len(args.layer) > 0:
                        net = model_utils.LayerActivation(net, layer_names[int(args.layer[-1]) -1])
	elif args.model in ['VGG11_bn', 'MobileNet']:
		net, tgsize = model_utils.which_network_classification(path, 1600)
		if len(args.layer) > 0:
                        net = DLtest.IntermediateModel(net, layer_names[int(args.layer[-1]) -1])
	net.to(device)
	net.eval()
	for muns in range(nb_obj):
		for ill in range(nb_illu):
			for exp in range(nb_ex):
				if args.testing_type == '4illu':
					img = test_addr + 'object%i_%s_%i.npy' %(muns,chr(ill+65),exp)
				elif args.testing_type == '5illu':
					img = test_addr + 'object%i/object%i_illu_%s_%s.npy' %(muns, muns, list_illus[ill], exp)
				elif args.testing_type == 'D65_masks':
					img = test_addr + 'object%i/object%i_illu_D_65_%s.npy' %(muns,muns,exp)
				else:
					img = test_addr + 'object%i_illu_%s.npy' %(muns,ILLU[10*ill*exp])
				if args.model == 'Original':
					conv1[m,muns,ill,exp], conv2[m,muns,ill,exp], conv3[m,muns,ill,exp], fc1[m,muns,ill,exp], fc2[m,muns,ill,exp], out[m,muns,ill,exp],  predictions[m,muns,ill,exp] = DLtest.retrieve_activations(net, img, val_im_empty_scenes, type = 'npy', testing = args.testing_condition, focus = args.focus)
				else:
					out[m,muns,ill,exp],  predictions[m,muns,ill,exp] = DLtest.compute_outputs(net, img, val_im_empty_scenes, type = 'npy', testing = args.testing_condition, focus = args.focus, prep = 'arash', layer = args.layer)				        
	for munsell in range(nb_obj):
		if args.model == 'Original':
			EVAL[m,munsell] = DLtest.evaluation(predictions[m,munsell], list_WCS_labels[munsell])
		else:
			EVAL[m,munsell] = DLtest.evaluation(predictions[m,munsell], WCS_alpha2num[munsell])
	print('Result = %d' %np.mean(EVAL[m]))

complement_addr = '_%s_%s_%s_%s_%s.npy'%(args.model, args.training_set, args.testing_type, args.testing_set, args.testing_condition)

DIR_SAVE = args.save_dir

if args.model == 'Original':
	np.save(DIR_SAVE +'layers/%s/conv1'%args.focus + complement_addr, conv1)
	np.save(DIR_SAVE +'layers/%s/conv2'%args.focus+ complement_addr, conv2)
	np.save(DIR_SAVE +'layers/%s/conv3'%args.focus + complement_addr, conv3)
	np.save(DIR_SAVE +'layers/%s/fc1'%args.focus + complement_addr, fc1)
	np.save(DIR_SAVE +'layers/%s/fc2'%args.focus + complement_addr, fc2)
#np.save(DIR_SAVE +'layers/%s/predictions'%args.focus + complement_addr, predictions)
np.save(DIR_SAVE +'outs/out' + args.layer + complement_addr, out)
#np.save(DIR_SAVE +'layers/%s/evaluation'%args.focus + complement_addr, EVAL)



