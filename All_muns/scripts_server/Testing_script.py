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


from DL_testing_functions_scripts import DL_testing_functions as DLtest
from utils_scripts import algos
from DL_training_functions_scripts import MODELS as M


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

args = parser.parse_args()

nb_models = 10

if args.testing_set == 'WCS':
    nb_obj = 330
elif args.testing_set == 'all':
    nb_obj = 1600

if args.testing_type == '5illu':
    nb_illu = 5
    nb_ex = 5
list_illus = ['A_norm', 'B_norm', 'C_norm', 'D_norm', 'D_65']

if args.testing_type == '4illu':
    nb_illu = 4
    nb_ex = 10
elif args.testing_type == 'var':
    nb_illu = 28
    nb_ex = 1
elif args.testing_type == 'D65':
    nb_illu = 1
    nb_ex = 28


## Xp set
####---------------------------------------------------------------------------------------------------------------------

list_WCS_labels = algos.compute_WCS_Munsells_categories() # import idxes of WCS munsells among the 1600

DIR_LOAD = args.load_dir


#epochmax = np.zeros((nb_models))

Training_curv, epochmax = DLtest.training_curves(DIR_LOAD + 'INST_%s/'%(args.training_set),args.training_set, 90)


epochmax[[2,-3]] = 37

# In[9]:

with open("/mnt/juggernaut/alban/project_color_constancy/PYTORCH/WCS/train_centered/WCS/ima_empty_scenes.txt", "rb") as fp:   # Unpickling
        val_im_empty_scenes = pickle.load(fp)



conv1 = np.zeros((nb_models,nb_obj,nb_illu,nb_ex, 16))
conv2 = np.zeros((nb_models,nb_obj,nb_illu,nb_ex, 32))
conv3 = np.zeros((nb_models,nb_obj,nb_illu,nb_ex, 64))
fc1 = np.zeros((nb_models,nb_obj,nb_illu,nb_ex, 250))
fc2 = np.zeros((nb_models,nb_obj,nb_illu,nb_ex, 250))
out = np.zeros((nb_models,nb_obj,nb_illu,nb_ex, 1600))
predictions = np.zeros((nb_models,nb_obj,nb_illu,nb_ex))
EVAL = np.zeros((nb_models,nb_obj,nb_illu))

if args.model == 'Ref':
	net = M.Ref()
elif args.model == 'Original':
    net = M.Net2_4ter_norm()

if args.testing_type == '4illu':
	test_addr = '/home/alban/project_color_constancy/TRAINING/DATA/PNG/WCS/Test_4_illu_centered/'
elif args.testing_type == 'D65':
	ILLU = np.load(npy_dir_path + 'ILLU.npy')
	test_addr = '/home/alban/project_color_constancy/TRAINING/DATA/PNG/WCS/validation_D65_centered/'
elif args.testing_type == '5illu':
	test_addr = '/home/alban/DATA/IM_CC/masks_5illu/'



for m in range(nb_models):
	print('Evaluation of model %i' %(m+1))
	weights = DIR_LOAD +'INST_%s/inst_%d_%s/epoch_%i.pt' %((args.training_set,m,args.training_set,epochmax[m]))
	net.load_state_dict(torch.load(weights))
	net.to(device)
	net.eval()
	for muns in range(nb_obj):
		for ill in range(nb_illu):
			for exp in range(nb_ex):
				if args.testing_type == '4illu':
					img = test_addr + 'object%i_%s_%i.npy' %(muns,chr(ill+65),exp)
				elif args.testing_type == '5illu':
					img = test_addr + 'object%i/object%i_illu_%s_%s.npy' %(muns, muns, list_illus[ill], exp)
				else:
					img = test_addr + 'object%i_illu_%s.npy' %(muns,ILLU[10*ill*exp])
				conv1[m,muns,ill,exp], conv2[m,muns,ill,exp], conv3[m,muns,ill,exp], fc1[m,muns,ill,exp], fc2[m,muns,ill,exp], out[m,muns,ill,exp],  predictions[m,muns,ill,exp] = DLtest.retrieve_activations(net, img, val_im_empty_scenes, type = 'npy', testing = args.testing_condition)
	for munsell in range(nb_obj):
		EVAL[m,munsell] = DLtest.evaluation(predictions[m,munsell], list_WCS_labels[munsell])
	print('Result = %d' %np.mean(EVAL[m]))

complement_addr = '_%s_%s_%s_%s_%s.npy'%(args.model, args.training_set, args.testing_type, args.testing_set, args.testing_condition)

DIR_SAVE = args.save_dir

np.save(DIR_SAVE +'layers/conv1'+ complement_addr, conv1)
np.save(DIR_SAVE +'layers/conv2'+ complement_addr, conv2)
np.save(DIR_SAVE +'layers/conv3'+ complement_addr, conv3)
np.save(DIR_SAVE +'layers/fc1'+ complement_addr, fc1)
np.save(DIR_SAVE +'layers/fc2'+ complement_addr, fc2)
np.save(DIR_SAVE +'layers/predictions'+ complement_addr, predictions)
np.save(DIR_SAVE +'layers/out'+ complement_addr, out)
np.save(DIR_SAVE +'layers/evaluation'+ complement_addr, EVAL)



