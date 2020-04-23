#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 14:24:13 2019

@author: alban
"""



from __future__ import print_function, division
import torch

import numpy as np

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


if args.testing_set == 'WCS':
    nb_obj = 330
elif args.testing_set == 'all':
    nb_obj = 1600

if args.testing_type == '4illu':
    nb_illu = 4
    nb_ex = 10
elif args.testing_type == 'var':
    nb_illu = 28
    nb_ex = 1
elif args.testing_type == 'D65':
    nb_illu = 1
    nb_ex = 28


list_WCS_labels = algos.compute_WCS_Munsells_categories() # import idxes of WCS munsells among the 1600

DIR_LOAD = args.load_dir

TRAINING_CURV, EPOCHMAX = DLtest.training_curves(DIR_LOAD + 'INST_%s/'%(args.training_set),args.training_set, 90)
training_curv_c1,epochmax_c1 = DLtest.training_curves(DIR_LOAD +'finetuning/', args.training_set, 30, Readout = True, layer = 'conv1')
training_curv_c2,epochmax_c2 = DLtest.training_curves(DIR_LOAD +'finetuning/', args.training_set, 30, Readout = True, layer = 'conv2')
training_curv_c3,epochmax_c3 = DLtest.training_curves(DIR_LOAD +'finetuning/', args.training_set, 30, Readout = True, layer = 'conv3')
training_curv_f1,epochmax_f1 = DLtest.training_curves(DIR_LOAD +'finetuning/', args.training_set, 30, Readout = True, layer = 'fc1')
training_curv_f2,epochmax_f2 = DLtest.training_curves(DIR_LOAD +'finetuning/', args.training_set, 30, Readout = True, layer = 'fc2')

epochmax_c1[epochmax_c1<26] = 26 # we saved only the last 5 pt files to save space. Only for convolutional layers
epochmax_c2[epochmax_c2<26] = 26
epochmax_c3[epochmax_c3<26] = 26
#epochmax_c1[epochmax_f1<26] = 26
#epochmax_c1[epochmax_f2<26] = 26

nb_models = len(TRAINING_CURV)
pc1 =  np.zeros((nb_models,nb_obj,nb_illu,nb_ex))
out_c1 = np.zeros((nb_models,nb_obj,nb_illu,nb_ex,1600))
pc2 =  pc1.copy()
out_c2 = np.zeros((nb_models,nb_obj,nb_illu, nb_ex,1600))
pc3 =  pc1.copy()
out_c3 = np.zeros((nb_models,nb_obj,nb_illu, nb_ex,1600))
pfc1 =  pc1.copy()
out_fc1 = np.zeros((nb_models,nb_obj,nb_illu,nb_ex,1600))
pfc2 =  pc1.copy()
out_fc2 = np.zeros((nb_models,nb_obj,nb_illu, nb_ex,1600))
EVAL = np.zeros((nb_models,nb_obj))

if args.model == 'Ref':
	net = M.Ref()
elif args.model == 'Original':
    net = M.Net2_4ter_norm()

Rc1 = M.Readout_net_c1()
Rc1.to(device)
Rc2 = M.Readout_net_c2()
Rc2.to(device)
Rc3 = M.Readout_net_c3()
Rc3.to(device)
Rf1 = M.Readout_net_fc()
Rf1.to(device)
Rf2 = M.Readout_net_fc()
Rf2.to(device)
Readout_nets = (Rc1, Rc2, Rc3, Rf1, Rf2)

if args.testing_type == '4illu':
	test_addr = '/home/alban/project_color_constancy/TRAINING/DATA/PNG/WCS/Test_4_illu_centered/'
elif args.testing_type == 'D65':
	ILLU = np.load(npy_dir_path + 'ILLU.npy')
	test_addr = '/home/alban/project_color_constancy/TRAINING/DATA/PNG/WCS/validation_D65_centered/'

for m in range(nb_models):
	print('Evaluation of model %i' %(m+1))
	weights = DIR_LOAD +'INST_%s/inst_%d_%s/epoch_%i.pt' %((args.training_set,m,args.training_set,EPOCHMAX[m]))
	net.load_state_dict(torch.load(weights))
	net.to(device)
	net.eval()
	weight_c1 = DIR_LOAD +'finetuning/inst_%i_%s/readout_conv1/epoch_%i.pt'%(m+1,args.training_set,epochmax_c1[m])
	weight_c2 = DIR_LOAD +'finetuning/inst_%i_%s/readout_conv2/epoch_%i.pt'%(m+1,args.training_set,epochmax_c2[m])
	weight_c3 = DIR_LOAD +'finetuning/inst_%i_%s/readout_conv3/epoch_%i.pt'%(m+1,args.training_set,epochmax_c3[m])
	weight_f1 = DIR_LOAD +'finetuning/inst_%i_%s/readout_fc1/epoch_%i.pt'%(m+1,args.training_set,epochmax_f1[m])
	weight_f2 = DIR_LOAD +'finetuning/inst_%i_%s/readout_fc2/epoch_%i.pt'%(m+1,args.training_set,epochmax_f2[m])
	Readout_nets[0].load_state_dict(torch.load(weight_c1))
	Readout_nets[1].load_state_dict(torch.load(weight_c2))
	Readout_nets[2].load_state_dict(torch.load(weight_c3))
	Readout_nets[3].load_state_dict(torch.load(weight_f1))
	Readout_nets[4].load_state_dict(torch.load(weight_f2))
	Readout_nets[0].eval()
	Readout_nets[1].eval()
	Readout_nets[2].eval()
	Readout_nets[3].eval()
	Readout_nets[4].eval()
	for muns in range(nb_obj):
		for ill in range(nb_illu):
			for exp in range(nb_ex):
				if args.testing_type == '4illu':
					img = test_addr + 'object%i_%s_%i.npy' %(muns,chr(ill+65),exp)
				else:
					img = test_addr + 'object%i_illu_%s.npy' %(muns,ILLU[10*ill*exp])
				out_c1[m,muns, ill, exp], pc1[m, muns, ill, exp] = DLtest.evaluation_Readouts(net, img, Readout_nets[0], 'conv1', args.testing_condition, 'npy')
				out_c2[m,muns, ill, exp], pc2[m, muns, ill, exp] = DLtest.evaluation_Readouts(net, img, Readout_nets[1], 'conv2', args.testing_condition, 'npy')
				out_c3[m,muns, ill, exp], pc3[m, muns, ill, exp] = DLtest.evaluation_Readouts(net, img, Readout_nets[2], 'conv3',args.testing_condition, 'npy')
				out_fc1[m,muns, ill, exp], pfc1[m, muns, ill, exp] = DLtest.evaluation_Readouts(net, img, Readout_nets[3], 'fc1',args.testing_condition,  'npy')
				out_fc2[m,muns, ill, exp], pfc2[m, muns, ill, exp] = DLtest.evaluation_Readouts(net, img, Readout_nets[4], 'fc2',args.testing_condition, 'npy')
	for munsell in range(nb_obj):
		EVAL[m,munsell] = DLtest.evaluation(pfc2[m,munsell],list_WCS_labels[munsell])
	print('Result = %d' %np.mean(EVAL[m]))


np.save(DIR_LOAD +'outs/out_c1_%s_%s_%s_%s_%s.npy'%(args.model, args.training_set, args.testing_type, args.testing_set, args.testing_condition),out_c1)
np.save(DIR_LOAD +'outs/out_c2_%s_%s_%s_%s_%s.npy'%(args.model, args.training_set, args.testing_type, args.testing_set, args.testing_condition),out_c2)
np.save(DIR_LOAD +'outs/out_c3_%s_%s_%s_%s_%s.npy'%(args.model, args.training_set, args.testing_type, args.testing_set, args.testing_condition),out_c3)
np.save(DIR_LOAD +'outs/out_fc1_%s_%s_%s_%s_%s.npy'%(args.model, args.training_set, args.testing_type, args.testing_set, args.testing_condition),out_fc1)
np.save(DIR_LOAD +'outs/out_fc2_%s_%s_%s_%s_%s.npy'%(args.model, args.training_set, args.testing_type, args.testing_set, args.testing_condition),out_fc2)

