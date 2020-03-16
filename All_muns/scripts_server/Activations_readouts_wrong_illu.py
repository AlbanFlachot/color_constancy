#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 14:24:13 2019

@author: alban
"""



from __future__ import print_function, division
import os
import torch
import matplotlib.pyplot as plt
from skimage import io, transform
import numpy as np
from torch.utils import data
from torchvision import transforms, utils
import pickle
import MODELS as M
import matplotlib.patheffects as PathEffects
import scipy.io as sio
from random import randint
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.append('../')

from color_scripts import color_transforms as CT # scripts with a bunch of functions for color transformations
from error_measures_scripts import Error_measures as EM
from display_scripts import display as dis
import torch.nn as nn
import torch.nn.functional as F

# In[2]:

import matplotlib.patheffects as PathEffects

# CUDA for PyTorch
os.environ["CUDA_VISIBLE_DEVICES"]="1"
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
#cudnn.benchmark = True


# In[3]: FUNCTIONS

def training_curves(training_dir, nb_mod, nb_epoch, Readout = False, layer = ''):
	Training_curv = np.zeros((nb_mod,nb_epoch))
	epochmax = np.zeros((nb_mod))
	for i in range(nb_mod):
		if Readout:
			Training_curv[i] = np.load( training_dir +'inst_%d/readout_'%((i+1)) + layer + '/train_curve.npy' )
		else:
			Training_curv[i] = np.load( training_dir +'inst%dtrain_curve.npy' %((i)))
		epochmax[i] = np.argmax(Training_curv[i])
	return Training_curv, epochmax

# In[9]: COMPUTE ACTIVATIONS DCC

test_mode = '_no_back'

WCS_muns = list()
with open("/home/alban/project_color_constancy/PYTORCH/WCS/WCS_muns.txt") as f:
    for line in f:
       WCS_muns.append(line.split()[0])

All_muns = list()
with open("/home/alban/project_color_constancy/PYTORCH/WCS/munsell_labels.txt") as f:
    for line in f:
       All_muns.append(line.split()[0])

list_WCS_labels = np.asarray([All_muns.index(WCS_muns[i]) for i in range(len(WCS_muns))])


TRAINING_CURV, EPOCHMAX = training_curves('./', 10, 90)
training_curv_c1,epochmax_c1 = training_curves('finetuning/', 10, 30, Readout = True, layer = 'conv1')
training_curv_c2,epochmax_c2 = training_curves('finetuning/', 10, 30, Readout = True, layer = 'conv2')
training_curv_c3,epochmax_c3 = training_curves('finetuning/', 10, 30, Readout = True, layer = 'conv3')
training_curv_f1,epochmax_f1 = training_curves('finetuning/', 10, 30, Readout = True, layer = 'fc1')
training_curv_f2,epochmax_f2 = training_curves('finetuning/', 10, 30, Readout = True, layer = 'fc2')

with open("/home/alban/project_color_constancy/PYTORCH/WCS/train_centered/WCS/ima_empty_scenes.txt", "rb") as fp:   # Unpickling
        val_im_empty_scenes = pickle.load(fp)

nb_models = 10
pc1 =  np.zeros((nb_models,330,4,10))
out_c1 = np.zeros((nb_models,330,4,10,1600))
pc2 =  np.zeros((nb_models,330,4,10))
out_c2 = np.zeros((nb_models,330,4,10,1600))
pc3 =  np.zeros((nb_models,330,4,10))
out_c3 = np.zeros((nb_models,330,4,10,1600))
pf1 =  np.zeros((nb_models,330,4,10))
out_f1 = np.zeros((nb_models,330,4,10,1600))
pf2 =  np.zeros((nb_models,330,4,10))
out_f2 = np.zeros((nb_models,330,4,10,1600))
EVAL = np.zeros((nb_models,330,4))

net = M.Net2_4ter_norm()
net.to(device)

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

dir_addr = '/home/alban/project_color_constancy/TRAINING/DATA/PNG/WCS/masks_centered/4illu/'
for m in range(nb_models):   
	print('Evaluation of model %i' %(m+1)) 
	weights = '/home/alban/project_color_constancy/PYTORCH/WCS/train_centered/All_muns/INST/inst%d/epoch_%i.pt' %((m,EPOCHMAX[m]))
	net.load_state_dict(torch.load(weights))
	net.eval()
	weight_c1 = 'finetuning/inst_%i/readout_conv1/epoch_%i.pt'%(m+1,epochmax_c1[m])
	weight_c2 = 'finetuning/inst_%i/readout_conv2/epoch_%i.pt'%(m+1,epochmax_c2[m])
	weight_c3 = 'finetuning/inst_%i/readout_conv3/epoch_%i.pt'%(m+1,epochmax_c3[m])
	weight_f1 = 'finetuning/inst_%i/readout_fc1/epoch_%i.pt'%(m+1,epochmax_f1[m])
	weight_f2 = 'finetuning/inst_%i/readout_fc2/epoch_%i.pt'%(m+1,epochmax_f2[m])
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
	for muns in range(330):
		for ill in range(4):
			for exp in range(10):
				img = dir_addr + 'object%i_%s_%i_mask.npy' %(muns,chr(ill+65),exp)    
				pc1[m, muns, ill, exp], pc2[m,muns, ill, exp], pc3[m,muns, ill, exp], pf1[m,muns, ill, exp], pf2[m,muns, ill, exp], out_c1[m,muns, ill, exp], out_c2[m,muns, ill, exp], out_c3[m,muns, ill, exp], out_f1[m,muns, ill, exp], out_f2[m,muns, ill, exp] = EM.evaluation_Readouts_back(net, Readout_nets, img, 'npy', val_im_empty_scenes, test_mode)
	for munsell in range(330):
		EVAL[m,munsell] = EM.evaluation(pf2[m,munsell],list_WCS_labels[munsell])
	print('Result = %d' %np.mean(EVAL[m]))

np.save('pc1'+ test_mode+'.npy',pc1)
np.save('pc2'+ test_mode+'.npy',pc2)
np.save('pc3'+ test_mode+'.npy',pc3)
np.save('pf1'+ test_mode+'.npy',pf1)
np.save('pf2'+ test_mode+'.npy',pf2)

np.save('out_c1'+ test_mode+'.npy',out_c1)
np.save('out_c2'+ test_mode+'.npy',out_c2)
np.save('out_c3'+ test_mode+'.npy',out_c3)
np.save('out_f1'+ test_mode+'.npy',out_f1)
np.save('out_f2'+ test_mode+'.npy',out_f2)

