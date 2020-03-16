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

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


import torch.nn as nn
import torch.nn.functional as F


# In[2]:

import matplotlib.patheffects as PathEffects

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
#cudnn.benchmark = True


# In[3]: FUNCTIONS


def princomp(A):
 """ performs principal components analysis
     (PCA) on the n-by-p data matrix A
     Rows of A correspond to observations, columns to variables.

 Returns :
  coeff :
    is a p-by-p matrix, each column containing coefficients
    for one principal component.
  score :
    the principal component scores; that is, the representation
    of A in the principal component space. Rows of SCORE
    correspond to observations, columns to components.
  latent :
    a vector containing the eigenvalues
    of the covariance matrix of A.
 """
 # computing eigenvalues and eigenvectors of covariance matrix
 M = (A-np.mean(A.T,axis=1)).T # subtract the mean (along columns)
 [latent,coeff] = np.linalg.eig(np.cov(M)) # attention:not always sorted
 sortedIdx = np.argsort(-latent)
 latent = latent[sortedIdx]
 explained = 100*latent/np.sum(latent)
 score = np.dot(coeff.T,M) # projection of the data in the new space
 coeff = coeff[:,sortedIdx]
 score = score[sortedIdx,:]
 return coeff,score,latent, explained

def LMS2Opp(x):
	M = np.array([[ 0.67,  1.0,  0.5],[ 0.67,  -1,  0.5],[ 0.67,  0,  -1]])
	return np.dot(x,M)

def cart2sph(x,y,z):
    XsqPlusYsq = x**2 + y**2
    r = np.sqrt(XsqPlusYsq + z**2)               # r
    elev = np.arctan(z/np.sqrt(XsqPlusYsq))     # theta
    az = np.arctan2(y,x)                           # phi
    return np.array([r, az, elev])


def retrieve_activations(net,img,type):
    if type == 'png':
        I = np.array(io.imread(img)).astype(float)
        if np.amax(I) > 2:
            I = (I/255)
        else:
            I = I
        #I = transform.resize(I,(224,128))
        I = np.moveaxis(I,-1,0)
        x = torch.from_numpy(I)
        x = x.type(torch.FloatTensor)
    elif type == 'npy':
        I = np.load(img).astype(float)
        I = np.moveaxis(I,-1,0)
        I = I - 3
        x = torch.from_numpy(I)
        x = x.type(torch.FloatTensor)
    else:
        x = torch.load(img)
    x  = x.unsqueeze(0)
    x = x.to(device)
    outputs = net(x)
    conv1 = np.amax(outputs['conv1'].cpu().detach().numpy(),axis = (2,3))
    conv2 = np.amax(outputs['conv2'].cpu().detach().numpy(),axis = (2,3))
    conv3 = np.amax(outputs['conv3'].cpu().detach().numpy(),axis = (2,3))
    fc1 = outputs['fc1'].cpu().detach().numpy()
    fc2 = outputs['fc2'].cpu().detach().numpy()
    _, p = torch.max(outputs['out'].data, 1)
    return conv1[0],conv2[0],conv3[0],fc1[0],fc2[0], (p.cpu().detach().numpy())[0]

def evaluation_Readouts(net,READOUT_NETS,img,type):
	if type == 'png':
		I = np.array(io.imread(img)).astype(float)
		if np.amax(I) > 2:
			I = (I/255)
		else:
			I = I
		#I = transform.resize(I,(224,128))
		I = np.moveaxis(I,-1,0)
		x = torch.from_numpy(I)
		x = x.type(torch.FloatTensor)
	elif type == 'npy':
		I = np.load(img).astype(float)
		I = np.moveaxis(I,-1,0)
		I = I - 3
		x = torch.from_numpy(I)
		x = x.type(torch.FloatTensor)
	else:
		x = torch.load(img)
	x  = x.unsqueeze(0)
	x = x.to(device)
	outputs = net(x)
	RC1 = READOUT_NETS[0](outputs['conv1'])
	RC2 = READOUT_NETS[1](outputs['conv2'])
	RC3 = READOUT_NETS[2](outputs['conv3'])
	RF1 = READOUT_NETS[3](outputs['fc1'])
	RF2 = READOUT_NETS[4](outputs['fc2'])
	_, p1 = torch.max(RC1['out'].data, 1)
	_, p2 = torch.max(RC2['out'].data, 1)
	_, p3 = torch.max(RC3['out'].data, 1)
	_, p4 = torch.max(RF1['out'].data, 1)
	_, p5 = torch.max(RF2['out'].data, 1)
	return (p1.cpu().detach().numpy())[0], (p2.cpu().detach().numpy())[0], (p3.cpu().detach().numpy())[0], (p4.cpu().detach().numpy())[0], (p5.cpu().detach().numpy())[0], (RC1['out'].cpu().detach().numpy())[0], (RC2['out'].cpu().detach().numpy())[0], (RC3['out'].cpu().detach().numpy())[0], (RF1['out'].cpu().detach().numpy())[0], (RF2['out'].cpu().detach().numpy())[0]


def evaluation_Readouts_no_patch(net,READOUT_NETS,img,type):
	if type == 'png':
		I = np.array(io.imread(img)).astype(float)
		if np.amax(I) > 2:
			I = (I/255)
		else:
			I = I
		#I = transform.resize(I,(224,128))
		I = np.moveaxis(I,-1,0)
		x = torch.from_numpy(I)
		x = x.type(torch.FloatTensor)
	elif type == 'npy':
		I = np.load(img).astype(float)
		trans_im = I.copy()
		local_mean = np.mean(trans_im[0:8,27:100],axis = (0))
		band = np.tile(local_mean[np.newaxis,:,:],(11,1,1))
		local_std= np.std(trans_im[0:8,10:115])
		lum_noise = np.random.normal(0,local_std/10,(11,73))
		trans_im[8:19,27:100] = band+ np.tile(lum_noise[:,:,np.newaxis],(1,1,3))
		I = trans_im.copy()
		I = np.moveaxis(I,-1,0)
		I = I - 3
		x = torch.from_numpy(I)
		x = x.type(torch.FloatTensor)
	else:
		x = torch.load(img)
	x  = x.unsqueeze(0)
	x = x.to(device)
	outputs = net(x)
	RC1 = READOUT_NETS[0](outputs['conv1'])
	RC2 = READOUT_NETS[1](outputs['conv2'])
	RC3 = READOUT_NETS[2](outputs['conv3'])
	RF1 = READOUT_NETS[3](outputs['fc1'])
	RF2 = READOUT_NETS[4](outputs['fc2'])
	_, p1 = torch.max(RC1['out'].data, 1)
	_, p2 = torch.max(RC2['out'].data, 1)
	_, p3 = torch.max(RC3['out'].data, 1)
	_, p4 = torch.max(RF1['out'].data, 1)
	_, p5 = torch.max(RF2['out'].data, 1)
	return (p1.cpu().detach().numpy())[0], (p2.cpu().detach().numpy())[0], (p3.cpu().detach().numpy())[0], (p4.cpu().detach().numpy())[0], (p5.cpu().detach().numpy())[0], (RC1['out'].cpu().detach().numpy())[0], (RC2['out'].cpu().detach().numpy())[0], (RC3['out'].cpu().detach().numpy())[0], (RF1['out'].cpu().detach().numpy())[0], (RF2['out'].cpu().detach().numpy())[0]

def evaluation_Readouts_no_back(net,READOUT_NETS,img,type):
	if type == 'png':
		I = np.array(io.imread(img)).astype(float)
		if np.amax(I) > 2:
			I = (I/255)
		else:
			I = I
		#I = transform.resize(I,(224,128))
		I = np.moveaxis(I,-1,0)
		x = torch.from_numpy(I)
		x = x.type(torch.FloatTensor)
	elif type == 'npy':
		i = img
		img = i[:-9] + i[-4:]
		I_mask = np.load(i)
		MASK = np.mean(I_mask,axis = -1)
		MASK[MASK > 0.1] = 1
		MASK[MASK != 1] = 0
		I = np.load(img)
		SCENE = np.zeros(I.shape)
		SCENE[MASK==1] = I[MASK==1]
		I = SCENE.astype(float)
		I = np.moveaxis(I,-1,0)
		I = I - 3
		x = torch.from_numpy(I)
		x = x.type(torch.FloatTensor)
	else:
		x = torch.load(img)
	x  = x.unsqueeze(0)
	x = x.to(device)
	outputs = net(x)
	RC1 = READOUT_NETS[0](outputs['conv1'])
	RC2 = READOUT_NETS[1](outputs['conv2'])
	RC3 = READOUT_NETS[2](outputs['conv3'])
	RF1 = READOUT_NETS[3](outputs['fc1'])
	RF2 = READOUT_NETS[4](outputs['fc2'])
	_, p1 = torch.max(RC1['out'].data, 1)
	_, p2 = torch.max(RC2['out'].data, 1)
	_, p3 = torch.max(RC3['out'].data, 1)
	_, p4 = torch.max(RF1['out'].data, 1)
	_, p5 = torch.max(RF2['out'].data, 1)
	return (p1.cpu().detach().numpy())[0], (p2.cpu().detach().numpy())[0], (p3.cpu().detach().numpy())[0], (p4.cpu().detach().numpy())[0], (p5.cpu().detach().numpy())[0], (RC1['out'].cpu().detach().numpy())[0], (RC2['out'].cpu().detach().numpy())[0], (RC3['out'].cpu().detach().numpy())[0], (RF1['out'].cpu().detach().numpy())[0], (RF2['out'].cpu().detach().numpy())[0]

def evaluation(predictions, label):
    s = np.sum(predictions == label)
    return 100*s/len(predictions)

def training_curves(training_dir, training_set, nb_epoch, Readout = False, layer = ''):
    nb_mod = len(glob.glob(training_dir+'inst_*_%s'%training_set))
	Training_curv = np.zeros((nb_mod,nb_epoch))
	epochmax = np.zeros((nb_mod))
	for i in range(nb_mod):
		if Readout:
			Training_curv[i] = np.load( training_dir +'inst_%d_%s/readout_'%((i+1,training_set)) + layer + '/train_curve.npy' )
		else:
			Training_curv[i] = np.load( training_dir +'inst%dtrain_curve.npy' %((i)))
		epochmax[i] = np.argmax(Training_curv[i])
	return Training_curv, epochmax


# In[9]: COMPUTE ACTIVATIONS DCC


parser = argparse.ArgumentParser(description='Parsing variables for rendering images')

parser.add_argument('--gpu_id', default=0, type=int, metavar='N',
                    help='ID of Gpu to use')

parser.add_argument('--models_dir', default='', type=str, metavar='str',
                    help='dir where to find models and training curves')

parser.add_argument('--testing_dir', default='', type=str, metavar='str',
                    help='dir where to find models and training curves')

parser.add_argument('--training_set', default='CC', type=str, metavar='str',
                    help='to distiguish between CC and D65')

parser.add_argument('--testing_set', default='WCS', type=str, metavar='str',
                    help='to distiguish between WCS and all muns')

parser.add_argument('--testing_type', default='4illu', type=str, metavar='str',
                    help='to distiguish between the illuminant kind')

parser.add_argument('--testing_condition', default='normal', type=str, metavar='str',
                    help='to distiguish between the different conditions (nromal, no patch, wrong patch)')

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


TRAINING_CURV, EPOCHMAX = training_curves('./', 10, 90)
training_curv_c1,epochmax_c1 = training_curves('finetuning/', 30, args.training_set, Readout = True, layer = 'conv1')
training_curv_c2,epochmax_c2 = training_curves('finetuning/', 30, args.training_set, Readout = True, layer = 'conv2')
training_curv_c3,epochmax_c3 = training_curves('finetuning/', 30, args.training_set, Readout = True, layer = 'conv3')
training_curv_f1,epochmax_f1 = training_curves('finetuning/', 30, args.training_set, Readout = True, layer = 'fc1')
training_curv_f2,epochmax_f2 = training_curves('finetuning/', 30, args.training_set, Readout = True, layer = 'fc2')



nb_models = len(TRAINING_CURV)
pc1 =  np.zeros((nb_models,nb_obj,nb_illu,nb_ex))
out_c1 = np.zeros((nb_models,nb_obj,nb_illu,1600))
pc2 =  pc1.copy()
out_c2 = np.zeros((nb_models,nb_obj,nb_illu,1600))
pc3 =  pc1.copy()
out_c3 = np.zeros((nb_models,nb_obj,nb_illu,1600))
pf1 =  pc1.copy()
out_f1 = np.zeros((nb_models,nb_obj,nb_illu,1600))
pf2 =  pc1.copy()
out_f2 = np.zeros((nb_models,nb_obj,nb_illu,1600))


if args.model == 'Ref':
	model = MODELS.Ref()
elif args.model == 'Original':
    model = MODELS.Net2_4ter_norm()

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

dir_addr = '/home/alban/project_color_constancy/TRAINING/DATA/PNG/WCS/Test_4_illu_centered/'
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
	for muns in range(nb_obj):
		for ill in range(nb_illu):
			for exp in range(nb_ex):
				img = dir_addr + 'object%i_%s_%i.npy' %(muns,chr(ill+65),exp)
				pc1[m, muns, ill, exp], pc2[m,muns, ill, exp], pc3[m,muns, ill, exp], pf1[m,muns, ill, exp], pf2[m,muns, ill, exp], out_c1[m,muns, ill, exp], out_c2[m,muns, ill, exp], out_c3[m,muns, ill, exp], out_f1[m,muns, ill, exp], out_f2[m,muns, ill, exp] = evaluation_Readouts_no_patch(net, Readout_nets, img, 'npy')
	for munsell in range(nb_obj):
		EVAL[m,munsell] = evaluation(pf2[m,munsell],list_WCS_labels[munsell])
	print('Result = %d' %np.mean(EVAL[m]))

np.save('pc1_%s_%s_%s_%s.npy',pc1)
np.save('pc2_%s_%s_%s_%s.npy',pc2)
np.save('pc3_%s_%s_%s_%s.npy',pc3)
np.save('pf1_%s_%s_%s_%s.npy',pf1)
np.save('pf2_%s_%s_%s_%s.npy',pf2)

np.save('out_c1_%s_%s_%s_%s.npy',out_c1)
np.save('out_c2_%s_%s_%s_%s.npy',out_c2)
np.save('out_c3_%s_%s_%s_%s.npy',out_c3)
np.save('out_f1_%s_%s_%s_%s.npy',out_f1)
np.save('out_f2_%s_%s_%s_%s.npy',out_f2)

