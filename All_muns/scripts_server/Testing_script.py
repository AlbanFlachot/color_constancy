#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 16:18:07 2019

@author: alban
"""

# In[1]:



from __future__ import print_function, division
import os
import torch
from skimage import io, transform
import numpy as np

import pickle
import MODELS as M
import matplotlib.patheffects as PathEffects
import scipy.io as sio
from random import randint
import FUNCTIONS as F

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")



# In[2]:

import matplotlib.patheffects as PathEffects

os.environ["CUDA_VISIBLE_DEVICES"]="3"

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
#cudnn.benchmark = True


# In[3]: FUNCTIONS

## TEST MODEL
def TESTING_FUNC(datatype,model,weights,list_WCS_labels = None):
        
    with open("val_labels" +datatype, "rb") as fp:   # Unpickling
        val_lab = pickle.load(fp)
    
    with open("val_ima" +datatype, "rb") as fp:   # Unpickling
        val_im = pickle.load(fp)
    
    LIST = [[None]] * (max(val_lab)+1)
    for count in range(len(val_lab)):
        LIST[val_lab[count]] = LIST[val_lab[count]]+[val_im[count]]
    
    
    net = model
    
    net.load_state_dict(torch.load(weights))
    net.to(device)
    net.eval()
    
    PREDICTION = np.ones((len(LIST),50))
    for l in range(len(LIST)):
        PREDICTION[l] = testing(net, LIST[l][1:51],'npy')
    
    EVAL = np.zeros(len(LIST))
    for i in range(len(EVAL)):
        if list_WCS_labels == None:
            EVAL[i] = evaluation(PREDICTION[i],i)
        else:
            EVAL[i] = evaluation(PREDICTION[i],list_WCS_labels[i])
    return PREDICTION,EVAL


def testing(net,list_obj1,type):
    count = 0
    #print(type)
    predictions = np.zeros(len(list_obj1))
    for i in list_obj1:
        if type == 'png':
            I = np.array(io.imread(i)).astype(float)
            if np.amax(I) > 2:
                I = (I/255)
            else:
                I = I
            #I = transform.resize(I,(224,128))
            I = np.moveaxis(I,-1,0)
            x = torch.from_numpy(I)
            x = x.type(torch.FloatTensor)
        elif type == 'npy':
            I = np.load(i).astype(float)
            I = np.moveaxis(I,-1,0)
            I = I - 3
            x = torch.from_numpy(I)
        else:
            x = torch.load(i)
        x  = x.unsqueeze(0)
        x = x.to(device)
        if x.size()[2] < 220:
            continue
        outputs = net(x)
        _, predictions[count] = torch.max(outputs['out'].data, 1)
        count +=1
    return predictions

def evaluation(predictions, label):
    s = np.sum(predictions == label)
    return 100*s/len(predictions)

def retrieve_activations(net,img,type = 'npy'):
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
	out = outputs['out'].cpu().detach().numpy()
	return conv1[0],conv2[0],conv3[0],fc1[0],fc2[0], out[0], (p.cpu().detach().numpy())[0]

def retrieve_activations_no_patch(net,img,type = 'npy'):
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
	conv1 = np.amax(outputs['conv1'].cpu().detach().numpy(),axis = (2,3))
	conv2 = np.amax(outputs['conv2'].cpu().detach().numpy(),axis = (2,3))
	conv3 = np.amax(outputs['conv3'].cpu().detach().numpy(),axis = (2,3))
	fc1 = outputs['fc1'].cpu().detach().numpy()
	fc2 = outputs['fc2'].cpu().detach().numpy()
	_, p = torch.max(outputs['out'].data, 1)
	out = outputs['out'].cpu().detach().numpy()
	return conv1[0],conv2[0],conv3[0],fc1[0],fc2[0], out[0], (p.cpu().detach().numpy())[0]

def retrieve_activations_wrong_illu(net,img,type,val_im_empty_scenes):
	index_illu = randint(0, len(val_im_empty_scenes)-1)
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
		image = img[:-9] + img[-4:]
		I_mask = np.load(img)
		MASK = np.mean(I_mask,axis = -1)
		MASK[MASK > np.amax(MASK)/255] = 1
		MASK[MASK != 1] = 0
		I = np.load(image)  
		scene = val_im_empty_scenes[index_illu]
		SCENE = np.load(scene)                     
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
	conv1 = np.amax(outputs['conv1'].cpu().detach().numpy(),axis = (2,3))
	conv2 = np.amax(outputs['conv2'].cpu().detach().numpy(),axis = (2,3))
	conv3 = np.amax(outputs['conv3'].cpu().detach().numpy(),axis = (2,3))
	fc1 = outputs['fc1'].cpu().detach().numpy()
	fc2 = outputs['fc2'].cpu().detach().numpy()
	_, p = torch.max(outputs['out'].data, 1)
	out = outputs['out'].cpu().detach().numpy()
	return conv1[0],conv2[0],conv3[0],fc1[0],fc2[0], out[0], (p.cpu().detach().numpy())[0]

# In[7]: PLOT TRAINING CURVES
nb_mod = 10
nb_epoch = 90
nb_obj = 330

## Xp set
####---------------------------------------------------------------------------------------------------------------------
Training_curv = np.zeros((nb_mod,nb_epoch))
Training_curv_D65 = np.zeros((nb_mod,nb_epoch))
epochmax = np.zeros((nb_mod))
epochmax_D65 = np.zeros((nb_mod))

for i in range(nb_mod):
    #Training_curv[i] = np.load('../mod_N2/inst_%dtrain_curve.npy' %((i+1)))
    Training_curv[i] = np.load('inst%itrain_curve.npy' %(i))
    epochmax[i] = np.argmax(Training_curv[i])
    print('inst %i achieves max accuracy of %d at epoch %i' %(i, np.amax(Training_curv[i]),np.argmax(Training_curv[i])))

epochmax[[2,-3]] = 37

# In[9]:

with open("/home/alban/project_color_constancy/PYTORCH/WCS/train_centered/WCS/ima_empty_scenes.txt", "rb") as fp:   # Unpickling
        val_im_empty_scenes = pickle.load(fp)

WCS_muns = list()
with open("/home/alban/project_color_constancy/PYTORCH/WCS/WCS_muns.txt") as f:
    for line in f:
       WCS_muns.append(line.split()[0])
       
       
All_muns = list()
with open("/home/alban/project_color_constancy/PYTORCH/WCS/munsell_labels.txt") as f:
    for line in f:
       All_muns.append(line.split()[0])

WCS_i = [All_muns.index(WCS_muns[i]) for i in range(len(WCS_muns))]

shape = (nb_mod, 330, 4,10)
conv1 = np.zeros((nb_mod, 330, 4,10, 16))
conv2 = np.zeros((nb_mod, 330, 4,10, 32))
conv3 = np.zeros((nb_mod, 330, 4,10, 64))
fc1 = np.zeros((nb_mod, 330, 4,10, 250))
fc2 = np.zeros((nb_mod, 330, 4,10, 250))
out = np.zeros((nb_mod, 330, 4,10, 1600))
predictions = np.zeros(shape)
EVAL = np.zeros(shape[:3])

dir_addr = '/home/alban/project_color_constancy/TRAINING/DATA/muns_illu/'
for m in range(nb_mod):
	print('Evaluation of model %i' %m)
	model = M.Net2_4ter_norm()
	#weights = '../mod_N2/inst_%d/model_3conv_epoch_%i.pt' %(m+1,epochmax[m])
	weights = 'INST/inst%i/epoch_%i.pt' %(m, epochmax[m])
	net = model
	net.to(device)
	net.load_state_dict(torch.load(weights))
	net.eval()
	for muns in range(330):
		for ill in range(4):
			for exp in range(10):
				img_addr = dir_addr + 'object%i/object%i_%s_%i.npy' %(muns,chr(ill+65),exp)
				conv1[m,muns,ill,exp], conv2[m,muns,ill,exp], conv3[m,muns,ill,exp], fc1[m,muns,ill,exp], fc2[m,muns,ill,exp], out[m,muns,ill,exp],  predictions[m,muns,ill,exp] = retrieve_activations_wrong_illu(net, img_addr,'npy', val_im_empty_scenes)
			EVAL[m,muns,ill] = evaluation(predictions[m,muns,ill],WCS_i[muns])
	print('Result = %d' %np.mean(EVAL[m]))


np.save('layers/conv1_centered_muns_illu.npy', conv1)
np.save('layers/conv2_centered_muns_illu.npy', conv2)
np.save('layers/conv3_centered_muns_illu.npy', conv3)
np.save('layers/fc1_centered_muns_illu.npy', fc1)
np.save('layers/fc2_centered_muns_illu.npy', fc2)
np.save('layers/predictions_centered_muns_illu.npy', predictions)
np.save('layers/out_centered_muns_illu.npy', out)
np.save('layers/evaluation_centered_muns_illu.npy', EVAL)



