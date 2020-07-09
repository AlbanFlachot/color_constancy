#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 11:10:29 2020

@author: alban

Script to do some analysis in Munsell coordinates
"""


# In[1]:



from __future__ import print_function, division

import matplotlib.pyplot as plt
import numpy as np
import pickle
import scipy.io as sio
import os



import sys
sys.path.append('../../')

from color_scripts import color_transforms as CT # scripts with a bunch of functions for color transformations
from error_measures_scripts import Error_measures as EM
from display_scripts import display as dis
from utils_scripts import algos


# Ignore warnings
import warnings

warnings.filterwarnings("ignore")


# In[2]:


def load_layer(path):
	import glob
	addrs = sorted(glob.glob(path + '/*.pickle'))
	LAYER = list()
	for addr in addrs:
		pickle_in = open(addr,'rb')
		layer = pickle.load(pickle_in)
		pickle_in.close()
		LAYER.extend(layer)
	return np.array(LAYER)


# In[9]: path to files

txt_dir_path = '../../txt_files/'
npy_dir_path = '../../npy_files/'
pickles_dir_path = '../pickles/'
figures_dir_path = '/home/alban/Dropbox/project_color_constancy/PAPER/figures/'

# In[9]: load mnodels results


def from330to8x40(X):
    '''
    Function to convert from array with one munsell dimension of 330 to an array with 2 dimensions (10,41),
    corresponding to the WCS coordinates
    Parameters:
        - X: array of shape = [...,330,...]
    Returns:
        - WCS_MAT: array of shape = [...,10,41,...] foolowing the WCS coordinates
    '''

    # List of WCS coordinates
    L = list()
    with open(txt_dir_path +"WCS_indx.txt") as f:
        for line in f:
           L.append(line.split())

    WCS_X = [ord(char[0][0].lower()) - 97 for char in L]
    WCS_Y = [int(char[0][1:]) for char in L]

    # identification of dim with size 330
    idx = np.array(X).shape.index(330)
    # move this dim to the first
    X = np.moveaxis(X,idx,0)
    # initialization of new array
    WCS_MAT = np.zeros(tuple([10,41]) + X.shape[1:])
    count = 0
    for i in range(330):
        WCS_MAT[WCS_X[i],WCS_Y[i]] = X[count].astype(float)
        count +=1
    return np.moveaxis(WCS_MAT,(0,1),(idx,idx+1)) # move dimensions in the right order



def load_pickle(path):
    import pickle
    f = open(path,"rb")
    return pickle.load(f, fix_imports=True, encoding='latin1', errors="strict")



def VXY2VHC(VXY, muns = 'True'):
    '''
    Fuction that converts Munsell representation from cardinal (Value X, Y) to cylindrical (Value, Hue, Chroma)
    '''

    shape = VXY.shape
    VXY = VXY.reshape(-1,3)

    VHC = VXY.copy()
    VHC[:,-1] = np.linalg.norm(VXY[:,1:], axis = -1)
    #import dbg; dbg.set_trace()
    VHC[:,1] = (np.arccos(VXY[:,1]/VHC[:,-1])*np.sign(VXY[:,2]))
    VHC[VHC[:,-1] == 0,1] = 0
    if muns:
        VHC[:,1] = VHC[:,1]*180/np.pi/4.5
    VHC = VHC.reshape(shape)
    return VHC

# In[9]: load mnodels results
conditions = ['normal','no_patch', 'wrong_illu', 'no_back', 'D65']


layers = ['fc2','fc1', 'c3', 'c2', 'c1']

DE_VHC = {}
Errors_D65_D65 = {}

path = pickles_dir_path

for layer in layers[:1]:
    DE_VHC[layer] = {}
    for condition in conditions[:-1]:
        DE_VHC[layer][condition] = load_pickle(path + 'Errors_Original_CC_4illu_WCS_%s_%s.pickle'%(layer, condition))['DE_3D']
    DE_VHC[layer][conditions[-1]] = load_pickle(path + 'Errors_Original_D65_4illu_WCS__%s_normal.pickle'%(layer))['DE_3D']
    #Errors_D65_D65[layer] = load_pickle(path + 'Errors_Original_D65_D65_WCS_%s_normal.pickle'%(layer))

if not os.path.exists(figures_dir_path):
    os.mkdir(figures_dir_path)



def error_muns(DE_VHC):
    '''
    Function that computes the error, in Munsell coordinates
    '''
    Hue_arr = np.arange(0,2*np.pi,2*np.pi/80)
    shape = DE_VHC.shape
    DE_VHC = DE_VHC.reshape(-1,3)
    DE_VHC[:,1] = DE_VHC[:,1]%80
    Hue_diff = Hue_arr[DE_VHC[:,1].astype(int)]
    error_hue = np.arccos(np.cos(Hue_diff))*np.sign(np.sin(Hue_diff))
    DE_VHC[:,1] = error_hue*180/np.pi/9
    DE_VHC = DE_VHC.reshape(shape)
    return DE_VHC

#for layer in layers[:1]:
#    for condition in conditions[:-1]:
#        DE_VHC[layer][condition] = error_muns(DE_VHC[layer][condition])

dis.DEFINE_PLT_RC(type = 1)

def histo_VHC(DE_VHC, titl, save = True, path = 0, col = 'k'):
    f, axes = plt.subplots(1,3,sharey = True, figsize = (10,3))
    axes[0].hist(DE_VHC.reshape(-1,3)[:,0], bins = np.arange(-10.5,10.5,1), color = col)
    axes[0].set_xlabel('Value')
    axes[0].set_ylabel('Number of cases')
    axes[0].ticklabel_format(style = 'sci',scilimits = [-3,3])

    axes[1].hist(DE_VHC.reshape(-1,3)[:,1], bins = np.arange(-20.5,20.5,1),color =col)
    axes[1].set_xlabel('Hue')
    axes[1].set_title(titl)
    axes[2].hist(DE_VHC.reshape(-1,3)[:,2]*2, bins = np.arange(-16.5,16.5,1),color =col)
    axes[2].set_xlabel('Chroma')
    axes[2].set_xticks(np.arange(-16,17,8))
    plt.subplots_adjust(top=0.88, bottom=0.19, left=0.095, right=0.95, hspace=0.1,
                    wspace=0.1)
    plt.show()
    if save:
        f.savefig(path)

def histo2D_VHC(DE_VHC, titl, save = True, path = 0, col = 'k'):
    f, axes = plt.subplots(1,3,sharey = True, figsize = (10,3))
    axes[0].hist2d(DE_VHC.reshape(-1,3)[:,0], DE_VHC.reshape(-1,3)[:,1], bins = [np.arange(-10.5,10.5,1),np.arange(-20.5,20.5,1)], cmap = 'Greys')
    axes[0].set_xlabel('Value')
    axes[0].set_ylabel('Hue')
    axes[0].set_ylim(-20,20)
    axes[0].ticklabel_format(style = 'sci',scilimits = [-3,3])

    axes[1].hist2d(DE_VHC.reshape(-1,3)[:,0], DE_VHC.reshape(-1,3)[:,2]*2, bins = [np.arange(-10.5,10.5,1), np.arange(-16.5,16.5,1)], cmap = 'Greys')
    axes[1].set_xlabel('Value')
    axes[1].set_ylabel('Chroma')
    #axes[1].set_xticks(np.arange(-16,17,8))
    axes[1].set_title(titl)
    
    axes[2].hist2d(DE_VHC.reshape(-1,3)[:,1], DE_VHC.reshape(-1,3)[:,2]*2, bins = [np.arange(-20.5,20.5,1), np.arange(-16.5,16.5,1)], cmap = 'Greys')
    axes[2].set_xlabel('Hue')
    axes[2].set_ylabel('Chroma')
    axes[2].set_ylim(-20,20)
    #axes[2].set_xticks(np.arange(-16,17,8))
    plt.subplots_adjust(top=0.88, bottom=0.19, left=0.095, right=0.95, hspace=0.1,
                    wspace=0.19)
    plt.show()
    if save:
        f.savefig(path)

for layer in layers[:1]:
    for condition in conditions[:]:
        histo_VHC(DE_VHC[layer][condition],titl = '%s'%( condition),save = True, path = figures_dir_path + 'error_muns_coord/%s_%s.png'%(layer, condition))

for layer in layers[:1]:
    for condition in conditions[:]:
        histo2D_VHC(DE_VHC[layer][condition],titl = '%s'%( condition),save = True, path = figures_dir_path + 'error_muns_coord/2D_%s_%s.png'%(layer, condition))
### What if model were to choose random munsell?


list_WCS_labels = algos.compute_WCS_Munsells_categories()
MUNSELL_VHC = np.load(npy_dir_path +'MUNS_COORDINATES_VHC.npy')


RANDOM_VAL = np.random.choice(MUNSELL_VHC[:,0],int(DE_VHC[layer][condition].size/3))
RANDOM_HUE = np.random.choice(MUNSELL_VHC[:,1],int(DE_VHC[layer][condition].size/3))
RANDOM_CHROMA = np.random.choice(MUNSELL_VHC[:,2],int(DE_VHC[layer][condition].size/3)).astype(int)

RANDOM_VHC = np.array([RANDOM_VAL, RANDOM_HUE, RANDOM_CHROMA]).T
ERROR_RANDOM_VHC = -(RANDOM_VHC.reshape(-1,330,3) - MUNSELL_VHC[list_WCS_labels])
ERROR_RANDOM_VHC = error_muns(ERROR_RANDOM_VHC)

histo_VHC(ERROR_RANDOM_VHC,titl = 'random', save = True, path = figures_dir_path + 'error_muns_coord/random.png', col = 'grey')

histo2D_VHC(ERROR_RANDOM_VHC,titl = 'random', save = True, path = figures_dir_path + 'error_muns_coord/2D_random.png')

