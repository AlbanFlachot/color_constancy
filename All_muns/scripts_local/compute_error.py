#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 11:29:59 2020

@author: alban
"""


# In[1]:



from __future__ import print_function, division

import numpy as np
import pickle
import os
import glob
import re


import sys
sys.path.append('../../')

from color_scripts import color_transforms as CT # scripts with a bunch of functions for color transformations
from error_measures_scripts import Error_measures as EM
#from display_scripts import display as dis
from utils_scripts import algos


# Ignore warnings
import warnings

warnings.filterwarnings("ignore")



#import torch.nn.functional as F


# In[38] PARSER definition
import argparse

parser = argparse.ArgumentParser(description='Parsing variables for rendering images')

parser.add_argument('--training_set', default='CC', type=str, metavar='str',
                    help='to distiguish between CC and D65')

parser.add_argument('--NetType', default='Original', type=str, metavar='str',
                    help='type of model to analyze')

parser.add_argument('--layer', default='', type=str, metavar='str',
                    help='layers activations we are looking at')

parser.add_argument('--testing_set', default='WCS', type=str, metavar='str',
                    help='to distiguish between WCS and all muns')

parser.add_argument('--testing_type', default='4illu', type=str, metavar='str',
                    help='to distiguish between the illuminant kind')

parser.add_argument('--testing_condition', default='normal', type=str, metavar='str',
                    help='to distiguish between the different conditions (nromal, no patch, wrong patch)')

parser.add_argument('--path2activations', default='/home/alban/mnt/DATA/project_color_constancy/data/training_centered/All_muns/layers/', type=str, metavar='str',
                    help='path to directory where activations (npy files) are')


args = parser.parse_args()

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

# In[9]: load mnodels results



def LoadandComputeOutputs(path2activations, NetType, training_set, Testing_type, testing_set, layer, testing_condition):
    '''
    Function that loads previously saved outputs and computes the softmax. Different cases for different models.
    PARAMETERS:
        - path2activations: path to directory where activations (npy files) are.
        - NetType: type of model to analyze.
        - Testing_type: Useful for the illuminants. Generaly should be 4illu.
    OUTPUTS:
        - Out_soft: Outputs after softmax.
    '''

    #img_paths = np.loadtxt(txt_dir_path + 'wcs_lms_1600_validation.txt', dtype=str)
    #train_labels = [int(re.search('object(.*?)/', addr).group(1)) for addr in img_paths]
    #sequence_1600 = np.array([train_labels[i] for i in range(0,train_labels.__len__(),28)])
    #sequence_330 = sequence_1600[sequence_1600<330]
    #print(path2activations + '_' + layer + '_' + NetType + '_' + training_set + '_' + Testing_type + '_' + testing_set + '_' + testing_condition +'.npy')
    OUT_soft = EM.softmax(np.load(path2activations + layer + '_' + NetType + '_' + training_set + '_' + Testing_type + '_' + testing_set + '_' + testing_condition +'.npy'))
    print(OUT_soft.shape)
    print(path2activations + layer + '_' + NetType + '_' + training_set + '_' + Testing_type + '_' + testing_set + '_' + testing_condition +'.npy')

    return OUT_soft


def computeErrors(path2activations, NetType, training_set, Testing_type, testing_set, layer, testing_condition, dim = '3D', space = 'CIELab'):
    '''
    Function that loads previously saved outputs and computes the softmax. Different cases for different models.
    Then computes DE error of the predictions from weighted average of the outputs

    PARAMETERS:
        - path2activations: path to directory where activations (npy files) are.
        - NetType: type of model to analyze.
        - Testing_type: Useful for the illuminants. Generaly should be 4illu.
    OUTPUTS:
        - DE: Delta E error.
    '''

    OUT_soft = LoadandComputeOutputs(path2activations, NetType, training_set, Testing_type, testing_set, layer, testing_condition) # Load output of last layer

    if (NetType != 'Original') | (NetType != 'RefConvNet'):
        list_WCS_labels = algos.compute_WCS_Munsells_categories() # import idxes of WCS munsells among the 1600
        muns_alpha2num = sorted(range(1600), key = str)
        muns_num2alpha = [muns_alpha2num.index(l) for l in range(1600)] # labels of muns in alpha order

        WCS_num2alpha = [sorted(range(1600), key = str).index(l) for l in list_WCS_labels] # labels of WCS in alpha order
        WCS_alpha2num = [muns_num2alpha[l] for l in list_WCS_labels] # indx of WCS in alpha order
        OUT_soft = OUT_soft[:,:,:,:,muns_num2alpha]

    # Compute Delta E ------------------------------------------------------------------------------------
    DE = EM.WEIGHTED_PREDICTION_LAB(OUT_soft, test_WCS = (testing_set =='WCS'), space = 'CIELab')
    DE_3D = EM.PREDICTION_3D(OUT_soft.argmax(axis = -1), test_WCS = (testing_set =='WCS'), space = 'Munsell')
    DE_3D = EM.error_muns(DE_3D)

    # Compute Accuracy ------------------------------------------------------------------------------------
    nb_mod = OUT_soft.shape[0]
    nb_obj = OUT_soft.shape[1]
    nb_illu = OUT_soft.shape[2]
    Eval = np.zeros((nb_mod, nb_obj, nb_illu))
    Eval5 = np.zeros((nb_mod, nb_obj, nb_illu))
    Accu_munscube = np.zeros((nb_mod, nb_obj, nb_illu))
    for mod in range(nb_mod):
        for muns in range(nb_obj):
            for illu in range(nb_illu):
                Eval[mod, muns, illu] = EM.evaluation(np.argmax(OUT_soft[mod, muns, illu], axis=-1), list_WCS_labels[muns])
                Eval5[mod, muns, illu] = EM.evaluation5(np.argsort(OUT_soft[mod, muns, illu], axis=-1)[:,-5:], list_WCS_labels[muns])
                Accu_munscube[mod, muns, illu] = EM.evaluation_munscube(DE_3D[mod, muns, illu])

    # Compute CCI ------------------------------------------------------------------------------------
    WCS_LAB_4 = np.load(npy_dir_path +'LAB_WCS_ABCD.npy')
    LAB_WCS = np.load(npy_dir_path +'WCS_LAB.npy')
    Displacement_LAB = WCS_LAB_4.T - LAB_WCS.T

    DE_WCS_all_illu = np.linalg.norm(Displacement_LAB, axis = (-1))
    CCI = 1 - np.moveaxis(DE, -1, 1)/DE_WCS_all_illu.T

    return Eval, DE, CCI, DE_3D, Eval5, Accu_munscube


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


# In[9]: Initialization variables


NetType = args.NetType

Testing_type = args.testing_type
Testing_set = args.testing_set

path2activations = args.path2activations
#path2activations = '/home/alban/mnt/awesome/arash/Software/repositories/kernelphysiology/python/src/kernelphysiology/dl/pytorch/activations/Test_4_illu/fc/wcs_lms_1600/original/'
#path2activations = '/home/alban/mnt/awesome/arash/Software/repositories/kernelphysiology/python/src/kernelphysiology/dl/pytorch/activations/fc/wcs_lms_1600/original/'


# In[9]: Compute errors CIELAb

#Accu, DE, CCI = computeErrors(path2activations, NetType, args.training_set, Testing_type, args.testing_set, args.layer, args.testing_condition)

def save_pickle(path, dict):
    import pickle
    f = open(path,"wb")
    pickle.dump(dict,f)
    f.close()

def load_pickle(path):
    import pickle
    f = open(path,"rb")
    return pickle.load(f)

if not os.path.exists('../pickles'):
    os.mkdir('../pickles')

#save_pickle('../pickles/Errors_%s_%s_%s_%s_%s_%s.pickle'%(NetType, args.training_set, Testing_type, Testing_set, args.layer, args.testing_condition), {'Accu': Accu, 'DE': DE, 'CCI': CCI})


# In[9]: Compute errors Munsells

Accu, DE, CCI, DE_3D, Accu5, Accu_munscube = computeErrors(path2activations, NetType, args.training_set, Testing_type, args.testing_set, args.layer, args.testing_condition)

print(Accu.mean())

if not os.path.exists('../pickles'):
    os.mkdir('../pickles')

save_pickle('../pickles/Errors_%s_%s_%s_%s_%s_%s.pickle'%(NetType, args.training_set, Testing_type, Testing_set, args.layer, args.testing_condition), {'Accu': Accu, 'DE': DE, 'CCI': CCI, 'DE_3D': DE_3D, 'Accu5': Accu5, 'Accu_munscube': Accu_munscube})
