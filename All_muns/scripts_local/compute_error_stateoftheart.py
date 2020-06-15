#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 17:58:48 2020

@author: alban

Scripts to compute errors for state of the art
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

parser.add_argument('--layer', default='fc2', type=str, metavar='str',
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

    img_paths = np.loadtxt(txt_dir_path + 'wcs_lms_1600_validation.txt', dtype=str)
    train_labels = [int(re.search('object(.*?)/', addr).group(1)) for addr in img_paths]
    sequence_1600 = np.array([train_labels[i] for i in range(0,train_labels.__len__(),28)])
    sequence_330 = sequence_1600[sequence_1600<330]
    #print(path2activations + '_' + layer + '_' + NetType + '_' + training_set + '_' + Testing_type + '_' + testing_set + '_' + testing_condition +'.npy')
    if (NetType == 'Original') | (NetType == 'ConvNet'):
        #import pdb; pdb.set_trace()
        OUT_soft = EM.softmax(np.load(path2activations + '_' + layer + '_' + NetType + '_' + training_set + '_' + Testing_type + '_' + testing_set + '_' + testing_condition +'.npy'))
    else:
        paths = glob.glob(path2activations + '*')
        if Testing_type == '4illu':
            s = (330,4,10,1600)
            OUT_soft = EM.softmax(np.array([load_layer(p).reshape(s) for p in paths]))
            OUT_soft = OUT_soft[:,np.argsort(sequence_330)]
        else:
            s = (1600,1,28,1600)
            OUT_soft = EM.softmax(np.array([load_layer(p).reshape(s) for p in paths[:2]]))
            OUT_soft = OUT_soft[:,np.argsort(sequence_1600)]
        OUT_soft = OUT_soft[:,:,:,:,np.argsort(sequence_1600)]
    return OUT_soft


def computeErrors(path2activations, NetType, training_set, Testing_type, testing_set, layer, testing_condition, dim = '3D', space = 'CIELab', order = 'alpha'):
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
    OUTS = np.genfromtxt(path2activations + NetType + '/' + NetType + '_original_0.csv', delimiter=',').astype('int')
    OUTS = OUTS.reshape(1,330,4,-1,4)
    list_WCS_labels = algos.compute_WCS_Munsells_categories() # import idxes of WCS munsells among the 1600

    muns_alpha2num = sorted(range(1600), key = str)
    muns_num2alpha = [muns_alpha2num.index(l) for l in range(1600)] # labels of muns in alpha order

    WCS_num2alpha = [sorted(range(1600), key = str).index(l) for l in list_WCS_labels] # labels of WCS in alpha order
    WCS_alpha2num = [OUTS[0,:,0,0,-1].tolist().index(i) for i in WCS_num2alpha] # indx of WCS in alpha order


    OUTS = OUTS[:,WCS_alpha2num]

    # Compute Delta E ------------------------------------------------------------------------------------
    DE = EM.PREDICTION_LAB(OUTS[:,:,:,:,2], test_WCS = (testing_set =='WCS'), space = 'CIELab', order = 'alpha')
    DE_3D = EM.PREDICTION_3D(OUTS[:,:,:,:,2], test_WCS = (testing_set =='WCS'), space = 'Munsell')
    DE_3D = EM.error_muns(DE_3D)

    # Compute Accuracy ------------------------------------------------------------------------------------
    nb_mod = OUTS.shape[0]
    nb_obj = OUTS.shape[1]
    nb_illu = OUTS.shape[2]
    Eval = np.zeros((nb_mod, nb_obj, nb_illu))
    Eval5 = np.zeros((nb_mod, nb_obj, nb_illu))
    Accu_munscube = np.zeros((nb_mod, nb_obj, nb_illu))
    for mod in range(nb_mod):
        for muns in range(nb_obj):
            for illu in range(nb_illu):
                Eval[mod,muns,illu] = 100*OUTS[mod, muns, illu,:,0].sum()/OUTS[mod, muns, illu,:,0].size
                Eval5[mod, muns, illu] = 100*OUTS[mod, muns, illu,:,1].sum()/OUTS[mod, muns, illu,:,1].size
                Accu_munscube[mod, muns, illu] =  EM.evaluation_munscube(DE_3D[mod,muns,illu])

    # Compute CCI ------------------------------------------------------------------------------------
    WCS_LAB_4 = np.load(npy_dir_path +'LAB_WCS_ABCD.npy')
    LAB_WCS = np.load(npy_dir_path +'WCS_LAB.npy')
    Displacement_LAB = WCS_LAB_4.T - LAB_WCS.T

    DE_WCS_all_illu = np.linalg.norm(Displacement_LAB, axis = (-1))
    CCI = 1 - np.moveaxis(DE, -1, 1)/DE_WCS_all_illu.T

    return Eval, DE, CCI, DE_3D, Eval5, Accu_munscube





# In[9]: Initialization variables

path = '/home/alban/Documents/pytorch/project_color_constancy/WCS_Xp/git/color_constancy/csv_files/'

OUTS = np.genfromtxt('/home/alban/Documents/pytorch/project_color_constancy/WCS_Xp/git/color_constancy/csv_files/alban_net/alban_net_original_0.csv', delimiter=',').astype('int')

NetType = args.NetType

Testing_type = args.testing_type
Testing_set = args.testing_set

path2activations = args.path2activations
#path2activations = '/home/alban/mnt/awesome/arash/Software/repositories/kernelphysiology/python/src/kernelphysiology/dl/pytorch/activations/Test_4_illu/fc/wcs_lms_1600/original/'
#path2activations = '/home/alban/mnt/awesome/arash/Software/repositories/kernelphysiology/python/src/kernelphysiology/dl/pytorch/activations/fc/wcs_lms_1600/original/'


# In[9]: Compute errors CIELAb

#Accu, DE, CCI = computeErrors(path2activations, NetType, args.training_set, Testing_type, args.testing_set, args.layer, args.testing_condition)



#save_pickle('../pickles/Errors_%s_%s_%s_%s_%s_%s.pickle'%(NetType, args.training_set, Testing_type, Testing_set, args.layer, args.testing_condition), {'Accu': Accu, 'DE': DE, 'CCI': CCI})


# In[9]: Compute errors Munsells

Accu, DE, CCI, DE_3D, Accu5, Accu_munscube = computeErrors(path, 'alban_net', 'CC', '4illu', 'WCS', 'fc2', 'normal')
