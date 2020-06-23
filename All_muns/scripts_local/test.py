#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 11:48:28 2020

@author: alban
"""



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

nb_mod = 1
nb_objects = 330
nb_illu = 4

#import torch.nn.functional as F


txt_dir_path = '../../txt_files/'
npy_dir_path = '../../npy_files/'


OUTS = np.genfromtxt('/home/alban/Documents/pytorch/project_color_constancy/WCS_Xp/git/color_constancy/csv_files/alban_net/alban_net_original_0.csv', delimiter=',').astype('int')

OUTS = OUTS.reshape((nb_mod, nb_objects, nb_illu,-1,4))
#DE = EM.PREDICTION_LAB(OUTS[:,:,:,:,2], test_WCS = True, space = 'CIELab', order = 'alpha')


PREDICTION = OUTS[:,:,:,:,2]
list_WCS_labels = algos.compute_WCS_Munsells_categories() # import idxes of WCS munsells among the 1600

muns_alpha2num = sorted(range(1600), key = str)
muns_num2alpha = [muns_alpha2num.index(l) for l in range(1600)] # labels of muns in alpha order

WCS_num2alpha = [sorted(range(1600), key = str).index(l) for l in list_WCS_labels] # labels of WCS in alpha order
WCS_alpha2num = [OUTS[0,:,0,0,-1].tolist().index(i) for i in WCS_num2alpha] # indx of WCS in alpha order

PREDICTION = OUTS[:,WCS_alpha2num][:,:,:,:,2]

data_training = 'all'
order = 'alpha'

MUNSELL_LAB = np.load(npy_dir_path +'MUNSELL_LAB.npy')
PREDICTION_ERROR = np.zeros((PREDICTION.shape))
for m in range(PREDICTION.shape[0]):
    for i in range(PREDICTION.shape[1]):
        for ill in range(PREDICTION.shape[2]):
            for exp in range(PREDICTION.shape[3]):
                if data_training == 'all':
                    if order == 'alpha':
                        dist = np.linalg.norm(MUNSELL_LAB[list_WCS_labels[i]] - MUNSELL_LAB[muns_alpha2num[PREDICTION[m,i,ill,exp]]])
                    else:
                        dist = np.linalg.norm(MUNSELL_LAB[list_WCS_labels[i]] - MUNSELL_LAB[sorted(range(1600), key = str)[PREDICTION[m,i,ill,exp].astype(int).tolist()]])
                else:
                    dist = np.linalg.norm(MUNSELL_LAB[list_WCS_labels[i]] - MUNSELL_LAB[list_WCS_labels[PREDICTION[m,i,ill,exp].astype(int).tolist()]])
                PREDICTION_ERROR[m,i,ill,exp] = dist
