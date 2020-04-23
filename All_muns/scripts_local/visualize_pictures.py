#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 14:16:04 2020

@author: alban
"""

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


dir1_path = '/home/alban/mnt/awesome/alban/project_color_constancy/TRAINING/DATA/PNG/WCS/Test_4_illu_centered/'

dir2_path = '/home/alban/mnt/awesome/alban/project_color_constancy/TRAINING/DATA/PNG/Test_4_illu/centered/'

def visualize_npy_im(im_path):
    im = np.load(im_path)
    fig = plt.figure()
    ax1 = plt.subplot(111)
    ax1.imshow(im/np.amax(im))
    plt.show()
    plt.close()
    print('Mean pix = %s' %np.array2string(np.mean(im, axis = (0,1))))
    print('Max pix = %s' %np.array2string(np.amax(im, axis = (0,1))))
    print('Std pix = %s' %np.array2string(np.std(im, axis = (0,1))))
    print('Size image = %i' %im.size)



visualize_npy_im(dir1_path + 'object150_A_0.npy')
visualize_npy_im(dir2_path + 'object150_A_0.npy')


visualize_npy_im(dir1_path + 'object150_B_0.npy')
visualize_npy_im(dir2_path + 'object150_B_0.npy')


visualize_npy_im(dir1_path + 'object150_C_0.npy')
visualize_npy_im(dir2_path + 'object150_C_0.npy')


visualize_npy_im(dir1_path + 'object150_D_0.npy')
visualize_npy_im(dir2_path + 'object150_D_0.npy')



