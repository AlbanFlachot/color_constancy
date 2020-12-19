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
import cv2


import sys
sys.path.append('../../')

from color_scripts import color_transforms as CT # scripts with a bunch of functions for color transformations
from error_measures_scripts import Error_measures as EM
from display_scripts import display as dis
from utils_scripts import algos


# Ignore warnings
import warnings


dir_path = '/home/alban/mnt/awesome/alban/DATA/IM_CC/masks_5illu/'

def visualize_npy_im(im_path, no_patch = False,patch_nb = 0, save=False, add = 'a', correction = False, coef = 0, masking =False, mask = 0):
    im = np.load(im_path)
    if no_patch == True:
        trans_im = im.copy()
        local_mean = np.mean(trans_im[0:8,27 + int(12*patch_nb) : 27+ int(12*(patch_nb+1))+1],axis = (0))
        band = np.tile(local_mean[np.newaxis,:,:],(11,1,1))
        local_std= np.std(trans_im[0:8,10:115])
        lum_noise = np.random.normal(0,local_std/10,(11,13))
        trans_im[8:19, 27 + int(12*patch_nb) : 27+ int(12*(patch_nb+1))+1] = band+ np.tile(lum_noise[:,:,np.newaxis],(1,1,3))
        im = trans_im.copy()
    if correction > 0:
        im = im/coef
    if masking:
        im = im*(np.stack((mask, mask, mask), axis = 2))
    fig = plt.figure()
    ax1 = plt.subplot(111)
    ax1.imshow(im/np.amax(im))
    plt.show()
    if save:
        fig.savefig(add, dpi = 400)
    plt.close()
    print('Mean pix = %s' %np.array2string(np.mean(im, axis = (0,1))))
    print('Max pix = %s' %np.array2string(np.amax(im, axis = (0,1))))
    print('Std pix = %s' %np.array2string(np.std(im, axis = (0,1))))
    print('Size image = %i' %im.size)

def receptive_field(layer, stride, window):
        rf = 1
        for count in range(layer):
                rf += np.power(2,count)*stride[count]*(window[count] - 1)
        return rf


def relevant_kernel_map(mask, layer, dim , stride , window , im_size):
        '''
        Function which finds which kernel of convolutional layer are detecting colored object within scene
        '''
        
        rf = window[0]
        rf = receptive_field(layer, stride, window)	
        relevance = cv2.GaussianBlur(mask.astype('float32'), (rf, rf),0)[::np.power(2,layer-1),::np.power(2,layer-1)]
        ind = (relevance.shape[0] - dim[layer-1])//2
        relevance = relevance[ind:dim[layer-1]+ind,ind:dim[layer-1]+ind]
        return (relevance > 0.33)*1



visualize_npy_im(dir_path + 'object150/object150_illu_B_norm_0.npy',no_patch = True, patch_nb = 5, save = True, add = '/home/alban/Dropbox/project_color_constancy/PAPER/exp_floating_object.png')

'''
visualize_npy_im(dir_path + 'object150/object150_illu_D_65_0.npy', save = True, add = '/home/alban/Dropbox/project_color_constancy/PAPER/exp_floating_object_D65.png')
visualize_npy_im(dir_path + 'object150/object150_illu_D_65_0_mask.npy', save = True, add = '/home/alban/Dropbox/project_color_constancy/PAPER/exp_floating_object_mask.png')



correction = np.array([0.85919557, 0.94538869, 1.19541574])
visualize_npy_im(dir_path + 'object150/object150_illu_B_norm_0.npy', correction = True, coef = correction, save = True, add = '/home/alban/Dropbox/project_color_constancy/PAPER/exp_floating_object_corrected.png')

I_mask = np.load(dir_path + 'object150/object150_illu_B_norm_0_mask.npy')
mask = np.mean(I_mask, axis = -1)
mask[mask > np.amax(mask)/255] = 1
mask[mask != 1] = 0 
visualize_npy_im(dir_path + 'object150/object150_illu_B_norm_0.npy', correction = True, coef = correction, masking = True, mask = mask, save = True, add = '/home/alban/Dropbox/project_color_constancy/PAPER/exp_floating_object_corrected_mask.png')



I_mask = np.load(dir_path + 'object32/object32_illu_D_65_1_mask.npy')
mask = np.mean(I_mask, axis = -1)
mask[mask > np.amax(mask)/255] = 1
mask[mask != 1] = 0 

relevant_conv1 = relevant_kernel_map(mask, 1, np.array([126,61,28]),np.array([1,1,1]), [5,3,3], 128)*1
relevant_conv2 = relevant_kernel_map(mask, 2, np.array([126,61,28]),np.array([1,1,1]), [5,3,3], 128)*1
relevant_conv3 = relevant_kernel_map(mask, 3, np.array([126,61,28]),np.array([1,1,1]), [5,3,3], 128)*1

plt.figure()
plt.imshow(np.stack((relevant_conv3,relevant_conv3,relevant_conv3), axis = 2)*255)
plt.show()
plt.close()

img_test = np.load(dir_path + 'object32_illu_D_65_1.npy')
'''