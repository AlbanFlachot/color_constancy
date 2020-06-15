#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 26 16:04:54 2020

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

warnings.filterwarnings("ignore")


# In[9]: path to files

txt_dir_path = '../../txt_files/'
npy_dir_path = '../../npy_files/'
pickles_dir_path = '../pickles/'
figures_dir_path = '../figures/'

# In[9]: load mnodels results

MUNS_COORDINATES = np.load(npy_dir_path + 'MUNSELL_COORDINATES.npy')
MUNS_COORDINATES_VHC = np.load(npy_dir_path + 'MUNSELL_coor.npy')

Hue_arr = np.arange(0,2*np.pi,2*np.pi/80)

MUNS_COORDINATES_VXY = np.zeros(MUNS_COORDINATES_VHC.shape)

for i in range(MUNS_COORDINATES_VHC.__len__()):
    MUNS_COORDINATES_VXY[i][0] = MUNS_COORDINATES_VHC[i][0]/3.6
    MUNS_COORDINATES_VXY[i][1] = MUNS_COORDINATES_VHC[i][2]*np.cos(Hue_arr[MUNS_COORDINATES_VHC[i][1].astype(int)])
    MUNS_COORDINATES_VXY[i][2] = MUNS_COORDINATES_VHC[i][2]*np.sin(Hue_arr[MUNS_COORDINATES_VHC[i][1].astype(int)])

RGB_muns = np.load(npy_dir_path +'RGB_1600_muns.npy')



def plot_Lab_muns(Array,RGB_muns):## Plot of munsells in the CIELab space following the XYZ2Lab conversion
	fig = plt.figure(figsize = (6,6))
	ax = fig.add_subplot(111)
	ax.scatter(Array[:,1], Array[:,2],marker = 'o',color=RGB_muns,s = 60)
	ax.set_xlim(-8,8)
	ax.set_ylim(-8,8)
	ax.set_xlabel('Red-Green',fontsize = 20)
	ax.set_ylabel('Blue-Yellow',fontsize = 20)
	plt.xticks(range(-8,9, 4),fontsize = 16)
	plt.yticks(range(-8,9, 4),fontsize = 16)


	fig = plt.figure(figsize = (6,6))
	ax = fig.add_subplot(111)
	ax.scatter(Array[:,1], Array[:,0],marker = 'o',color=RGB_muns,s = 60)
	#ax.set_title('CIELab values under %s'%ill,fontsize = 18)
	ax.set_xlabel('Red-Green',fontsize = 20)
	ax.set_ylabel('Value',fontsize = 20)
	ax.set_xlim(-8,8)
	ax.set_ylim(0,10)
	plt.xticks(range(-8,9, 4),fontsize = 16)
	plt.yticks(range(0,11, 5),fontsize = 16)
	plt.show()
	plt.close()


plot_Lab_muns(MUNS_COORDINATES_VXY, RGB_muns)

MUNS_COORDINATES_VHC[:,0] = MUNS_COORDINATES_VHC[:,0]/3.6


np.save(npy_dir_path + 'MUNS_COORDINATES_VXY.npy',MUNS_COORDINATES_VXY)
np.save(npy_dir_path + 'MUNS_COORDINATES_VHC.npy', MUNS_COORDINATES_VHC)

