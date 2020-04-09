#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 14:04:32 2019

@author: alban
"""

import numpy as np
import matplotlib.pyplot as plt
import re

from computations_scripts import computations as comp

import sys
sys.path.append('/home/alban/Documents/pytorch/project_color_constancy/WCS_Xp/training_centered/')

from color_scripts import color_transforms as CT # scripts with a bunch of functions for color transformations
from error_measures_scripts import Error_measures as EM
from display_scripts import display as dis




# In[9]: INITIALIZATIONS

im_paths = list()
with open("im_paths_4il.txt") as f:
    for line in f:
       im_paths.append(line.split())

train_labels = [int(re.search('object(.*?)_', addr[0]).group(1)) for addr in im_paths]
#illu = [re.search('_(.*)_', addr[0]).group(1) for addr in im_paths]
#val_labels = [int(re.search('object(.*)_', addr).group(1)[:-2]) for addr in val_addrs]
#val_labels = [int(re.search('object(.*?)_', addr).group(1)) for addr in val_addrs]

mat_classic_cc = np.load('NPY_files/all_luminanes_4il.npy')
mat_classic_cc_mat = np.empty((330,4,10,5,3))
im_paths_mat = np.empty((330,4,10),dtype=np.object)
count = 0
for muns in range(len(im_paths_mat)):
	for illu in range(im_paths_mat.shape[1]):
		for exp in range(im_paths_mat.shape[2]):
			im_paths_mat[train_labels[count],illu,exp] = im_paths[count][0]
			mat_classic_cc_mat[train_labels[count], illu, exp] = mat_classic_cc[count]
			count +=1

mat_classic_cc_mat = ((mat_classic_cc_mat.T)/np.mean((mat_classic_cc_mat.T), axis = 0)).T


MUNSELLS_LMS = np.load('/home/alban/Documents/project_color_constancy/LMS_WCS_D65.npy')

wh  = np.array([17.95504939, 18.96455292, 20.36406225])
XYZ_MUNS = CT.LMS2XYZ(MUNSELLS_LMS).T

MUNS_LAB = CT.XYZ2Lab(XYZ_MUNS,white = wh)

RGB_WCS = np.load('/home/alban/Documents/project_color_constancy/RGB_WCS.npy')
dis.scatter_LAB(MUNS_LAB, RGB_WCS)


# In[9]: COMPUTATIONS

path2save = '/home/alban/mnt/DATA/project_color_constancy/classic_constancy/obj_segmented/munsell_%i_WCS_4illu.npy'
OBJ = comp.from_img_2_obj(im_paths_mat, path2save, mat_classic_cc_mat, (330,4,10,5,3),'/home/alban/mnt/awesome/alban/project_color_constancy/PYTORCH/WCS/train_centered/classic_color_constancy/corrected_4_illu/')

path2obj = '/home/alban/mnt/DATA/project_color_constancy/classic_constancy/obj_segmented/munsell_%i_WCS_4illu.npy'
CHROMA_OBJ_MEAN, CHROMA_OBJ_LUM = comp.Obj_chromaticity(path2obj, (330,4,10,5,3), MUNSELLS_LMS)

PREDICTION_XYZ = CT.LMS2XYZ(CHROMA_OBJ_MEAN).T
PREDICTION_LAB = CT.XYZ2Lab(PREDICTION_XYZ, white = wh)
np.save('PREDICTION_LAB_WCS_mean.npy',PREDICTION_LAB)
#PREDICTION_LAB = np.load('PREDICTION_LAB_lumnorm_mean.npy')

dis.scatter_LAB(PREDICTION_LAB[:,-1,0,0,:], RGB_WCS)
DELTAE = EM.DE_error_all(PREDICTION_LAB.T, MUNS_LAB.T)
np.mean(DELTAE,axis = (0,1,2))
np.median(DELTAE,axis = (0,2,-1))

PREDICTION_XYZ = CT.LMS2XYZ(CHROMA_OBJ_LUM).T
PREDICTION_LAB = CT.XYZ2Lab(PREDICTION_XYZ, white = wh)
np.save('PREDICTION_LAB_WCS_lum.npy',PREDICTION_LAB)

dis.scatter_LAB(PREDICTION_LAB[:,-1,0,0,:], RGB_WCS)
DELTAE = EM.DE_error_all(PREDICTION_LAB.T, MUNS_LAB.T)
np.mean(DELTAE,axis = (0,1,2))
np.median(DELTAE,axis = (0,2,-1))

np.median(DELTAE,axis = (0,2))
np.amin(DELTAE)

#idx_min = np.unravel_index(np.argmin(DELTAE),[330,4,10])


#image = np.load(im_paths_mat[idx_min[0],idx_min[1],idx_min[2]][:12]+'mnt/awesome/alban/'+im_paths_mat[idx_min[0],idx_min[1],idx_min[2]][12:])
#image_norm = image/np.amax(image)
#image_norm = np.expand_dims(image_norm,axis = 2)
#image_norm_ntr = np.repeat(image_norm,5,axis = 2)/mat_classic_cc_mat[idx_min[0],idx_min[1],idx_min[2],:,:]

#plt.imshow(image_norm[:,:,0,:])
#plt.show()
#plt.imshow((image_norm_ntr)[:,:,0,:])
#plt.show()


# In[9]: Grey patch in background


path2save_gp = '/home/alban/mnt/DATA/project_color_constancy/classic_constancy/obj_segmented/munsell_%i_GP.npy'
OBJ_GP = comp.from_img_2_obj(im_paths_mat, path2save_gp, mat_classic_cc_mat, (330,4,10,1,3))

path2obj = '/home/alban/mnt/DATA/project_color_constancy/classic_constancy/obj_segmented/munsell_%i_GP.npy'
CHROMA_OBJ_MEAN_GP, CHROMA_OBJ_LUM_GP = comp.Obj_chromaticity(path2obj, (330,4,10,1,3), MUNSELLS_LMS)


PREDICTION_XYZ_GP = CT.LMS2XYZ(CHROMA_OBJ_MEAN_GP).T
PREDICTION_LAB_GP = CT.XYZ2Lab(PREDICTION_XYZ_GP,white = wh)

np.save('PREDICTION_LAB_mean_GP.npy',PREDICTION_LAB_GP)
PREDICTION_LAB_GP = np.load('PREDICTION_LAB_mean_GP.npy')

dis.scatter_LAB(PREDICTION_LAB_GP[:,-1,5,0,:], RGB_WCS)
DELTAE_GP = EM.DE_error_all(PREDICTION_LAB_GP.T, MUNS_LAB.T)
np.mean(DELTAE_GP,axis = (0,2))
np.median(DELTAE_GP,axis = (0,2))

