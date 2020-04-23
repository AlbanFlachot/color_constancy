#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 14:04:32 2019

@author: alban
"""


from __future__ import print_function, division

import matplotlib.pyplot as plt
import numpy as np
import re

import sys
sys.path.append('../../')

from color_scripts import color_transforms as CT # scripts with a bunch of functions for color transformations
from display_scripts import display as dis
from utils_scripts import algos
from utils_scripts import computations_classic_CC as comp

import warnings

warnings.filterwarnings("ignore")


# In[9]: path to files

txt_dir_path = '../../txt_files/'
npy_dir_path = '../../npy_files/'
pickles_dir_path = '../pickles/'
figures_dir_path = '../figures/'

# In[9]: INITIALIZATIONS

im_paths = list()
with open(txt_dir_path + "im_paths_4il.txt") as f:
    for line in f:
       im_paths.append(line.split())

im_paths = list()
with open(txt_dir_path + "im_paths_4il.txt") as f:
    for line in f:
       im_paths.append(line.split())

train_labels = [int(re.search('object(.*?)_', addr[0]).group(1)) for addr in im_paths]


mat_classic_cc = np.load(npy_dir_path + 'all_luminanes_4il.npy')
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


MUNSELLS_LMS = np.load(npy_dir_path +'LMS_WCS_D65.npy')

#extract_max_pix = np.array([0,0,0])
#for i in im_paths:
#    extract_max_pix = np.amax((extract_max_pix, np.amax(np.load(i[0][:12]+'mnt/awesome/alban/'+i[0][12:]), axis = (0,1))), axis = 0)

#extract_min_pix = np.array([3,3,3])
#for i in im_paths:
#    extract_min_pix = np.amin((extract_min_pix, np.amin(np.load(i[0][:12]+'mnt/awesome/alban/'+i[0][12:]), axis = (0,1))), axis = 0)


max_pxl = np.array([22.2250821 , 25.19385033, 25.49259802])
min_pxl = np.array([0.00260734, 0.00236998, 0.00086242])

wh  = np.array([17.95504939, 18.96455292, 20.36406225])
XYZ_MUNS = CT.LMS2XYZ(MUNSELLS_LMS).T

MUNS_LAB = CT.XYZ2Lab(XYZ_MUNS,white = wh)

RGB_WCS = np.load(npy_dir_path + 'RGB_WCS.npy')
dis.scatter_LAB(MUNS_LAB, RGB_WCS)


# In[9]: COMPUTATIONS

def scaling(X, Y):
    '''
    Function to scale X according to Y
    '''

    #maxX = np.amax(X, axis = ax)
    #minX = np.amin(X, axis = ax)
    maxX = X[0].max()
    minX = X[-1].min()

    maxY = Y[0].max()
    minY = Y[-1].min()

    #maxY = np.amax(Y, axis = ax)
    #minY = np.amin(Y, axis = ax)

    return (X - minX)*((maxY-minY)/(maxX-minX)) + minY

def computations_chromaticities(im_paths_mat, illus, shape, MUNSELLS_LMS, max_pxl,algo = 0):
    CHROMA_OBJ_MEAN, CHROMA_OBJ_LUM, CHROMA_OBJ_MEDIAN = predicted_chromaticity(im_paths_mat,illus, shape, MUNSELLS_LMS, max_pxl)
    #CHROMA_OBJ_MEAN_noCC = (CHROMA_OBJ_MEAN_noCC + np.amin(MUNSELLS_LMS))*np.amax(MUNSELLS_LMS)/(np.amax(CHROMA_OBJ_MEAN_noCC)- np.amin(MUNSELLS_LMS))
    #CHROMA_OBJ_MEDIAN_noCC = (CHROMA_OBJ_MEDIAN_noCC + np.amin(MUNSELLS_LMS))*np.amax(MUNSELLS_LMS)/(np.amax(CHROMA_OBJ_MEDIAN_noCC)- np.amin(MUNSELLS_LMS))
    #CHROMA_OBJ_LUM_noCC = (CHROMA_OBJ_LUM_noCC + np.amin(MUNSELLS_LMS))*np.amax(MUNSELLS_LMS)/(np.amax(CHROMA_OBJ_LUM_noCC)- np.amin(MUNSELLS_LMS))

    chroma_mean = scaling(np.mean(CHROMA_OBJ_MEAN, axis = (2)), MUNSELLS_LMS)

    #import pdb; pdb.set_trace()
    PREDICTION_XYZ = CT.LMS2XYZ(chroma_mean).T
    PREDICTION_LAB = CT.XYZ2Lab(PREDICTION_XYZ,white = wh)
    #PREDICTION_LAB = (PREDICTION_LAB.T - np.array([ 1.72402202, -0.27628075,  0.65468516])).T

    for i in range(PREDICTION_LAB.shape[2]):
        dis.scatter_LAB(PREDICTION_LAB[:,algo,i], RGB_WCS)
    DELTAE = comp.DE_error_all(PREDICTION_LAB[:,algo].T, MUNS_LAB.T)
    print(np.mean(DELTAE,axis = (0)))
    print(np.median(DELTAE,axis = (0)))
    return {'mean_LMS': CHROMA_OBJ_MEAN, 'median_LMS': CHROMA_OBJ_MEDIAN, 'bright_LMS': CHROMA_OBJ_LUM, 'XYZ': PREDICTION_XYZ, 'LAB': PREDICTION_LAB}


#path2obj = '/home/alban/mnt/DATA/project_color_constancy/classic_constancy/obj_segmented/munsell_%i_WCS_4illu.npy'
#CHROMA_OBJ_MEAN, CHROMA_OBJ_LUM, CHROMA_OBJ_MEDIAN = comp.Obj_chromaticity(path2obj, (330,4,10,5,3), MUNSELLS_LMS)

#computations_chromaticities(im_paths_mat, mat_classic_cc_mat, (330,4,10,5,3), MUNSELLS_LMS, max_pxl)

#PREDICTION_XYZ = CT.LMS2XYZ(CHROMA_OBJ_MEDIAN).T
#PREDICTION_LAB = CT.XYZ2Lab(PREDICTION_XYZ, white = wh)
#np.save('PREDICTION_LAB_WCS_mean.npy',PREDICTION_LAB)
#PREDICTION_LAB = np.load('PREDICTION_LAB_lumnorm_mean.npy')

#dis.scatter_LAB(PREDICTION_LAB[:,-1,0,0,:], RGB_WCS)
#DELTAE = EM.DE_error_all(PREDICTION_LAB.T, MUNS_LAB.T)
#np.mean(DELTAE,axis = (0,1,2))
#np.median(DELTAE,axis = (0,2,-1))

#PREDICTION_XYZ = CT.LMS2XYZ(CHROMA_OBJ_LUM).T
#PREDICTION_LAB = CT.XYZ2Lab(PREDICTION_XYZ, white = wh)
#np.save('PREDICTION_LAB_WCS_lum.npy',PREDICTION_LAB)

#dis.scatter_LAB(PREDICTION_LAB[:,-1,0,0,:], RGB_WCS)
#DELTAE = EM.DE_error_all(PREDICTION_LAB.T, MUNS_LAB.T)
#np.mean(DELTAE,axis = (0,1,2))
#np.median(DELTAE,axis = (0,2,-1))

#np.median(DELTAE,axis = (0,2))
#np.amin(DELTAE)





# In[9]: Controle: perfect estimation illu with von kries

LMS_4illu = np.array([[0.8608155 , 0.87445185, 0.77403174],[0.78124027, 0.84468902, 1.04376177], [0.87937024, 0.95460385, 0.97006115],[0.77854056, 0.79059459, 0.86517277]])

DICT_vK = computations_chromaticities(im_paths_mat, LMS_4illu, (330,4,10,1,3), MUNSELLS_LMS, max_pxl)

np.save('DICT_vK.npy', DICT_vK)

np.median(np.absolute(DICT_vK['LAB'][:,0,1] - MUNS_LAB), axis = (1))
np.mean(np.mean(DICT_vK['LAB'],axis =(1,2)) - MUNS_LAB, axis = -1)
np.mean(np.median(DICT_vK['LAB'],axis =(1,2)) - MUNS_LAB, axis = -1)

# In[9]: Controle: No color constancy

DICT_noCC = computations_chromaticities(im_paths_mat, np.array([1,1,1]), (330,4,10,1,3), MUNSELLS_LMS, max_pxl)
np.mean(DICT_noCC['LAB'][:,0,0] - MUNS_LAB, axis = (1))

print('Error without illuminant correction:')
print(np.median(np.linalg.norm(DICT_noCC['LAB'] - DICT_vK['LAB'], axis = 0)))

error_noCC = np.median(np.linalg.norm(DICT_noCC['LAB'] - DICT_vK['LAB'], axis = 0))

np.save('DICT_noCC.npy', DICT_noCC)

# In[9]: classic color constancy


DICT_CCC = computations_chromaticities(im_paths_mat, mat_classic_cc_mat, (330,4,10,5,3), MUNSELLS_LMS, max_pxl)

np.save('DICT_CCC.npy',DICT_CCC)

np.mean(DICT_noCC['LAB'][:,0,0] - MUNS_LAB, axis = (1))

print('Error grey world:')
print(np.median(np.linalg.norm(DICT_CCC['LAB'][:,0,:] - DICT_vK['LAB'][:,0], axis = 0)))
error_GW = np.median(np.linalg.norm(DICT_CCC['LAB'][:,0,:] - DICT_vK['LAB'][:,0], axis = 0))

print('Error white patch:')
print(np.median(np.linalg.norm(DICT_CCC['LAB'][:,1] - DICT_vK['LAB'][:,0], axis = 0)))
error_WP = np.median(np.linalg.norm(DICT_CCC['LAB'][:,1] - DICT_vK['LAB'][:,0], axis = 0))

print('Error edge1:')
print(np.median(np.linalg.norm(DICT_CCC['LAB'][:,2] - DICT_vK['LAB'][:,0], axis = 0)))
error_edge1 = np.median(np.linalg.norm(DICT_CCC['LAB'][:,2] - DICT_vK['LAB'][:,0], axis = 0))


print('Error edge2:')
print(np.median(np.linalg.norm(DICT_CCC['LAB'][:,3] - DICT_vK['LAB'][:,0], axis = 0)))
error_edge2 = np.median(np.linalg.norm(DICT_CCC['LAB'][:,3] - DICT_vK['LAB'][:,0], axis = 0))

print('Error contrast:')
print(np.median(np.linalg.norm(DICT_CCC['LAB'][:,4] - DICT_vK['LAB'][:,0], axis = 0)))
error_contrast = np.median(np.linalg.norm(DICT_CCC['LAB'][:,4] - DICT_vK['LAB'][:,0], axis = 0))



fig = plt.figure(figsize = (6,6))
ax1 = fig.add_subplot(111)
ax1.bar([1,2,3,4,5,6],[0, error_GW, error_WP, error_edge2, error_contrast, error_noCC], color = [[0.5,0.5,0.5],[0.3,0.3,0.3],[0.3,0.3,0.3],[0.3,0.3,0.3],[0.3,0.3,0.3],'k'],linewidth = 4)
ax1.set_xticks([1,2,3,4,5,6])
ax1.set_xticklabels([])
plt.xticks(fontsize = 21)
#plt.yticks(fontsize = 14)
plt.yticks(np.arange(0,20,5),fontsize = 21)
ax1.set_ylabel('$\Delta$E',fontsize = 25)
ax1.set_xticklabels(['perfect\nCC','GW','WP','edge 2','contrast', 'no CC'], rotation = 45)
fig.tight_layout()
#fig.savefig(figures_dir_path + 'Accuracy.png', dpi=400)
plt.show()

#path2save_gp = '/home/alban/mnt/DATA/project_color_constancy/classic_constancy/obj_segmented/munsell_%i_GP.npy'
#OBJ_GP = comp.from_img_2_obj(im_paths_mat, path2save_gp, mat_classic_cc_mat, (330,4,10,1,3),'/home/alban/mnt/awesome/alban/project_color_constancy/PYTORCH/WCS/train_centered/classic_color_constancy/corrected_4_illu/')

#path2obj = '/home/alban/mnt/DATA/project_color_constancy/classic_constancy/obj_segmented/munsell_%i_GP.npy'
#CHROMA_OBJ_MEAN_GP, CHROMA_OBJ_LUM_GP, CHROMA_OBJ_MEDIAN_GP = comp.Obj_chromaticity(path2obj, (330,4,10,1,3), MUNSELLS_LMS, max_pxl)

#computations_chromaticities(im_paths_mat, mat_classic_cc_mat, (330,4,10,1,3), MUNSELLS_LMS, max_pxl)

#PREDICTION_XYZ_GP = CT.LMS2XYZ(CHROMA_OBJ_MEAN_GP).T
#PREDICTION_LAB_GP = CT.XYZ2Lab(PREDICTION_XYZ_GP,white = wh)

#np.save('PREDICTION_LAB_mean_GP.npy',PREDICTION_LAB_GP)
#PREDICTION_LAB_GP = np.load('PREDICTION_LAB_mean_GP.npy')

#dis.scatter_LAB(PREDICTION_LAB_GP[:,-1,5,0,:], RGB_WCS)
#DELTAE_GP = EM.DE_error_all(PREDICTION_LAB_GP.T, MUNS_LAB.T)
#np.mean(DELTAE_GP,axis = (0,2))
#np.median(DELTAE_GP,axis = (0,2))

# In[9]: Controle: under D64

import glob
path_images_D64 = glob.glob('/home/alban/mnt/awesome/alban/project_color_constancy/TRAINING/DATA/PNG/masks_centered_arash/object*/object*_illu_D_64.npy')

labels_D64 = [int(re.search('object(.*?)/', addr).group(1)) for addr in path_images_D64]

D64_paths_mat = np.empty((1600,1,1),dtype=np.object)
count = 0
for muns in range(len(D64_paths_mat)):
	for illu in range(D64_paths_mat.shape[1]):
		for exp in range(D64_paths_mat.shape[2]):
			D64_paths_mat[labels_D64[count], illu, exp] = path_images_D64[count][:12] + path_images_D64[count][30:]
			count +=1

list_WCS_labels = algos.compute_WCS_Munsells_categories()

D64_WCS_path_mat = D64_paths_mat[list_WCS_labels]

DICT_D64 = computations_chromaticities(D64_WCS_path_mat, np.array([1,1,1]), (330,1,1,1,3), MUNSELLS_LMS, max_pxl,algo = 0)

np.mean(np.absolute(DICT_D64['LAB'][:,0,0] - MUNS_LAB), axis = (1))

im_path = '/home/alban/mnt/awesome/alban/project_color_constancy/TRAINING/DATA/PNG/masks_centered_arash/object100/object100_illu_D_64.npy'
illu = np.array([1,1,1])
chroma = MUNSELLS_LMS[100]
#white_balance_image(im_path, max_pxl, illu, chroma)