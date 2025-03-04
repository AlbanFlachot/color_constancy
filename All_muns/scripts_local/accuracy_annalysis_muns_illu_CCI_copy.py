#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 16:52:28 2019

@author: alban
"""

# In[1]:



from __future__ import print_function, division

import matplotlib.pyplot as plt
import numpy as np
import pickle
#import MODELS as M

import scipy.io as sio


import sys
sys.path.append('/home/alban/Documents/pytorch/project_color_constancy/WCS_Xp/training_centered/')

from color_scripts import color_transforms as CT # scripts with a bunch of functions for color transformations
from error_measures_scripts import Error_measures as EM
from display_scripts import display as dis


# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


CC_dir_path = '/home/alban/Documents/pytorch/project_color_constancy/WCS_Xp/'


# In[2]:


#cudnn.benchmark = True

# In[9]: Compute accuracy as a measure of LAB distance





# In[9]: load models results


test_mode = ''

out_c1 = EM.softmax(np.load('/home/alban/mnt/DATA/project_color_constancy/data/training_centered/All_muns/out_c1'+test_mode+'.npy'))
out_c2 = EM.softmax(np.load('/home/alban/mnt/DATA/project_color_constancy/data/training_centered/All_muns/out_c2'+test_mode+'.npy'))
out_c3 = EM.softmax(np.load('/home/alban/mnt/DATA/project_color_constancy/data/training_centered/All_muns/out_c3'+test_mode+'.npy'))
out_f1 = EM.softmax(np.load('/home/alban/mnt/DATA/project_color_constancy/data/training_centered/All_muns/out_f1'+test_mode+'.npy'))
out_f2 = EM.softmax(np.load('/home/alban/mnt/DATA/project_color_constancy/data/training_centered/All_muns/out_f2'+test_mode+'.npy'))

pc1 = np.load('/home/alban/mnt/DATA/project_color_constancy/data/training_centered/All_muns/pc1'+test_mode+'.npy')
pc2 = np.load('/home/alban/mnt/DATA/project_color_constancy/data/training_centered/All_muns/pc2'+test_mode+'.npy')
pc3 = np.load('/home/alban/mnt/DATA/project_color_constancy/data/training_centered/All_muns/pc3'+test_mode+'.npy')
pf1 = np.load('/home/alban/mnt/DATA/project_color_constancy/data/training_centered/All_muns/pf1'+test_mode+'.npy')
pf2 = np.load('/home/alban/mnt/DATA/project_color_constancy/data/training_centered/All_muns/pf2'+test_mode+'.npy')

nb_mod = 10
nb_obj = 330

#
# In[9]: Load WCS coordinates
L = list()
with open(CC_dir_path + "WCS_indx.txt") as f:
    for line in f:
       L.append(line.split())

WCS_X = [ord(char[0][0].lower()) - 97 for char in L]
WCS_Y = [int(char[0][1:]) for char in L]

WCS_X1 = [9-x for x in WCS_X]



# In[9]: Accuracy per munsell


CHROMA = list()
with open(CC_dir_path + "WCS_chroma.txt") as f:
    for line in f:
       CHROMA.append(int(line.split()[0]))



WCS_MAT_CHROMA = np.zeros((10,41))
count = 0
for i in range(nb_obj):
    WCS_MAT_CHROMA[WCS_X[i],WCS_Y[i]] = float(CHROMA[count])
    count +=1
#F.display_munsells_inv(WCS_MAT_CHROMA,16)


mat_speakers = sio.loadmat(CC_dir_path +'matrix_WCS_speakers.mat')['WCS_speakers']


mat_consistency = sio.loadmat(CC_dir_path +'consistency_map4Alban.mat')['cmap']
#F.display_munsells_inv(mat_consistency,1)

mat_consistency2 = np.zeros(mat_consistency.shape)
for i in range(len(mat_consistency2)):
    mat_consistency2[i] = mat_consistency[7-i]
#F.display_munsells_inv(mat_consistency2,1)

general_mat_consistency = sio.loadmat(CC_dir_path +'general_consensusconsistency_map4Alban.mat')['cmap_general']
general_mat_consistency2 = np.zeros(general_mat_consistency.shape)
for i in range(len(general_mat_consistency2)):
    general_mat_consistency2[i] = general_mat_consistency[7-i]
#F.display_munsells(general_mat_consistency2[0:-3],1)



# In[9]: Compute accuracy as a measure of LAB distance

WCS_LAB_4 = np.moveaxis(np.load('LAB_MUNS_ABCD.npy'),0,-1)

WCS_muns = list()
with open(CC_dir_path +"WCS_muns.txt") as f:
    for line in f:
       WCS_muns.append(line.split()[0])


All_muns = list()
with open(CC_dir_path +"munsell_labels.txt") as f:
    for line in f:
       All_muns.append(line.split()[0])

list_WCS_labels = np.asarray([All_muns.index(WCS_muns[i]) for i in range(len(WCS_muns))])

LAB_WCS = np.load(CC_dir_path +'WCS_LAB.npy')
MUNSELL_LAB = np.load(CC_dir_path +'MUNSELL_LAB.npy')

PREDICTION_ERROR_C1 = EM.PREDICTION_LAB(pc1, MUNSELL_LAB, list_WCS_labels,data_training = 'all')
PREDICTION_ERROR_C2 = EM.PREDICTION_LAB(pc2, MUNSELL_LAB, list_WCS_labels,data_training = 'all')
PREDICTION_ERROR_C3 = EM.PREDICTION_LAB(pc3, MUNSELL_LAB, list_WCS_labels,data_training = 'all')
PREDICTION_ERROR_F1 = EM.PREDICTION_LAB(pf1, MUNSELL_LAB, list_WCS_labels,data_training = 'all')
PREDICTION_ERROR_F2 = EM.PREDICTION_LAB(pf2, MUNSELL_LAB, list_WCS_labels,data_training = 'all')

DIFF_C1 = EM.Weighted_DIFF_LAB_4(out_c1, WCS_LAB_4, list_WCS_labels, data_training = 'all')
DIFF_C2 = EM.Weighted_DIFF_LAB_4(out_c2, WCS_LAB_4, list_WCS_labels, data_training = 'all')
DIFF_C3 = EM.Weighted_DIFF_LAB_4(out_c3, WCS_LAB_4, list_WCS_labels, data_training = 'all')
DIFF_F1 = EM.Weighted_DIFF_LAB_4(out_f1, WCS_LAB_4, list_WCS_labels, data_training = 'all')
DIFF_F2 = EM.Weighted_DIFF_LAB_4(out_f2, WCS_LAB_4, list_WCS_labels, data_training = 'all')


PREDICTION_ERROR_C1_MAT = np.zeros((nb_mod,10,41))
PREDICTION_ERROR_C2_MAT = np.zeros((nb_mod,10,41))
PREDICTION_ERROR_C3_MAT = np.zeros((nb_mod,10,41))
PREDICTION_ERROR_F1_MAT = np.zeros((nb_mod,10,41))
PREDICTION_ERROR_F2_MAT = np.zeros((nb_mod,10,41))

PREDICTION_DE_F2 = WEIGHTED_PREDICTION_LAB(out_f2, MUNSELL_LAB, list_WCS_labels,data_training = 'all')
PREDICTION_DE_F2_MAT = np.zeros((10,41))

for i in range(nb_obj):
    PREDICTION_ERROR_C1_MAT[:,WCS_X[i],WCS_Y[i]] = np.mean(PREDICTION_ERROR_C1[:,i],axis = (-1,-2))
    PREDICTION_ERROR_C2_MAT[:,WCS_X[i],WCS_Y[i]] = np.mean(PREDICTION_ERROR_C2[:,i],axis = (-1,-2))
    PREDICTION_ERROR_C3_MAT[:,WCS_X[i],WCS_Y[i]] = np.mean(PREDICTION_ERROR_C3[:,i],axis = (-1,-2))
    PREDICTION_ERROR_F1_MAT[:,WCS_X[i],WCS_Y[i]] = np.mean(PREDICTION_ERROR_F1[:,i],axis = (-1,-2))
    PREDICTION_ERROR_F2_MAT[:,WCS_X[i],WCS_Y[i]] = np.mean(PREDICTION_ERROR_F2[:,i],axis = (-1,-2))
    PREDICTION_DE_F2_MAT[WCS_X[i],WCS_Y[i]] = np.median(PREDICTION_DE_F2[:,i],axis = (0,1,2))

#F.display_munsells_inv(np.mean(PREDICTION_ERROR_C1_MAT,axis = 0),np.amax(np.mean(PREDICTION_ERROR_C1_MAT,axis = 0)))
#F.display_munsells_inv(np.mean(PREDICTION_ERROR_C2_MAT,axis = 0),np.amax(np.mean(PREDICTION_ERROR_C2_MAT,axis = 0)))
#F.display_munsells_inv(np.mean(PREDICTION_ERROR_C3_MAT,axis = 0),np.amax(np.mean(PREDICTION_ERROR_C3_MAT,axis = 0)))
#F.display_munsells_inv(np.mean(PREDICTION_ERROR_F1_MAT,axis = 0),np.amax(np.median(PREDICTION_ERROR_F1_MAT,axis = 0)))
#F.display_munsells_inv(np.mean(PREDICTION_ERROR_F2_MAT,axis = 0),np.amax(np.mean(PREDICTION_ERROR_F2_MAT,axis = 0)))
#F.display_munsells_inv(PREDICTION_DE_F2_MAT,np.amax(PREDICTION_DE_F2_MAT))
np.amin(PREDICTION_DE_F2_MAT)
np.amax(PREDICTION_DE_F2_MAT)

np.mean(PREDICTION_ERROR_C1)
np.mean(PREDICTION_ERROR_C2)
np.mean(PREDICTION_ERROR_C3)
np.mean(PREDICTION_ERROR_F1)
np.mean(PREDICTION_ERROR_F2)

DeltaE_err = [np.mean(PREDICTION_ERROR_C1),np.mean(PREDICTION_ERROR_C2),np.mean(PREDICTION_ERROR_C3),np.mean(PREDICTION_ERROR_F1),np.mean(PREDICTION_ERROR_F2)]
DeltaE_bar = [np.std(np.mean(PREDICTION_ERROR_C1,axis = (1,2))),np.std(np.mean(PREDICTION_ERROR_C2,axis = (1,2))),np.std(np.mean(PREDICTION_ERROR_C3,axis = (1,2))),np.std(np.mean(PREDICTION_ERROR_F1,axis = (1,2))),np.std(np.mean(PREDICTION_ERROR_F2,axis = (1,2)))]

fig = plt.figure(figsize = (5,5))
ax1 = fig.add_subplot(111)
ax1.errorbar([1,2,3,4,5],DeltaE_err,yerr = DeltaE_bar,color = 'k',linewidth = 2)
ax1.set_xlabel('Layer',fontsize = 18)
ax1.set_xticks([1,2,3,4,5])
ax1.set_ylabel('Mean deltaE',fontsize = 18)
ax1.set_ylim([0,35])
#ax1.set_ylabel('Layer',fontsize = 15)
#fig.text(0.5,0.94,title,ha='center',fontsize = 18)
fig.tight_layout()
plt.show()


PREDICTION_Accuracy_C1 = np.zeros((nb_mod,nb_obj))
PREDICTION_Accuracy_C2 = np.zeros((nb_mod,nb_obj))
PREDICTION_Accuracy_C3 = np.zeros((nb_mod,nb_obj))
PREDICTION_Accuracy_F1 = np.zeros((nb_mod,nb_obj))
PREDICTION_Accuracy_F2 = np.zeros((nb_mod,nb_obj))

for m in range(nb_mod):
    for i in range(nb_obj):
        PREDICTION_Accuracy_C1[m,i] = EM.evaluation(pc1[m,i].flatten(),list_WCS_labels[i])
        PREDICTION_Accuracy_C2[m,i] = EM.evaluation(pc2[m,i].flatten(),list_WCS_labels[i])
        PREDICTION_Accuracy_C3[m,i] = EM.evaluation(pc3[m,i].flatten(),list_WCS_labels[i])
        PREDICTION_Accuracy_F1[m,i] = EM.evaluation(pf1[m,i].flatten(),list_WCS_labels[i])
        PREDICTION_Accuracy_F2[m,i] = EM.evaluation(pf2[m,i].flatten(),list_WCS_labels[i])

np.mean(PREDICTION_Accuracy_C1)
np.mean(PREDICTION_Accuracy_C2)
np.mean(PREDICTION_Accuracy_C3)
np.mean(PREDICTION_Accuracy_F1)
np.mean(PREDICTION_Accuracy_F2)

PREDICTION_Accuracy_F2_MAT = np.zeros((nb_mod,10,41))
for i in range(nb_obj):
    PREDICTION_Accuracy_F2_MAT[:,WCS_X[i],WCS_Y[i]] = PREDICTION_Accuracy_F2[:,i]

#F.display_munsells_inv(np.mean(PREDICTION_Accuracy_F2_MAT,axis = 0),np.amax(np.mean(PREDICTION_Accuracy_F2_MAT,axis = 0)))

Acc = [np.mean(PREDICTION_Accuracy_C1),np.mean(PREDICTION_Accuracy_C2),np.mean(PREDICTION_Accuracy_C3),np.mean(PREDICTION_Accuracy_F1),np.mean(PREDICTION_Accuracy_F2)]
Acc_bar = [np.std(np.mean(PREDICTION_Accuracy_C1,axis = (1))),np.std(np.mean(PREDICTION_Accuracy_C2,axis = (1))),np.std(np.mean(PREDICTION_Accuracy_C3,axis = (1))),np.std(np.mean(PREDICTION_Accuracy_F1,axis = (1))),np.std(np.mean(PREDICTION_Accuracy_F2,axis = (1)))]


fig = plt.figure(figsize = (5,5))
ax1 = fig.add_subplot(111)
ax1.errorbar([1,2,3,4,5],Acc,yerr = Acc_bar,color = 'k',linewidth = 2)
ax1.set_xlabel('Layer',fontsize = 18)
ax1.set_xticks([1,2,3,4,5])
ax1.set_ylabel('Accuracy',fontsize = 18)
ax1.set_ylim([0,100])
#ax1.set_ylabel('Layer',fontsize = 15)
#fig.text(0.5,0.94,title,ha='center',fontsize = 18)
fig.tight_layout()
plt.show()


# In[9]: COLOR CONSTANCY INDEX

WCS_LAB_all = np.load(CC_dir_path +'training_centered/All_muns/LAB_WCS_ABCD.npy')

Displacement_LAB = WCS_LAB_all.T - LAB_WCS.T

DE_WCS_all_illu = np.linalg.norm(Displacement_LAB, axis = (-1))


CCI_c1 = 1 - np.moveaxis(np.linalg.norm(DIFF_C1,axis = (-1)),-1,1)/DE_WCS_all_illu.T
CCI_c2 = 1 - (np.moveaxis(np.linalg.norm(DIFF_C2,axis = (-1)),-1,1)/DE_WCS_all_illu.T)
CCI_c3 = 1 - (np.moveaxis(np.linalg.norm(DIFF_C3,axis = (-1)),-1,1)/DE_WCS_all_illu.T)
CCI_f1 = 1 - (np.moveaxis(np.linalg.norm(DIFF_F1,axis = (-1)),-1,1)/DE_WCS_all_illu.T)
CCI_f2 = 1 - (np.moveaxis(np.linalg.norm(DIFF_F2,axis = (-1)),-1,1)/DE_WCS_all_illu.T)

CCI_f2_MAT = np.zeros((10,41))



for i in range(nb_obj):
    CCI_f2_MAT[WCS_X[i],WCS_Y[i]] = np.median(CCI_f2[:,:,i],axis = (0,-1,1))

#F.display_munsells_inv(CCI_f2_MAT,np.amax(CCI_f2_MAT))



CCI_f2_MAT_chrom = CCI_f2_MAT[1:-1,1:]
CCI_f2_MAT_achrom = CCI_f2_MAT[:,0]

with open(CC_dir_path +"XYZ_WCS.txt", "rb") as fp:   #Pickling
    XYZ_WCS = pickle.load(fp)

RGB_muns = [XYZ2sRGB(XYZ) for XYZ in XYZ_WCS]
WCS_MAT_RGB = np.zeros((10,41,3))
count = 0
for i in range(len(RGB_muns)):
    WCS_MAT_RGB[WCS_X[i],WCS_Y[i]] = RGB_muns[count]
    count +=1

WCS_MAT_sRGB = (WCS_MAT_RGB - np.amin(WCS_MAT_RGB))/(np.amax(WCS_MAT_RGB)-np.amin(WCS_MAT_RGB))
RGB_muns = (RGB_muns - np.amin(RGB_muns))/(np.amax(RGB_muns)-np.amin(RGB_muns))

plt.imshow((WCS_MAT_sRGB))
plt.show()


fig = plt.figure(figsize = (5,4))
ax1 = fig.add_subplot(111)
ax1.scatter(range(40)*8,CCI_f2_MAT_chrom.flatten(),color = WCS_MAT_sRGB[1:-1,1:].reshape(-1,3),marker = '.',s = 400,alpha=0.7)
ax1.set_xlabel('Hue',fontsize = 15)
ax1.set_xticks([1.5,5.5,9.5,13.5,18.5,22.5,26.5,30.5,34.5,38.5])
ax1.set_xticklabels(['R','YR','Y','GY','G','BG','B','PB','P','RP'])
plt.xticks(fontsize = 14)
plt.yticks(np.arange(0,1.1,0.25),fontsize = 14)
ax1.set_ylabel('Color Constancy Index',fontsize = 15)
fig.tight_layout()
#fig.savefig('/home/alban/Documents/pytorch/project_color_constancy/WCS_Xp/training_centered/POSTER_VSS_19/CCI_vs_hue.png', dpi=800)
plt.show()

fig = plt.figure(figsize = (5,4))
ax1 = fig.add_subplot(111)
ax1.scatter(range(9,1,-1)*40,CCI_f2_MAT_chrom.T.flatten(),color = np.moveaxis(WCS_MAT_sRGB[1:-1,1:],1,0).reshape(-1,3),marker = '.',s = 400,alpha=0.7)
ax1.scatter(range(10,0,-1),CCI_f2_MAT_achrom,color = WCS_MAT_sRGB[:,0],marker = '.',s = 400)
ax1.set_xlabel('Value',fontsize = 15)
ax1.set_xticks(range(0,11,2))
#ax1.set_xticklabels(['R','YR','Y','GY','G','BG','B','PB','P','RP'])
plt.xticks(fontsize = 14)
plt.yticks(np.arange(0,1.1,0.25),fontsize = 14)
ax1.set_ylabel('Color Constancy Index',fontsize = 15)
fig.tight_layout()
#fig.savefig('/home/alban/Documents/pytorch/project_color_constancy/WCS_Xp/training_centered/POSTER_VSS_19/CCI_vs_value.png', dpi=800)
plt.show()

WCS_MAT = np.zeros(WCS_MAT_sRGB.shape)
for i in range(len(WCS_MAT_sRGB)):
    WCS_MAT[i] = (WCS_MAT_sRGB[9-i])
WCS_MAT_achro = WCS_MAT[:,0].reshape(10,1,3)
WCS_MAT_chro = WCS_MAT[1:-1,1:]
# definitions for the axes
left, width = 0.05, 0.8
bottom, height = 0.1, 0.8
left_h = left + 0.8/40
bottom_h = 0

rect_achro = [left, bottom_h, 0.8/40, 1]
rect_chro = [left_h, bottom, width, height]


fig = plt.figure(1, figsize=(8, 3))

axchro = plt.axes(rect_chro)
axachro = plt.axes(rect_achro)


axchro.imshow(WCS_MAT_chro)
axachro.imshow(WCS_MAT_achro)

axachro.set_xticks([])
axchro.set_yticks([])


axachro.set_ylabel('Value',fontsize = 16)
axchro.set_xlabel('Hue',fontsize = 18)
#plt.setp(axchro.set_ylabel('Tuning |elevation| (deg)',fontsize = 25))
plt.setp(axchro.get_xticklabels(), fontsize=16)
#axchro.set_yticks(['1','3','5','7'])
plt.setp(axachro.get_yticklabels(), fontsize=16)
plt.show()

fig = plt.figure(figsize = (5,4))
ax1 = fig.add_subplot(111)
ax1.scatter(range(40)*8,CCI_f2_MAT_chrom.flatten(),color = WCS_MAT_sRGB[1:-1,1:].reshape(-1,3),marker = '.',s = 400)
ax1.set_xlabel('Hue',fontsize = 22)
ax1.set_xticks([1.5,5.5,9.5,13.5,18.5,22.5,26.5,30.5,34.5,38.5])
ax1.set_xticklabels(['R','YR','Y','GY','G','BG','B','PB','P','RP'])
plt.xticks(fontsize = 19)
plt.yticks(np.arange(0,1.1,0.5),fontsize = 19)
plt.xlim(-1,40)
ax1.set_ylabel('Median CCI',fontsize = 22)
fig.tight_layout()
fig.savefig(CC_dir_path +'training_centered/All_muns/CCI_vs_hue' + test_mode + '.png', dpi=800)
plt.show()

fig = plt.figure(figsize = (5,4))
ax1 = fig.add_subplot(111)
ax1.scatter(range(9,1,-1)*40,CCI_f2_MAT_chrom.T.flatten(),color = np.moveaxis(WCS_MAT_sRGB[1:-1,1:],1,0).reshape(-1,3),marker = '.',s = 400)
ax1.scatter(range(10,0,-1),CCI_f2_MAT_achrom,color = WCS_MAT_sRGB[:,0],marker = '.',s = 400)
ax1.set_xlabel('Value',fontsize = 22)
ax1.set_xticks(range(0,11,2))
#ax1.set_xticklabels(['R','YR','Y','GY','G','BG','B','PB','P','RP'])
plt.xticks(fontsize = 19)
plt.yticks(np.arange(0,1.1,0.5),fontsize = 19)
ax1.set_ylabel('Median CCI',fontsize = 22)
plt.xlim(0.5,10.5)
fig.tight_layout()
fig.savefig(CC_dir_path +'training_centered/All_muns/CCI_vs_value' + test_mode + '.png', dpi=800)
plt.show()

CCI_all = [np.median(CCI_c1),np.median(CCI_c2),np.median(CCI_c3),np.median(CCI_f1),np.median(CCI_f2)]
CCI_bar = [np.std(np.median(CCI_c1,axis = (1,2,3))),np.std(np.median(CCI_c2,axis = (1,2,3))),np.std(np.median(CCI_c3,axis = (1,2,3))),np.std(np.median(CCI_f1,axis = (1,2,3))),np.std(np.median(CCI_f2,axis = (1,2,3)))]

np.save('CCI_fost_weighted'+test_mode+'.npy',CCI_all)
np.save('CCI_fost_weighted'+test_mode+'_std.npy',CCI_bar)

CCI_all_4 = [np.mean(CCI_c1,axis= (0,1,2)),np.mean(CCI_c2,axis= (0,1,2)),np.mean(CCI_c3,axis= (0,1,2)),np.mean(CCI_f1,axis= (0,1,2)),np.mean(CCI_f2,axis= (0,1,2))]

CCI_all = np.load('CCI_fost_weighted.npy')
CCI_bar = np.load('CCI_fost_weighted_std.npy')
CCI_all_no_patch = np.load(CC_dir_path +'training_centered/All_muns/CCI_fost_weighted_no_patch.npy')
CCI_bar_no_patch = np.load(CC_dir_path +'training_centered/All_muns/CCI_fost_weighted_no_patch_std.npy')

CCI_all_no_back = np.load(CC_dir_path +'training_centered/WCS/finetuning/CCI_fost_weighted_no_back.npy')
CCI_bar_no_back = np.load(CC_dir_path +'training_centered/WCS/finetuning/CCI_fost_weighted_no_back_std.npy')
CCI_all_wrong_illu = np.load(CC_dir_path +'training_centered/All_muns/CCI_fost_weighted_wrong_illu.npy')
CCI_bar_wrong_illu = np.load(CC_dir_path +'training_centered/All_muns/CCI_fost_weighted_wrong_illu_std.npy')

CCI_all_no_patch[CCI_all_no_patch<0] = 0
CCI_all[CCI_all<0] = 0
CCI_all_no_back[CCI_all_no_back<0] = 0
CCI_all_wrong_illu[CCI_all_wrong_illu<0] = 0
fig = plt.figure(figsize = (8,8))
ax1 = fig.add_subplot(111)
ax1.errorbar([1,2,3,4,5],CCI_all, yerr = [0,0,0,0,0],color = 'k',linewidth = 6)
ax1.errorbar([1,2,3,4,5],CCI_all_no_patch, yerr = [0,0,0,0,0],color = [0.4,0.7,0.8],linewidth = 6)
ax1.errorbar([1,2,3,4,5],CCI_all_wrong_illu +[0.01,0.01,0.01,0.01,0.01],yerr = [0,0,0,0,0],color = [0.4,0.8,0.4],linewidth = 6)
ax1.errorbar([1,2,3,4,5], CCI_all_no_back +[0,0,0,0,0],yerr = [0,0,0,0,0],color = [0.7,0.8,0.4],linewidth = 6)
ax1.errorbar([1,2,3,4,5],  [-0.01,-0.01,-0.01,-0.01,-0.01],yerr = [0,0,0,0,0],color = [0.8,0.4,0.4],linewidth = 6)
ax1.set_xlabel('Readouts',fontsize = 20)
ax1.set_xticks([1,2,3,4,5])
ax1.set_xticklabels(['RC1','RC2','RC3','RF1','RF2'])
plt.xticks(fontsize = 17)
plt.yticks(fontsize = 17)
#plt.yticks(np.arange(-1,1.1,0.5),fontsize = 14)
ax1.set_ylabel('Median CCI',fontsize = 20)
fig.tight_layout()
fig.savefig('../POSTER_VSS_19/CCI_fost_readout.png', dpi=1200)
plt.show()

CCI = CCI_f2
CCI_2 = CCI_f2.copy()
CCI_2[CCI<0] = 0

hY = np.histogram(CCI_2[:,:,:,0].flatten(),bins = np.arange(-0.025,1.075,0.05))
hB = np.histogram(CCI_2[:,:,:,1].flatten(),bins = np.arange(-0.025,1.075,0.05))
hG = np.histogram(CCI_2[:,:,:,2].flatten(),bins = np.arange(-0.025,1.075,0.05))
hR = np.histogram(CCI_2[:,:,:,3].flatten(),bins = np.arange(-0.025,1.075,0.05))

fig = plt.figure(figsize = (7,6))
plt.plot(hY[1][:-1]+0.025, hY[0].astype(float)/np.sum(hY[0]), color= [0.8,0.7,0.3],lw = 5)
plt.plot(hB[1][:-1]+0.025, hB[0].astype(float)/np.sum(hB[0]), color= [0.3,0.4,0.8],lw = 5)
plt.plot(hG[1][:-1]+0.025, hG[0].astype(float)/np.sum(hG[0]), color= [0.4,0.8,0.4],lw = 5)
plt.plot(hR[1][:-1]+0.025, hR[0].astype(float)/np.sum(hR[0]), color= [0.7,0.3,0.7],lw = 5)
#plt.scatter(np.nanmedian(CCI[:,:,:,0]),0,color = 'orange',marker = '*',s = 200)
#plt.scatter(np.nanmedian(CCI[:,:,:,1]),0,color = 'b',marker = '*',s = 200)
#plt.scatter(np.nanmedian(CCI[:,:,:,2]),0,color = 'g',marker = '*',s = 200)
#plt.scatter(np.nanmedian(CCI[:,:,:,3]),0,color = 'r',marker = '*',s = 200)
plt.xlabel('CCI',fontsize = 22)
plt.ylabel('Frequency',fontsize = 22)
plt.xticks(np.arange(0,1.1,0.25),fontsize = 19)
plt.yticks(np.arange(0,0.4,0.1),fontsize = 19)
plt.xlim(-0.05,1)
#plt.ylim(0,1)
#plt.title('Munsell chips WCS',fontsize = 18)
fig.tight_layout
fig.savefig(CC_dir_path +'training_centered/All_muns/YBGR_distrib_CCI.png', dpi=800)
plt.show()

fig = plt.figure(figsize = (4,3))
ax1 = fig.add_subplot(111)
ax1.bar(np.arange(0.6,4.2,1),[np.nanmedian(CCI[:,:,:,0]), np.nanmedian(CCI[:,:,:,1]), np.nanmedian(CCI[:,:,:,2]), np.nanmedian(CCI[:,:,:,3]) ],color = [[0.8,0.7,0.3],[0.3,0.4,0.8],[0.4,0.8,0.4],[0.7,0.3,0.7]])
#ax1.errorbar([1,2,3,4,5],CCI_all_no_patch,yerr = CCI_bar_no_patch,color = [0.4,0.7,0.8],linewidth = 6)
#ax1.set_xlabel('Readouts',fontsize = 15)
ax1.set_xticks([1,2,3,4])
ax1.set_xticklabels(['Y','B','G','R'],rotation=0,fontsize = 22)
plt.xticks(fontsize = 19)
#plt.yticks(fontsize = 14)
plt.yticks(np.arange(0,1.1,0.25),fontsize = 19)
ax1.set_ylabel('Median CCI',fontsize = 22)
plt.ylim(0.5,1)
fig.tight_layout()
fig.savefig(CC_dir_path +'training_centered/All_muns/median_YBGR.png', dpi=800)
plt.show()

# In[9]: 3D CCI

PREDICTION_3D_error = DIFF_F2
PREDICTION_3D_error = np.moveaxis(PREDICTION_3D_error,-2,1)

CCI_L = 1 - np.absolute(PREDICTION_3D_error[:,:,:,:,0]/Displacement_LAB[:,:,0].T)
CCI_a = 1 - np.absolute(PREDICTION_3D_error[:,:,:,2:,1]/Displacement_LAB[2:,:,1].T)
CCI_b = 1 - np.absolute(PREDICTION_3D_error[:,:,:,:2,2]/Displacement_LAB[:2,:,2].T)



fig = plt.figure(figsize = (9,4))
ax1 = fig.add_subplot(131)
p1 = CCI_L
perc1 = np.nanpercentile(p1,np.arange(0,100,0.1))
ax1.plot(np.arange(0,10,0.1),perc1[:100], color= 'black')
ax1.hlines(np.nanmean(p1),xmin = 0, xmax = 10,color = 'r')

ax2 = fig.add_subplot(132,sharex=ax1)
p2 = CCI_a
perc2 = np.nanpercentile(p2,np.arange(0,100,0.1))
ax2.plot(np.arange(0,10,0.1),perc2[:100], color= 'black')
ax2.hlines(np.nanmean(p2),xmin = 0, xmax = 10,color = 'r')

ax3 = fig.add_subplot(133,sharex=ax1)
p3 = CCI_b
perc3 = np.nanpercentile(p3,np.arange(0,100,0.1))
ax3.plot(np.arange(0,10,0.1),perc3[:100], color= 'black')
ax3.hlines(np.nanmean(p3),xmin = 0, xmax = 10,color = 'r')

ax1.set_title('L*', fontsize = 15)
ax2.set_title('a*', fontsize = 15)
ax3.set_title('b*', fontsize = 15)

ax1.set_ylabel('CCI', fontsize = 15)
ax2.set_xlabel('Percentile', fontsize = 15)

fig.tight_layout
plt.show()


### Figure histogram CCIs.

fig = plt.figure(figsize = (9,4))
ax1 = fig.add_subplot(131)
p1 = CCI_L
h = ax1.hist((p1).flatten(), bins = np.arange(-1, 1.11, 0.1), color= 'black')
ax1.vlines(np.nanmean(p1),ymin = 0, ymax = 80000,color = 'r')
#ax1.vlines(np.nanpercentile(1-p1,25),ymin = 0, ymax = 80000,color = 'orange')
ax1.vlines(np.nanpercentile(p1,10),ymin = 0, ymax = 80000,color = 'g')
ax2 = fig.add_subplot(132,sharex=ax1)
p2 =  CCI_a
ax2.hist((p2).flatten(), bins = np.arange(-1, 1.11, 0.1), color= 'k')
ax2.vlines(np.nanmean((p2)),ymin = 0, ymax = 80000,color = 'r')
#ax2.vlines(np.nanpercentile((1-p2),25),ymin = 0, ymax = 80000,color = 'orange')
ax2.vlines(np.nanpercentile((p2),10),ymin = 0, ymax = 80000,color = 'g')
ax3 = fig.add_subplot(133,sharex=ax1)
p3 = CCI_b
ax3.hist((p3).flatten(), bins = np.arange(-1, 1.11, 0.1), color= 'k')
ax3.vlines(np.nanmean((p3)),ymin = 0, ymax = 80000,color = 'r')
#ax3.vlines(np.nanpercentile((1-p3),25),ymin = 0, ymax = 80000,color = 'orange')
ax3.vlines(np.nanpercentile((p3),10),ymin = 0, ymax = 80000,color = 'g')
ax2.set_yticks([])
ax3.set_yticks([])
ax1.set_title('L*', fontsize = 15)
ax2.set_title('a*', fontsize = 15)
ax3.set_title('b*', fontsize = 15)
ax1.set_xlim(-1,1.1)
ax2.set_xlim(-1,1.1)
ax3.set_xlim(-1,1.1)

ax1.set_ylabel('Count', fontsize = 15)
ax2.set_xlabel('CCI', fontsize = 15)

fig.tight_layout
plt.show()

### Figure histogram CCIs.

fig = plt.figure(figsize = (9,4))
ax2 = fig.add_subplot(121)
p2 =  CCI_a
h = ax2.hist((p2).flatten(), bins = np.arange(-1, 1.11, 0.1), color= 'k')
ax2.vlines(np.nanmedian((p2)),ymin = 0, ymax = 35000,color = 'r')
#ax2.vlines(np.nanpercentile((1-p2),25),ymin = 0, ymax = 80000,color = 'orange')
#ax2.vlines(np.nanpercentile((p2),10),ymin = 0, ymax = 53000,color = 'g')
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
ax3 = fig.add_subplot(122,sharex=ax2)
p3 = CCI_b
ax3.hist((p3).flatten(), bins = np.arange(-1, 1.11, 0.1), color= 'k')
ax3.vlines(np.nanmedian((p3)),ymin = 0, ymax = 35000,color = 'r')
#ax3.vlines(np.nanpercentile((1-p3),25),ymin = 0, ymax = 80000,color = 'orange')
#ax3.vlines(np.nanpercentile((p3),10),ymin = 0, ymax = 53000,color = 'g')
ax3.set_yticks([])
ax2.set_title('a*', fontsize = 15)
ax3.set_title('b*', fontsize = 15)
ax2.set_xlim(0,1)
ax3.set_xlim(0,1)
ax2.set_ylabel('Count', fontsize = 15)
ax2.set_xlabel('CCI', fontsize = 15)
ax3.set_xlabel('CCI', fontsize = 15)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
fig.savefig('../POSTER_VSS_19/ab_distrib_CCI.png', dpi=800)
fig.tight_layout
plt.show()



fig = plt.figure(figsize = (4,4))
ax1 = fig.add_subplot(111)
ax1.bar([1,2,3,4,5],[Acc, Acc_no_patch, Acc_f_wrong_illu, Acc_no_back, 2.1],color = ['k',[0.4,0.7,0.8],[0.4,0.8,0.4],[0.7,0.8,0.4],[0.8,0.4,0.4],'grey'],linewidth = 6)
#ax1.errorbar([1,2,3,4,5],CCI_all_no_patch,yerr = CCI_bar_no_patch,color = [0.4,0.7,0.8],linewidth = 6)
#ax1.set_xlabel('Readouts',fontsize = 15)
ax1.set_xticks([1,2,3,4,5])
ax1.set_xticklabels([])
plt.xticks(fontsize = 21)
#plt.yticks(fontsize = 14)
plt.yticks(np.arange(0,105,25),fontsize = 21)
ax1.set_ylabel('Accuracy',fontsize = 25)
fig.tight_layout()
fig.savefig('../POSTER_VSS_19/Accuracy.png', dpi=800)
plt.show()
