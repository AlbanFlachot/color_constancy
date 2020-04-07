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
figures_dir_path = '../figures/'

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
    return pickle.load(f)


# In[9]: LOAD ERROR MEASURES

conditions = ['normal','no_patch', 'wrong_illu', 'no_back', 'D65']


layers = ['fc2','fc1', 'c3', 'c2', 'c1']

ERRORS = {}

path = pickles_dir_path

for layer in layers:
    ERRORS[layer] = {}
    for condition in conditions[:-1]:
        ERRORS[layer][condition] = load_pickle(path + 'Errors_Original_CC_4illu_WCS_%s_%s.pickle'%(layer, condition))
    ERRORS[layer][conditions[-1]] = load_pickle(path + 'Errors_Original_D65_4illu_WCS_%s_normal.pickle'%(layer))

if not os.path.exists('../figures'):
    os.mkdir('../figures')

#_____________________________________________________________________________________________________________________________

### ANALYZE OUTPUT LAST LAYER NORMAL CONDITION
#_____________________________________________________________________________________________________________________________

CHROMA = list()
with open(txt_dir_path +"WCS_chroma.txt") as f:
    for line in f:
       CHROMA.append(int(line.split()[0]))


WCS_MAT_CHROMA = from330to8x40(CHROMA)
dis.display_munsells_inv(WCS_MAT_CHROMA,16)


mat_speakers = sio.loadmat(npy_dir_path +'matrix_WCS_speakers.mat')['WCS_speakers']


mat_consistency = sio.loadmat(npy_dir_path +'consistency_map4Alban.mat')['cmap']
#dis.display_munsells_inv(mat_consistency,1)

mat_consistency2 = np.zeros(mat_consistency.shape)
for i in range(len(mat_consistency2)):
    mat_consistency2[i] = mat_consistency[7-i]
#dis.display_munsells_inv(mat_consistency2,1)

general_mat_consistency = sio.loadmat(npy_dir_path +'general_consensusconsistency_map4Alban.mat')['cmap_general']
general_mat_consistency2 = np.zeros(general_mat_consistency.shape)
for i in range(len(general_mat_consistency2)):
    general_mat_consistency2[i] = general_mat_consistency[7-i]
dis.display_munsells(general_mat_consistency2[0:-3],1)


with open(txt_dir_path +"XYZ_WCS.txt", "rb") as fp:   #Pickling
    XYZ_WCS = pickle.load(fp)

RGB_muns = [CT.XYZ2sRGB(XYZ) for XYZ in XYZ_WCS]
WCS_MAT_RGB =from330to8x40(RGB_muns)

WCS_MAT_sRGB = (WCS_MAT_RGB - np.amin(WCS_MAT_RGB))/(np.amax(WCS_MAT_RGB)-np.amin(WCS_MAT_RGB))
RGB_muns = (RGB_muns - np.amin(RGB_muns))/(np.amax(RGB_muns)-np.amin(RGB_muns))

dis.display_munsells_inv(WCS_MAT_sRGB,1)


nb_mod = ERRORS[layers[0]][conditions[0]]['DE'].shape[0]
nb_obj = ERRORS[layers[0]][conditions[0]]['DE'].shape[1]



# In[9]: Compute accuracy as a measure of LAB delta E

# CIELab coordinates of WCS muns under YBGR illuminants
WCS_LAB_4 = np.load(npy_dir_path +'LAB_WCS_ABCD.npy')

MUNS_LAB_4 = np.moveaxis(np.load(npy_dir_path +'LAB_MUNS_ABCD.npy'),0,-1)

MUNSELL_LAB = np.load(npy_dir_path +'MUNSELL_LAB.npy')



DE_illu = np.mean(ERRORS[layers[0]][conditions[0]]['DE'],axis = (0,1,-1))
DE_muns = np.mean(ERRORS[layers[0]][conditions[0]]['DE'],axis = (0,-2,-1))
print(DE_illu)

DE_MAT = from330to8x40(np.mean(ERRORS[layers[0]][conditions[0]]['DE'],axis = (-1,-2)))


Accu_MAT = from330to8x40(ERRORS[layers[0]][conditions[0]]['Accu'])

dis.display_munsells_inv(np.mean(Accu_MAT,axis = (0,-1)), np.amax(np.mean(Accu_MAT,axis = (0,-1))))
dis.display_munsells_inv(np.mean(DE_MAT,axis = 0), np.amax(np.mean(DE_MAT,axis = 0)))
dis.display_munsells_inv(DE_MAT[0], np.amax(DE_MAT[0]))


# Comparison with consistency
dis.display_munsells_inv(np.std(DE_MAT,axis = 0), np.amax(np.std(DE_MAT,axis = 0)))
np.corrcoef(np.std(DE_MAT,axis = 0)[1:-1].flatten(), WCS_MAT_CHROMA[1:-1].flatten())

relative = np.std(DE_MAT,axis = 0)/np.mean(DE_MAT,axis = 0)
dis.display_munsells_inv(relative, np.nanmax(relative))


# In[9]: COLOR CONSTANCY INDEX


CCI_MAT = from330to8x40(ERRORS[layers[0]][conditions[0]]['CCI'])
dis.display_munsells_inv(np.mean(CCI_MAT,axis = (0,1,-1)), np.nanmax(np.mean(CCI_MAT,axis = (0,1,-1))))

### Some stats

CCI_MAT_chrom = np.median(CCI_MAT, axis = (0,1,-1))[1:-1,1:]
CCI_MAT_achrom = np.median(CCI_MAT, axis = (0,1,-1))[:,0]

dis.DEFINE_PLT_RC(type = 0.5)

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.scatter(range(40)*8,CCI_MAT_chrom.flatten(),color = WCS_MAT_sRGB[1:-1,1:].reshape(-1,3),marker = '.',s = 400)
ax1.set_xlabel('Hue')
ax1.set_xticks([1.5,5.5,9.5,13.5,18.5,22.5,26.5,30.5,34.5,38.5])
ax1.set_xticklabels(['R','YR','Y','GY','G','BG','B','PB','P','RP'])

plt.yticks(np.arange(0,1.1,0.5))
plt.xlim(-1,40)
ax1.set_ylabel('Median CCI')
fig.tight_layout()
fig.savefig(figures_dir_path +'CCI_vs_hue.png', dpi=400)
plt.show()

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.scatter(range(9,1,-1)*40,CCI_MAT_chrom.T.flatten(),color = np.moveaxis(WCS_MAT_sRGB[1:-1,1:],1,0).reshape(-1,3),marker = '.',s = 400)
ax1.scatter(range(10,0,-1),CCI_MAT_achrom,color = WCS_MAT_sRGB[:,0],marker = '.',s = 400)
ax1.set_xlabel('Value')
ax1.set_xticks(range(0,11,2))
#ax1.set_xticklabels(['R','YR','Y','GY','G','BG','B','PB','P','RP'])

plt.yticks(np.arange(0,1.1,0.5),fontsize = 19)
ax1.set_ylabel('Median CCI')
plt.xlim(0.5,10.5)
fig.tight_layout()
fig.savefig(figures_dir_path+'CCI_vs_value.png', dpi=400)
plt.show()


#certainty = np.moveaxis(np.amax(OUT_soft = LoadandComputeOutputs(path2activations, NetType, Testing_type, layer, testing_condition) # Load output of last layer,axis = -1),-1, 1)


np.median(ERRORS[layers[0]][conditions[0]]['CCI'])
np.median(ERRORS[layers[0]][conditions[0]]['DE'])
#np.mean(ERRORS[layers[0]][conditions[0]]['CCI'][certainty>0.25])


CCI_Y = ERRORS[layers[0]][conditions[0]]['CCI'][:,:,:,0]
CCI_B = ERRORS[layers[0]][conditions[0]]['CCI'][:,:,:,1]
CCI_G = ERRORS[layers[0]][conditions[0]]['CCI'][:,:,:,2]
CCI_R = ERRORS[layers[0]][conditions[0]]['CCI'][:,:,:,3]
np.median((CCI_Y,CCI_B,CCI_G, CCI_R))
np.median((CCI_Y,CCI_B))
np.median((CCI_G,CCI_R))

#np.mean((CCI_Y[certainty[:,:,:,0]>0.25],CCI_B[certainty[:,:,:,1]>0.25]))
#np.mean((CCI_G[certainty[:,:,:,2]>0.25],CCI_R[certainty[:,:,:,3]>0.25]))

np.median(ERRORS[layers[0]][conditions[0]]['CCI'])
np.amax(ERRORS[layers[0]][conditions[0]]['CCI'])
np.amin(ERRORS[layers[0]][conditions[0]]['CCI'])
np.percentile(ERRORS[layers[0]][conditions[0]]['CCI'],50)
#np.corrcoef(DE2.flatten(),ERRORS[layers[0]][conditions[0]]['CCI'])
#np.corrcoef(DE2.flatten(),certainty.flatten())
#np.corrcoef(certainty.flatten(),CCI.flatten())

### Distribution of CCI

fig = plt.figure()
plt.hist(ERRORS[layers[0]][conditions[0]]['CCI'].flatten(),bins = 40, color= 'black')
plt.vlines(np.nanmean(ERRORS[layers[0]][conditions[0]]['CCI']),ymin = 0, ymax = 70000,color = 'r')
#ax1.vlines(np.nanpercentile(1-p1,25),ymin = 0, ymax = 80000,color = 'orange')
plt.vlines(np.nanpercentile(ERRORS[layers[0]][conditions[0]]['CCI'],10),ymin = 0, ymax = 80000,color = 'g')
plt.xlabel('Color Constancy Index',fontsize = 15)
plt.ylabel('Count',fontsize = 15)
plt.xticks(np.arange(-1,1.1,0.5),fontsize = 14)
plt.yticks(fontsize = 14)
plt.xlim(-2,1)

fig.tight_layout
plt.show()


CCI_2 = ERRORS[layers[0]][conditions[0]]['CCI'].copy()
CCI_2[(ERRORS[layers[0]][conditions[0]]['CCI'])<0] = 0


hY = np.histogram(CCI_2[:,:,:,0].flatten(),bins = np.arange(-0.025,1.075,0.05))
hB = np.histogram(CCI_2[:,:,:,1].flatten(),bins = np.arange(-0.025,1.075,0.05))
hG = np.histogram(CCI_2[:,:,:,2].flatten(),bins = np.arange(-0.025,1.075,0.05))
hR = np.histogram(CCI_2[:,:,:,3].flatten(),bins = np.arange(-0.025,1.075,0.05))

dis.DEFINE_PLT_RC(type = 0.5)

fig = plt.figure()
plt.plot(hY[1][:-1]+0.025, hY[0].astype(float)/np.sum(hY[0]), color= [0.8,0.7,0.3],lw = 5)
plt.plot(hB[1][:-1]+0.025, hB[0].astype(float)/np.sum(hB[0]), color= [0.3,0.4,0.8],lw = 5)
plt.plot(hG[1][:-1]+0.025, hG[0].astype(float)/np.sum(hG[0]), color= [0.4,0.8,0.4],lw = 5)
plt.plot(hR[1][:-1]+0.025, hR[0].astype(float)/np.sum(hR[0]), color= [0.7,0.3,0.7],lw = 5)
#plt.scatter(np.nanmedian(CCI[:,:,:,0]),0,color = 'orange',marker = '*',s = 200)
#plt.scatter(np.nanmedian(CCI[:,:,:,1]),0,color = 'b',marker = '*',s = 200)
#plt.scatter(np.nanmedian(CCI[:,:,:,2]),0,color = 'g',marker = '*',s = 200)
#plt.scatter(np.nanmedian(CCI[:,:,:,3]),0,color = 'r',marker = '*',s = 200)
plt.xlabel('CCI')
plt.ylabel('Frequency')
plt.xticks(np.arange(0,1.1,0.5))
plt.yticks(np.arange(0,0.4,0.1))
plt.xlim(-0.05,1)
#fig.tight_layout
fig.savefig(figures_dir_path +'YBGR_distrib_CCI.png', dpi=400)
plt.show()

dis.DEFINE_PLT_RC(type = 1)

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.bar(np.arange(1,5),[np.nanmedian(CCI_Y), np.nanmedian(CCI_B), np.nanmedian(CCI_G), np.nanmedian(CCI_R) ],color = [[0.8,0.7,0.3],[0.3,0.4,0.8],[0.4,0.8,0.4],[0.7,0.3,0.7]])
ax1.set_xticks([1,2,3,4])
ax1.set_xticklabels(['Y','B','G','R'],rotation=0)
plt.yticks(np.arange(0,1.1,0.25))
ax1.set_ylabel('Median CCI')
plt.ylim(0.5,1)
#fig.tight_layout()
fig.savefig(figures_dir_path +'median_YBGR.png', dpi=400)
plt.show()


'''
fig = plt.figure()
perc = np.nanpercentile(1-CCI[certainty > 0.25],np.arange(0,100,0.1))
plt.plot(np.arange(0,100,0.1),perc, color= 'black',lw = 5)
plt.hlines(np.nanmedian(perc),xmin = 0, xmax = 100,color = 'r',lw = 5)
plt.xlabel('Percentile',fontsize = 15)
plt.ylabel('CCI',fontsize = 15)
plt.xticks(np.arange(0,101,25),fontsize = 14)
plt.yticks(fontsize = 14)
#plt.xlim(0,10)
#plt.title('Munsell chips WCS',fontsize = 18)
#fig.savefig('Munsell_chips_WCS.png')
fig.tight_layout
fig.savefig('CCI_percentile.png', dpi=800)
plt.show()'''

fig = plt.figure()
perc = np.nanpercentile(ERRORS[layers[0]][conditions[0]]['CCI'],np.arange(0,100,0.1))
plt.plot(np.arange(0,100,0.1),perc, color= 'black')
plt.hlines(np.nanmean(perc),xmin = 0, xmax = 100,color = 'r')
plt.xlabel('Percentile',fontsize = 15)
plt.ylabel('CCI',fontsize = 15)
#plt.xticks(np.arange(0,101,25),fontsize = 14)
plt.yticks(fontsize = 14)
plt.xlim(-1,10)

#fig.tight_layout
plt.show()


### FINDING OUT WHY SOME STIMULI LEAD TO SUCH BAD RESULTS

ARG = np.argwhere(ERRORS[layers[0]][conditions[0]]['CCI']<np.nanpercentile(ERRORS[layers[0]][conditions[0]]['CCI'],2.5))

fig = plt.figure()
p = plt.hist(ARG[:,-1],bins = range(0,5,1), color= 'black')
plt.xlabel('Illuminant',fontsize = 15)
plt.ylabel('count',fontsize = 15)
#plt.xticks(np.arange(0,101,25),fontsize = 14)
plt.yticks(fontsize = 14)
plt.xticks(range(5),fontsize = 14)
#plt.xlim(-1,10)
plt.title('Illuminant for CCI percentile 2.5',fontsize = 18)
#fig.savefig(figures_dir_path + 'Munsell_chips_WCS.png', dpi = 400)
fig.tight_layout
plt.show()


fig = plt.figure()
p = plt.hist(ARG[:,0],bins = range(0,11,1), color= 'black')
plt.xlabel('Model index',fontsize = 15)
plt.ylabel('count',fontsize = 15)
#plt.xticks(np.arange(0,101,25),fontsize = 14)
plt.yticks(fontsize = 14)
#plt.xlim(-1,10)
plt.title('Model for CCI percentile 2.5',fontsize = 18)
#fig.savefig('Munsell_chips_WCS.png')
fig.tight_layout
plt.show()


# In[9]: Comparison between conditions


def closest_idx(vect, list_vect):
    return np.argmin(np.linalg.norm(list_vect - vect, axis = -1))

def predicted_errors(WCS_LAB_4, MUNSELL_LAB, MUNS_LAB_4, wcs_idx, illu):
    '''
    Arguments:
        WCS_LAB_4: array of WCS munsells' chromaticities under 4 illuminants. dim = [4,330,3]
        MUNSELL_LAB: array of munsell chips chromaticities under D65. dim = [1600, 3]
        MUNS_LAB_4: array of munsell chips chromaticities under 4 illuminants. dim = [1600, 4, 3]
        WCS_idx: WCS index of the munsell for which to compute errors
        illu: illuminant index
    '''
    list_WCS_labels = algos.compute_WCS_Munsells_categories()
    predicted_idx = closest_idx(WCS_LAB_4[illu, wcs_idx], MUNSELL_LAB)
    de_predicted = np.linalg.norm(MUNS_LAB_4[predicted_idx, illu] - WCS_LAB_4[illu,wcs_idx])
    de_shifted = np.linalg.norm(WCS_LAB_4[illu, wcs_idx] - MUNSELL_LAB[list_WCS_labels[wcs_idx]])
    cci = 1 - de_predicted / de_shifted
    return (predicted_idx == list_WCS_labels[wcs_idx])*1, de_predicted, cci

def full_predicted_errors(WCS_LAB_4, MUNSELL_LAB, MUNS_LAB_4):
    '''
    Arguments:
        WCS_LAB_4: array of WCS munsells' chromaticities under 4 illuminants. dim = [4,330,3]
        MUNSELL_LAB: array of munsell chips chromaticities under D65. dim = [1600, 3]
        MUNS_LAB_4: array of munsell chips chromaticities under 4 illuminants. dim = [1600, 4, 3]
    '''
    ERROR = np.zeros((len(WCS_LAB_4), len(WCS_LAB_4[0]), 3))
    for illu in range(len(WCS_LAB_4)):
        for wcs_idx in range(len(WCS_LAB_4[illu])):
            ERROR[illu, wcs_idx] = predicted_errors(WCS_LAB_4, MUNSELL_LAB, MUNS_LAB_4, wcs_idx, illu)
    return {'Accu': 100*np.sum(ERROR[:,:,0], axis = (1))/ERROR.shape[1], 'DE': np.median(ERROR[:,:,1], axis = 1), 'CCI': np.median(ERROR[:,:,2], axis = 1)}

#WCS_LAB_4
#MUNS_LAB_4
#MUNSELL_LAB


predicted_errors = full_predicted_errors(WCS_LAB_4.T, MUNSELL_LAB, MUNS_LAB_4 )

#list_WCS_labels = algos.compute_WCS_Munsells_categories() # import idxes of WCS munsells among the 1600



CCI_normal = np.array([np.median(ERRORS[layer]['normal']['CCI']) for layer in layers[::-1]])
DE_normal = np.array([np.mean(ERRORS[layer]['normal']['DE']) for layer in layers[::-1]])
Accu_normal = np.array([np.mean(ERRORS[layer]['normal']['Accu']) for layer in layers[::-1]])
#CCI_normal_bar = np.array([np.std(ERRORS[layer]['normal']['CCI']) for layer in layers])

CCI_no_patch = np.array([np.median(ERRORS[layer]['no_patch']['CCI']) for layer in layers[::-1]])
DE_no_patch = np.array([np.mean(ERRORS[layer]['no_patch']['DE']) for layer in layers[::-1]])
Accu_no_patch = np.array([np.mean(ERRORS[layer]['no_patch']['Accu']) for layer in layers[::-1]])
#CCI_no_patch = np.load(CC_dir_path +'training_centered/All_muns/CCI_fost_weighted_no_patch_std.npy')

CCI_no_back = np.array([np.median(ERRORS[layer]['no_back']['CCI']) for layer in layers[::-1]])
DE_no_back = np.array([np.mean(ERRORS[layer]['no_back']['DE']) for layer in layers[::-1]])
Accu_no_back = np.array([np.mean(ERRORS[layer]['no_back']['Accu']) for layer in layers[::-1]])
#CCI_no_back = np.load(CC_dir_path +'training_centered/WCS/finetuning/CCI_fost_weighted_no_back_std.npy')

CCI_wrong_illu = np.array([np.median(ERRORS[layer]['wrong_illu']['CCI']) for layer in layers[::-1]])
DE_wrong_illu = np.array([np.mean(ERRORS[layer]['wrong_illu']['DE']) for layer in layers[::-1]])
Accu_wrong_illu = np.array([np.mean(ERRORS[layer]['wrong_illu']['Accu']) for layer in layers[::-1]])
#CCI_wrong_illu = np.load(CC_dir_path +'training_centered/All_muns/CCI_fost_weighted_wrong_illu_std.npy')

CCI_D65 = np.array([np.nanmedian(ERRORS[layer]['D65']['CCI']) for layer in layers[::-1]])
DE_D65 = np.array([np.nanmean(ERRORS[layer]['D65']['DE']) for layer in layers[::-1]])
Accu_D65 = np.array([np.nanmean(ERRORS[layer]['D65']['Accu']) for layer in layers[::-1]])


fig = plt.figure(figsize = (6,6))
ax1 = fig.add_subplot(111)
ax1.bar([1,2,3,4,5,6],[Accu_normal[-1], Accu_no_patch[-1], Accu_wrong_illu[-1], Accu_no_back[-1], Accu_D65[-1], predicted_errors['Accu'].mean()], color = ['k',[0.4,0.7,0.8],[0.4,0.8,0.4],[0.7,0.8,0.4],[0.8,0.4,0.4],'grey'],linewidth = 6)
ax1.set_xticks([1,2,3,4,5,6])
ax1.set_xticklabels([])
plt.xticks(fontsize = 21)
#plt.yticks(fontsize = 14)
plt.yticks(np.arange(0,105,25),fontsize = 21)
ax1.set_ylabel('Accuracy',fontsize = 25)
ax1.set_xticklabels(['CC$_{normal}$','CC$_{no patch}$','CC$_{wrong back}$','CC$_{no back}$','D65$_{normal}$', 'naive'], rotation = 45)
fig.tight_layout()
fig.savefig(figures_dir_path + 'Accuracy.png', dpi=400)
plt.show()

fig = plt.figure(figsize = (6,6))
ax1 = fig.add_subplot(111)
ax1.bar([1,2,3,4,5, 6],[DE_normal[-1], DE_no_patch[-1], DE_wrong_illu[-1], DE_no_back[-1], DE_D65[-1], predicted_errors['DE'].mean()],color = ['k',[0.4,0.7,0.8],[0.4,0.8,0.4],[0.7,0.8,0.4],[0.8,0.4,0.4],'grey'],linewidth = 6)
ax1.set_xticks([1,2,3,4,5, 6])
ax1.set_xticklabels([])
plt.xticks(fontsize = 21)
ax1.set_xticklabels(['CC$_{normal}$','CC$_{no patch}$','CC$_{wrong back}$','CC$_{no back}$','D65$_{normal}$', 'naive'], rotation = 45)
#plt.yticks(fontsize = 14)
plt.yticks(np.arange(0,41,10),fontsize = 21)
ax1.set_ylabel('$\Delta$E',fontsize = 25)
fig.tight_layout()
fig.savefig(figures_dir_path + 'DeltaE.png', dpi=400)
plt.show()

fig = plt.figure(figsize = (6,6))
ax1 = fig.add_subplot(111)
ax1.bar([1,2,3,4,5,6],[CCI_normal[-1], CCI_no_patch[-1], CCI_wrong_illu[-1], CCI_no_back[-1], CCI_D65[-1], predicted_errors['CCI'].mean()],color = ['k',[0.4,0.7,0.8],[0.4,0.8,0.4],[0.7,0.8,0.4],[0.8,0.4,0.4],'grey'],linewidth = 6)
ax1.set_xticks([1,2,3,4,5,6])
ax1.set_xticklabels([])
plt.xticks(fontsize = 21)
ax1.set_xticklabels(['CC$_{normal}$','CC$_{no patch}$','CC$_{wrong back}$','CC$_{no back}$','D65$_{normal}$', 'naive'], rotation = 45)
#plt.yticks(fontsize = 14)
plt.yticks(np.arange(-4.5,1.1,2),fontsize = 21)
ax1.set_ylabel('CCI',fontsize = 25)
fig.tight_layout()
fig.savefig(figures_dir_path + 'CCI.png', dpi=800)
plt.show()

#CCI_no_patch[CCI_no_patch<0] = 0
#CCI_normal[CCI_normal<0] = 0
#CCI_no_back[CCI_no_back<0] = 0
#CCI_wrong_illu[CCI_wrong_illu<0] = 0
#CCI_D65[CCI_D65<0] = 0
fig = plt.figure(figsize = (8,8))
ax1 = fig.add_subplot(111)
ax1.errorbar([1,2,3,4,5],CCI_normal, yerr = [0,0,0,0,0],color = 'k',linewidth = 6)
ax1.errorbar([1,2,3,4,5],CCI_no_patch, yerr = [0,0,0,0,0],color = [0.4,0.7,0.8],linewidth = 6)
ax1.errorbar([1,2,3,4,5],CCI_wrong_illu +[0.01,0.01,0.01,0.01,0.01],yerr = [0,0,0,0,0],color = [0.4,0.8,0.4],linewidth = 6)
ax1.errorbar([1,2,3,4,5], CCI_no_back +[0,0,0,0,0],yerr = [0,0,0,0,0],color = [0.7,0.8,0.4],linewidth = 6)
ax1.errorbar([1,2,3,4,5], CCI_D65 + [-0.01,-0.01,-0.01,-0.01,-0.01],yerr = [0,0,0,0,0],color = [0.8,0.4,0.4],linewidth = 6)
ax1.set_xlabel('Readouts',fontsize = 20)
ax1.set_xticks([1,2,3,4,5])
ax1.set_xticklabels(['RC1','RC2','RC3','RF1','RF2'])
plt.xticks(fontsize = 17)
plt.yticks(fontsize = 17)
#plt.yticks(np.arange(-1,1.1,0.5),fontsize = 14)
ax1.set_ylabel('Median CCI',fontsize = 20)
fig.tight_layout()
fig.savefig(figures_dir_path + 'CCI_fost_readout.png', dpi=1200)
plt.show()


