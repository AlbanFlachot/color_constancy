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
import seaborn as sns
from statannot import add_stat_annotation
import scipy.stats as stats
import pandas as pd

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
		layer = pickle.load(pickle_in, encoding='latin1')
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
    return pickle.load(f, fix_imports=True, encoding='latin1', errors="strict")


# In[9]: LOAD ERROR MEASURES

measures = ['Accu', 'DE', 'CCI', 'DE_3D', 'Accu5', 'Accu_munscube']

conditions = ['normal','no_patch', 'wrong_illu', 'no_back', 'D65']

SofA_models = ['RefResNet','MobileNet', 'ResNet11', 'ResNet18', 'ResNet50', 'VGG11_bn']

layers = ['fc2','fc1', 'c3', 'c2', 'c1']

ERRORS = {}
Errors_D65_D65 = {}

path = pickles_dir_path

## RefConvNet errors
for layer in layers:
    ERRORS[layer] = {}
    for condition in conditions[:-1]:
        ERRORS[layer][condition] = load_pickle(path + 'Errors_Original_CC_4illu_WCS_%s_%s.pickle'%(layer, condition))
    ERRORS[layer][conditions[-1]] = load_pickle(path + 'Errors_Original_D65_4illu_WCS_%s_normal.pickle'%(layer))
    Errors_D65_D65[layer] = load_pickle(path + 'Errors_Original_D65_D65_WCS_%s_normal.pickle'%(layer))

for condition in conditions[:2]:
    for measure in measures:
        if measure == 'CCI':
            print('We found a %s of %f for RefConvNet under %s '%(measure, np.median(ERRORS['fc2'][condition][measure]), condition))
        else:
            print('We found a %s of %f for RefConvNet under %s '%(measure, np.mean(ERRORS['fc2'][condition][measure]), condition))

if not os.path.exists('../figures'):
    os.mkdir('../figures')

# State of the art
ERRORS_SofA = {}
for mS_model in SofA_models:
    ERRORS_SofA[mS_model] = {}
    for condition in conditions[:2]:
        ERRORS_SofA[mS_model][condition] = load_pickle(path + 'Errors_%s_CC_5illu_WCS__%s.pickle'%(mS_model, condition))

for mS_model in SofA_models:
    for condition in conditions[:2]:
        for measure in measures:
            if measure == 'CCI':
                print('We found a %s of %f for model %s under %s '%(measure, np.median(ERRORS_SofA[mS_model][condition][measure]), mS_model, condition))

            else:
                print('We found a %s of %f for model %s under %s '%(measure, np.mean(ERRORS_SofA[mS_model][condition][measure]), mS_model, condition))


#_____________________________________________________________________________________________________________________________

### ANALYZE OUTPUT LAST LAYER NORMAL CONDITION
#_____________________________________________________________________________________________________________________________

CHROMA = list()
with open(txt_dir_path +"WCS_chroma.txt") as f:
    for line in f:
       CHROMA.append(int(line.split()[0]))


WCS_MAT_CHROMA = from330to8x40(CHROMA)
dis.display_munsells_inv(WCS_MAT_CHROMA,16,'WCS Chroma')


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
    XYZ_WCS = pickle.load(fp, encoding='latin1')

RGB_muns = [CT.XYZ2sRGB(XYZ) for XYZ in XYZ_WCS]
WCS_MAT_RGB =from330to8x40(RGB_muns)

WCS_MAT_sRGB = (WCS_MAT_RGB - np.amin(WCS_MAT_RGB))/(np.amax(WCS_MAT_RGB)-np.amin(WCS_MAT_RGB))
RGB_muns = (RGB_muns - np.amin(RGB_muns))/(np.amax(RGB_muns)-np.amin(RGB_muns))

dis.display_munsells_inv(WCS_MAT_sRGB,1,'WCS Munsells')


nb_mod = ERRORS[layers[0]][conditions[0]]['DE'].shape[0]
nb_obj = ERRORS[layers[0]][conditions[0]]['DE'].shape[1]



# In[9]: Compute accuracy as a measure of LAB delta E

# CIELab coordinates of WCS muns under YBGR illuminants
WCS_LAB_4 = np.load(npy_dir_path +'LAB_WCS_ABCD.npy')

MUNS_LAB_4 = np.moveaxis(np.load(npy_dir_path +'LAB_MUNS_ABCD.npy'),0,-1)

MUNSELL_LAB = np.load(npy_dir_path +'MUNSELL_LAB.npy')

MUNS_neighbouring_dist = np.load(npy_dir_path + 'MUNS_neighbouring_dist.npy')
WCS_neighbouring_dist = MUNS_neighbouring_dist[algos.compute_WCS_Munsells_categories()]

DE_illu = np.mean(ERRORS[layers[0]][conditions[0]]['DE'],axis = (0,1,-1))
DE_muns = np.mean(ERRORS[layers[0]][conditions[0]]['DE'],axis = (0,-2,-1))
print('Average Delta E value per illuminations:')
print(DE_illu)

DE_D65 = np.mean(Errors_D65_D65[layers[0]]['DE'],axis = (0,1,-1))
print('Average Delta E value for D65:')
print(DE_D65)


DE_MAT = from330to8x40(np.mean(ERRORS[layers[0]][conditions[0]]['DE'],axis = (-1,-2)))
DE_MAT_D65 = from330to8x40(np.mean(Errors_D65_D65[layers[0]]['DE'],axis = (-1,-2)))
DIST_neighbours_MAT = from330to8x40(WCS_neighbouring_dist)

normalized_DELTAE = np.mean(DE_MAT, axis = 0)/DIST_neighbours_MAT
normalized_DELTAE[np.mean(DE_MAT, axis = 0) == 0] = 0

Accu_MAT = from330to8x40(ERRORS[layers[0]][conditions[0]]['Accu'])

dis.display_munsells_inv(np.mean(Accu_MAT,axis = (0,-1)), np.amax(np.mean(Accu_MAT,axis = (0,-1))), 'Accuracy')
dis.display_munsells_inv(np.mean(DE_MAT,axis = 0), np.amax(np.mean(DE_MAT,axis = 0)), '$\Delta$E')

DE_MAT_normal = DE_MAT
DE_MAT_nopatch =  from330to8x40(np.mean(ERRORS[layers[0]][conditions[1]]['DE'],axis = (-1,-2)))
DE_MAT_wrongback =  from330to8x40(np.mean(ERRORS[layers[0]][conditions[2]]['DE'],axis = (-1,-2)))
DE_MAT_noback =  from330to8x40(np.mean(ERRORS[layers[0]][conditions[3]]['DE'],axis = (-1,-2)))
DE_MAT_D65D65 =  from330to8x40(np.mean(ERRORS[layers[0]][conditions[-1]]['DE'],axis = (-1,-2)))

dis.display_munsells_inv(np.mean(DE_MAT_normal,axis = 0), np.amax(np.mean(DE_MAT_normal,axis = 0)), '$\Delta$E$_{normal}$', save = True, add = figures_dir_path + '$\Delta$E$_{normal}$.png')
dis.display_munsells_inv(np.mean(DE_MAT_nopatch,axis = 0), np.amax(np.mean(DE_MAT_nopatch,axis = 0)), '$\Delta$E$_{nopatch}$', save = True, add = figures_dir_path + '$\Delta$E$_{nopatch}$')
dis.display_munsells_inv(np.mean(DE_MAT_wrongback,axis = 0), np.amax(np.mean(DE_MAT_wrongback,axis = 0)), '$\Delta$E$_{wrongback}$', save = True,add = figures_dir_path + '$\Delta$E$_{wrongback}$')
dis.display_munsells_inv(np.mean(DE_MAT_noback,axis = 0), np.amax(np.mean(DE_MAT_noback,axis = 0)), '$\Delta$E$_{noback}$', save = True, add = figures_dir_path + '$\Delta$E$_{noback}$')
dis.display_munsells_inv(np.mean(DE_MAT_D65D65,axis = 0), np.amax(np.mean(DE_MAT_D65D65,axis = 0)), '$\Delta$E$_{D65}$', save = True, add = figures_dir_path + '$\Delta$E$_{D65}$')

CCI_MAT_normal = from330to8x40(np.mean(ERRORS[layers[0]][conditions[0]]['CCI'],axis = (1,-1)))
CCI_MAT_nopatch =  from330to8x40(np.mean(ERRORS[layers[0]][conditions[1]]['CCI'],axis = (-1,1)))
CCI_MAT_wrongback =  from330to8x40(np.mean(ERRORS[layers[0]][conditions[2]]['CCI'],axis = (-1,1)))
CCI_MAT_noback =  from330to8x40(np.mean(ERRORS[layers[0]][conditions[3]]['CCI'],axis = (-1,1)))
CCI_MAT_D65D65 =  from330to8x40(np.mean(ERRORS[layers[0]][conditions[-1]]['CCI'],axis = (-1,1)))

dis.display_munsells_inv(1-np.mean(CCI_MAT_normal,axis = 0), np.amax(1-np.mean(CCI_MAT_normal,axis = 0)), 'CCI$_{normal}$',  save = True, add = figures_dir_path + 'CCI$_{normal}$.png')
dis.display_munsells_inv(1-np.mean(CCI_MAT_nopatch,axis = 0), np.amax(1-np.mean(CCI_MAT_nopatch,axis = 0)), 'CCI$_{nopatch}$',  save = True, add = figures_dir_path + 'CCI$_{nopatch}$.png')
dis.display_munsells_inv(1-np.mean(CCI_MAT_wrongback,axis = 0), np.amax(1-np.mean(CCI_MAT_wrongback,axis = 0)), 'CCI$_{wrongback}$',  save = True, add = figures_dir_path + 'CCI$_{wrongback}$.png')
dis.display_munsells_inv(1-np.mean(CCI_MAT_noback,axis = 0), np.amax(1-np.mean(CCI_MAT_noback,axis = 0)), 'CCI$_{noback}$',  save = True, add = figures_dir_path + 'CCI$_{noback}$.png')
dis.display_munsells_inv(1-np.mean(CCI_MAT_D65D65,axis = 0), np.amax(1-np.mean(CCI_MAT_D65D65,axis = 0)), 'CCI$_{D65}$',  save = True, add = figures_dir_path + 'CCI$_{D65}$.png')

np.corrcoef(WCS_MAT_CHROMA[1:-1,1:].flatten(), np.mean(DE_MAT_wrongback,axis = 0)[1:-1,1:].flatten())
np.corrcoef(WCS_MAT_CHROMA[1:-1,1:].flatten(), 1-np.mean(CCI_MAT_wrongback,axis = 0)[1:-1,1:].flatten())
np.corrcoef(WCS_MAT_CHROMA[1:-1,1:].flatten(), np.mean(DE_MAT_noback,axis = 0)[1:-1,1:].flatten())
np.corrcoef(WCS_MAT_CHROMA[1:-1,1:].flatten(), 1-np.mean(CCI_MAT_noback,axis = 0)[1:-1,1:].flatten())
np.corrcoef(WCS_MAT_CHROMA[1:-1,1:].flatten(), np.mean(DE_MAT_D65D65,axis = 0)[1:-1,1:].flatten())
dis.display_munsells_inv(np.mean(DE_MAT_D65,axis = 0), np.amax(np.mean(DE_MAT_D65,axis = 0)), '$\Delta$E$_{D65}$')
dis.display_munsells_inv(DE_MAT[0], np.amax(DE_MAT[0]),'$\Delta$E')
dis.display_munsells_inv(DIST_neighbours_MAT, np.amax(DIST_neighbours_MAT),'Dist Neighbours')
#dis.display_munsells_inv(normalized_DELTAE,np.amax(normalized_DELTAE),'Normalize $\Delta$E')

# Comparison with consistency
dis.display_munsells_inv(np.std(DE_MAT,axis = 0), np.amax(np.std(DE_MAT,axis = 0)), 'STD $\Delta$E')
np.corrcoef(np.std(DE_MAT,axis = 0)[1:-1].flatten(), WCS_MAT_CHROMA[1:-1].flatten())

relative = np.std(DE_MAT,axis = 0)/np.mean(DE_MAT,axis = 0)
#dis.display_munsells_inv(relative, np.nanmax(relative))


# In[9]: COLOR CONSTANCY INDEX


CCI_MAT = from330to8x40(ERRORS[layers[0]][conditions[0]]['CCI'])
dis.display_munsells_inv(np.mean(CCI_MAT,axis = (0,1,-1)), np.nanmax(np.mean(CCI_MAT,axis = (0,1,-1))),'CCI')

### Some stats

CCI_MAT_chrom = np.median(CCI_MAT, axis = (0,1,-1))[1:-1,1:]
CCI_MAT_achrom = np.median(CCI_MAT, axis = (0,1,-1))[:,0]

dis.DEFINE_PLT_RC(type = 0.5)

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.scatter(list(range(40))*8,CCI_MAT_chrom.flatten(),color = WCS_MAT_sRGB[1:-1,1:].reshape(-1,3),marker = '.',s = 400)
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
ax1.scatter(list(range(9,1,-1))*40,CCI_MAT_chrom.T.flatten(),color = np.moveaxis(WCS_MAT_sRGB[1:-1,1:],1,0).reshape(-1,3),marker = '.',s = 400)
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

dis.DEFINE_PLT_RC(type = 0.5)

colors = [tuple([0.8,0.7,0.3]),tuple([0.3,0.4,0.8]),tuple([0.4,0.8,0.4]),tuple([0.7,0.3,0.7])]
#colors = ['o','b','g','m']

### Figures bbox for the effect of the illumination on the Color constancy Index

#ax = stats.ttest_rel(np.nanmedian(CCI_Y, axis = (1,2)), np.nanmedian(CCI_B, axis = (1,2)))

DICT_CCI = {}
DICT_CCI['Y'],DICT_CCI['B'], DICT_CCI['G'],DICT_CCI['R'] = np.nanmedian(CCI_Y, axis = (1,2)), np.nanmedian(CCI_B, axis = (1,2)), np.nanmedian(CCI_G, axis = (1,2)), np.nanmedian(CCI_R, axis = (1,2)) 
dat = pd.DataFrame(DICT_CCI)

fig = plt.figure(figsize = (7,10))
ax = sns.boxplot( data = dat , linewidth = 3, palette = colors, width = 0.85)
test_results = add_stat_annotation(ax, data=dat,
                                   box_pairs=[('Y', 'B'), ('G', 'R'), ('B', 'G')],
                                   test='t-test_paired', text_format='star',
                                   loc='outside', verbose=2)
ax.set_yticks([0.6,0.8,1])
ax.set_ylabel('Median CCI')
plt.ylim(0.6,1.1)
#plt.title('ConvNet')
fig.subplots_adjust(top=0.973,bottom=0.049,left=0.214,right=0.955,hspace=0.2,wspace=0.2)
fig.savefig(figures_dir_path +'ConvNet_median_YBGR.png', dpi=400)
plt.show()


CCI_ResNet = ERRORS_SofA['RefResNet'][conditions[0]]['CCI']
DICT_CCI = {}
DICT_CCI['Y'],DICT_CCI['B'], DICT_CCI['G'],DICT_CCI['R'] = np.nanmedian(CCI_ResNet[:,:,:,0], axis = (1,2)), np.nanmedian(CCI_ResNet[:,:,:,1], axis = (1,2)), np.nanmedian(CCI_ResNet[:,:,:,2], axis = (1,2)), np.nanmedian(CCI_ResNet[:,:,:,3], axis = (1,2)) 
dat = pd.DataFrame(DICT_CCI)

fig = plt.figure(figsize = (6.5,10))
ax = sns.boxplot( data = dat , linewidth = 3, palette = colors, width = 0.85)
test_results = add_stat_annotation(ax, data=dat,
                                   box_pairs=[('Y', 'B'), ('G', 'R'), ('B', 'G')],
                                   test='t-test_paired', text_format='star',
                                   loc='outside', verbose=2)
ax.set_yticks([0.9,1])
#ax.set_ylabel('Median CCI')
plt.ylim(0.85,1.01)
#plt.title('ResNet')
fig.subplots_adjust(top=0.973,bottom=0.049,left=0.139,right=0.955,hspace=0.2,wspace=0.2)
fig.savefig(figures_dir_path +'ResNet_median_YBGR.png', dpi=400)
plt.show()



'''
fig = plt.figure()
ax1 = fig.add_subplot(111)
bplot = ax1.boxplot([np.nanmedian(CCI_Y, axis = (1,2)), np.nanmedian(CCI_B, axis = (1,2)), np.nanmedian(CCI_G, axis = (1,2)), np.nanmedian(CCI_R, axis = (1,2)) ],
                     vert=True,  # vertical box alignment
                     patch_artist=True,  # fill with color
                     )
ax1.set_xticks([1,2,3,4])
ax1.set_xticklabels(['Y','B','G','R'],rotation=0)
plt.yticks(np.arange(0,1.1,0.25))
ax1.set_ylabel('Median CCI')
plt.ylim(0.6,1.1)
for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)

#fig.tight_layout()
fig.savefig(figures_dir_path +'median_YBGR.png', dpi=400)
plt.show()



CCI_ResNet = ERRORS_SofA['RefResNet'][conditions[0]]['CCI']
fig = plt.figure()
ax1 = fig.add_subplot(111)
bplot = ax1.boxplot([np.nanmedian(CCI_ResNet[:,:,:,0], axis = (1,2)), np.nanmedian(CCI_ResNet[:,:,:,1], axis = (1,2)), np.nanmedian(CCI_ResNet[:,:,:,2], axis = (1,2)), np.nanmedian(CCI_ResNet[:,:,:,3], axis = (1,2)) ],
                     vert=True,  # vertical box alignment
                     patch_artist=True,  # fill with color
                     )
ax1.set_xticks([1,2,3,4])
ax1.set_xticklabels(['Y','B','G','R'],rotation=0)
plt.yticks(np.arange(0,1.1,0.25))
ax1.set_ylabel('Median CCI')
plt.title('ResCC')
plt.ylim(0.5,1)
#fig.tight_layout()
fig.savefig(figures_dir_path +'ResNet_median_YBGR.png', dpi=400)
plt.show()'''

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

CCI_normal = np.array([np.median(ERRORS[layer]['normal']['CCI'], axis = (1,2,3)) for layer in layers[::-1]])
DE_normal = np.array([np.mean(ERRORS[layer]['normal']['DE'], axis = (1,2,3)) for layer in layers[::-1]])
Accu_normal = np.array([np.mean(ERRORS[layer]['normal']['Accu'], axis = (1,2)) for layer in layers[::-1]])
Accu5_normal = np.array([np.mean(ERRORS[layer]['normal']['Accu5'], axis = (1,2)) for layer in layers[::-1]])
Accu_munscube_normal = np.array([np.mean(ERRORS[layer]['normal']['Accu_munscube'], axis = (1,2)) for layer in layers[::-1]])
#CCI_normal_bar = np.array([np.std(ERRORS[layer]['normal']['CCI']) for layer in layers])

CCI_no_patch = np.array([np.median(ERRORS[layer]['no_patch']['CCI'], axis = (1,2,3)) for layer in layers[::-1]])
DE_no_patch = np.array([np.mean(ERRORS[layer]['no_patch']['DE'], axis = (1,2,3)) for layer in layers[::-1]])
Accu_no_patch = np.array([np.mean(ERRORS[layer]['no_patch']['Accu'], axis = (1,2)) for layer in layers[::-1]])
Accu5_no_patch = np.array([np.mean(ERRORS[layer]['no_patch']['Accu5'], axis = (1,2)) for layer in layers[::-1]])
Accu_munscube_no_patch = np.array([np.mean(ERRORS[layer]['no_patch']['Accu_munscube'], axis = (1,2)) for layer in layers[::-1]])
#CCI_no_patch = np.load(CC_dir_path +'training_centered/All_muns/CCI_fost_weighted_no_patch_std.npy')

CCI_no_back = np.array([np.median(ERRORS[layer]['no_back']['CCI'], axis = (1,2,3)) for layer in layers[::-1]])
DE_no_back = np.array([np.mean(ERRORS[layer]['no_back']['DE'], axis = (1,2,3)) for layer in layers[::-1]])
Accu_no_back = np.array([np.mean(ERRORS[layer]['no_back']['Accu'], axis = (1,2)) for layer in layers[::-1]])
Accu5_no_back = np.array([np.mean(ERRORS[layer]['no_back']['Accu5'], axis = (1,2)) for layer in layers[::-1]])
Accu_munscube_no_back = np.array([np.mean(ERRORS[layer]['no_back']['Accu_munscube'], axis = (1,2)) for layer in layers[::-1]])
#CCI_no_back = np.load(CC_dir_path +'training_centered/WCS/finetuning/CCI_fost_weighted_no_back_std.npy')

CCI_wrong_illu = np.array([np.median(ERRORS[layer]['wrong_illu']['CCI'], axis = (1,2,3)) for layer in layers[::-1]])
DE_wrong_illu = np.array([np.mean(ERRORS[layer]['wrong_illu']['DE'], axis = (1,2,3)) for layer in layers[::-1]])
Accu_wrong_illu = np.array([np.mean(ERRORS[layer]['wrong_illu']['Accu'], axis = (1,2)) for layer in layers[::-1]])
Accu5_wrong_illu = np.array([np.mean(ERRORS[layer]['wrong_illu']['Accu5'], axis = (1,2)) for layer in layers[::-1]])
Accu_munscube_wrong_illu = np.array([np.mean(ERRORS[layer]['wrong_illu']['Accu_munscube'], axis = (1,2)) for layer in layers[::-1]])
#CCI_wrong_illu = np.load(CC_dir_path +'training_centered/All_muns/CCI_fost_weighted_wrong_illu_std.npy')

CCI_D65 = np.array([np.nanmedian(ERRORS[layer]['D65']['CCI'], axis = (1,2,3)) for layer in layers[::-1]])
DE_D65 = np.array([np.nanmean(ERRORS[layer]['D65']['DE'], axis = (1,2,3)) for layer in layers[::-1]])
Accu_D65 = np.array([np.nanmean(ERRORS[layer]['D65']['Accu'], axis = (1,2)) for layer in layers[::-1]])
#Accu5_D65 = np.array([np.nanmean(ERRORS['fc2']['D65']['Accu5'], axis = (1,2)) for layer in layers[::-1]])
#Accu_munscube_D65 = np.array([np.mean(ERRORS['fc2']['D65']['Accu_munscube'], axis = (1,2)) for layer in layers[::-1]])

fig = plt.figure(figsize = (6,6))
ax1 = fig.add_subplot(111)
ax1.bar([1,2,3,4,5],[np.mean(Accu_normal, axis = -1)[-1], np.mean(Accu_no_patch,axis = -1)[-1], np.mean(Accu_wrong_illu, axis = -1)[-1], np.mean(Accu_no_back, axis = -1)[-1], np.mean(Accu_D65, axis = -1)[-1]], color = ['k',[0.4,0.7,0.8],[0.4,0.8,0.4],[0.7,0.8,0.4],[0.8,0.4,0.4]],linewidth = 6)
ax1.set_xticks([1,2,3,4,5])
ax1.set_xticklabels([])
plt.xticks(fontsize = 21)
#plt.yticks(fontsize = 14)
plt.yticks(np.arange(0,105,25),fontsize = 21)
ax1.set_ylabel('top1 Accuracy',fontsize = 25)
ax1.set_xticklabels(['CC$_{normal}$','CC$_{no patch}$','CC$_{wrong back}$','CC$_{no back}$','D65$_{normal}$'], rotation = 45)
fig.tight_layout()
fig.savefig(figures_dir_path + 'Accuracy.png', dpi=400)
plt.show()

fig = plt.figure(figsize = (6,6))
ax1 = fig.add_subplot(111)
ax1.bar([1,2,3,4,5],[np.mean(Accu5_normal, axis = -1)[-1], np.mean(Accu5_no_patch,axis = -1)[-1], np.mean(Accu5_wrong_illu, axis = -1)[-1], np.mean(Accu5_no_back, axis = -1)[-1], 6], color = ['k',[0.4,0.7,0.8],[0.4,0.8,0.4],[0.7,0.8,0.4],[0.8,0.4,0.4]],linewidth = 6)
ax1.set_xticks([1,2,3,4,5])
ax1.set_xticklabels([])
plt.xticks(fontsize = 21)
#plt.yticks(fontsize = 14)
plt.yticks(np.arange(0,105,25),fontsize = 21)
ax1.set_ylabel('top5 Accuracy',fontsize = 25)
ax1.set_xticklabels(['CC$_{normal}$','CC$_{no patch}$','CC$_{wrong back}$','CC$_{no back}$','D65$_{normal}$'], rotation = 45)
fig.tight_layout()
fig.savefig(figures_dir_path + 'Accuracy5.png', dpi=400)
plt.show()

fig = plt.figure(figsize = (6,6))
ax1 = fig.add_subplot(111)
ax1.bar([1,2,3,4,5],[np.mean(Accu_munscube_normal, axis = -1)[-1], np.mean(Accu_munscube_no_patch,axis = -1)[-1], np.mean(Accu_munscube_wrong_illu, axis = -1)[-1], np.mean(Accu_munscube_no_back, axis = -1)[-1], 9], color = ['k',[0.4,0.7,0.8],[0.4,0.8,0.4],[0.7,0.8,0.4],[0.8,0.4,0.4]],linewidth = 6)
ax1.set_xticks([1,2,3,4,5])
ax1.set_xticklabels([])
plt.xticks(fontsize = 21)
#plt.yticks(fontsize = 14)
plt.yticks(np.arange(0,105,25),fontsize = 21)
ax1.set_ylabel('Accuracy MunsCube',fontsize = 25)
ax1.set_xticklabels(['CC$_{normal}$','CC$_{no patch}$','CC$_{wrong back}$','CC$_{no back}$','D65$_{normal}$'], rotation = 45)
fig.tight_layout()
fig.savefig(figures_dir_path + 'Accuracy_munscube.png', dpi=400)
plt.show()

fig = plt.figure(figsize = (6,6))
ax1 = fig.add_subplot(111)
ax1.bar([1,2,3,4,5, 6],[np.mean(DE_normal, axis = -1)[-1], np.mean(DE_no_patch,axis = -1)[-1], np.mean(DE_wrong_illu, axis = -1)[-1], np.mean(DE_no_back, axis = -1)[-1], np.mean(DE_D65, axis = -1)[-1], predicted_errors['DE'].mean()],color = ['k',[0.4,0.7,0.8],[0.4,0.8,0.4],[0.7,0.8,0.4],[0.8,0.4,0.4],'grey'],linewidth = 6)
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
ax1.bar([1,2,3,4,5,6],[np.mean(CCI_normal, axis = -1)[-1], np.mean(CCI_no_patch,axis = -1)[-1], np.mean(CCI_wrong_illu, axis = -1)[-1], np.mean(CCI_no_back, axis = -1)[-1], np.mean(CCI_D65, axis = -1)[-1], predicted_errors['CCI'].mean()],color = ['k',[0.4,0.7,0.8],[0.4,0.8,0.4],[0.7,0.8,0.4],[0.8,0.4,0.4],'grey'],linewidth = 6)
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



dis.DEFINE_PLT_RC(type = 1)

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.errorbar([1,2,3,4,5], np.mean(Accu5_normal, axis = -1), yerr = np.std(Accu5_normal, axis = -1),
             color = 'k', label = 'Top5$_{normal}$')
ax1.errorbar([1,2,3,4,5],np.mean(Accu5_no_patch, axis = -1), yerr = np.std(Accu5_no_patch, axis = -1),
             color = [0.4,0.7,0.8], label = 'Top5$_{nopatch}$')
ax1.set_xlabel('Readouts')
ax1.set_xticks([1,2,3,4,5])
ax1.set_xticklabels(['RC1','RC2','RC3','RF1','RF2'])
ax1.set_ylabel('Top5 Accuracy')
plt.legend()
fig.tight_layout()
fig.savefig(figures_dir_path + 'Accu5_readout.png', dpi=400)
plt.show()

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.errorbar([1,2,3,4,5],np.mean(CCI_normal, axis = -1), yerr = np.std(CCI_normal, axis = -1),
             color = 'k', label = 'CC$_{normal}$')
ax1.errorbar([1,2,3,4,5],np.mean(CCI_no_patch, axis = -1), yerr = np.std(CCI_no_patch, axis = -1),
             color = [0.4,0.7,0.8], label = 'CC$_{nopatch}$')
ax1.set_xlabel('Readouts')
ax1.set_xticks([1,2,3,4,5])
ax1.set_xticklabels(['RC1','RC2','RC3','RF1','RF2'])
ax1.set_ylabel('Median CCI')
plt.legend()
fig.tight_layout()
fig.savefig(figures_dir_path + 'CCI_fost_readout.png', dpi=400)
plt.show()

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.errorbar([1,2,3,4,5], np.mean(Accu_munscube_normal, axis = -1), yerr = np.std(Accu_munscube_normal, axis = -1),
             color = 'k', label = 'MunsCube$_{normal}$')
ax1.errorbar([1,2,3,4,5],np.mean(Accu_munscube_no_patch, axis = -1), yerr = np.std(Accu_munscube_no_patch, axis = -1),
             color = [0.4,0.7,0.8], label = 'MunsCube$_{nopatch}$')
ax1.set_xlabel('Readouts')
ax1.set_xticks([1,2,3,4,5])
ax1.set_xticklabels(['RC1','RC2','RC3','RF1','RF2'])

ax1.set_ylabel('Munscube Accuracy')
plt.legend()
fig.tight_layout()
fig.savefig(figures_dir_path + 'Munscube_readout.png', dpi=400)
plt.show()


diff = np.absolute(((1 - CCI_normal) - (1 - CCI_no_patch)))/ (1 - CCI_no_patch)

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.errorbar([1,2,3,4,5], np.mean(diff, axis = -1), np.std(diff, axis = -1), color = 'k', ls = '-')
ax1.set_xlabel('Readouts')
ax1.set_xticks([1,2,3,4,5])
ax1.set_xticklabels(['RC1','RC2','RC3','RF1','RF2'])
#plt.yticks(np.arange(-1,1.1,0.5),fontsize = 14)
ax1.set_ylabel('Relative difference')
fig.tight_layout()
fig.savefig(figures_dir_path + 'CCI_fost_readout_diff.png', dpi=400)
plt.show()

np.mean((CCI_normal - CCI_no_patch)/CCI_normal, axis = -1)
(CCI_normal - CCI_no_patch)/CCI_normal

