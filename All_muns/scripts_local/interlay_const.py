#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 11:01:25 2019

@author: alban
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 11:13:59 2018

@author: alban
"""

# Script that computes correlations in activation between munsells, dissimilarity matrices and computes MDS

import numpy as np
import matplotlib.pyplot as plt
import sklearn
import pickle
import scipy.io as sio
import scipy


# In[9]: FUNCTIONS

def compute_munsells(EVAL,WCS_X,WCS_Y):
    WCS_MAT = np.zeros((len(EVAL),10,41))
    for j in range(len(EVAL)):
        for i in range(len(EVAL[0])):
            WCS_MAT[j,WCS_X[i],WCS_Y[i]] = EVAL[j][i]
    return WCS_MAT

def display_munsells(WCS_MAT, norm):
    WCS_MAT = (WCS_MAT/norm)
    if WCS_MAT.shape == (10,41):
        WCS_MAT_achro = WCS_MAT[:,0]
        WCS_MAT_chro = WCS_MAT[1:-1,1:]
        # definitions for the axes
        left, width = 0.05, 0.8
        bottom, height = 0.1, 0.8
        left_h = left + width + 0.05
        bottom_h = 0
        
        rect_chro = [left, bottom, width, height]
        rect_achro = [left_h, bottom_h, 0.8/40, 1]
        
        fig = plt.figure(1, figsize=(8, 3))
        
        axchro = plt.axes(rect_chro)
        axachro = plt.axes(rect_achro)
        
        # the scatter plot:
        axchro.imshow(np.stack((WCS_MAT_chro,WCS_MAT_chro,WCS_MAT_chro),axis = 2))
        axachro.imshow(np.stack((WCS_MAT_achro,WCS_MAT_achro,WCS_MAT_achro),axis = 2))
        #axchro.set_xticks((np.arange(0,40)))
        #axchro.set_yticks((np.arange(1,9)))
        axachro.set_xticks([])
        #axachro.set_yticks((np.arange(0,10)))
        
        axchro.set_ylabel('Value',fontsize = 15)
        axchro.set_xlabel('Hue',fontsize = 15)
        #plt.setp(axchro.set_ylabel('Tuning |elevation| (deg)',fontsize = 25))
        plt.setp(axchro.get_xticklabels(), fontsize=12)
        plt.setp(axchro.get_yticklabels(), fontsize=12)
        
        #axchro.set_xlim((-45, 315))
        
        #fig.text(0.5, 0.02, 'Tuning azimuth (deg)', ha='center',fontsize = 25)
        
        #fig.savefig('mod_N2/control2_lay1_tuning.png')
        plt.show()

    else:
        fig = plt.figure(figsize = (8,3))
        plt.imshow(np.stack((WCS_MAT,WCS_MAT,WCS_MAT),axis = 2))
        plt.xlabel('Munsell hue',fontsize = 15)
        plt.ylabel('Munsell value',fontsize = 15)
        #plt.title('Munsell chips WCS',fontsize = 18)
        #fig.savefig('Munsell_chips_WCS.png')
        fig.tight_layout
        plt.show()

def display_munsells_inv(WCS_MAT, norm):
    WCS_MAT2 = (WCS_MAT/norm)
    WCS_MAT = np.zeros(WCS_MAT2.shape)
    if WCS_MAT.shape == (10,41):
        for i in range(len(WCS_MAT2)):
            WCS_MAT[i] = WCS_MAT2[9-i]
        WCS_MAT_achro = WCS_MAT[:,0].reshape(10,1)
        WCS_MAT_chro = WCS_MAT[1:-1,1:]
        # definitions for the axes
        left, width = 0.05, 0.8
        bottom, height = 0.1, 0.8
        left_h = left + width + 0.05
        bottom_h = 0
        
        rect_chro = [left, bottom, width, height]
        rect_achro = [left_h, bottom_h, 0.8/40, 1]
        
        fig = plt.figure(1, figsize=(8, 3))
        
        axchro = plt.axes(rect_chro)
        axachro = plt.axes(rect_achro)
        
        # the scatter plot:
        axchro.imshow(np.stack((WCS_MAT_chro,WCS_MAT_chro,WCS_MAT_chro),axis = 2))
        axachro.imshow(np.stack((WCS_MAT_achro,WCS_MAT_achro,WCS_MAT_achro),axis = 2))
        #axchro.set_xticks((np.arange(0,40)))
        #axchro.set_yticks((np.arange(1,9)))
        axachro.set_xticks([])
        #axachro.set_yticks((np.arange(0,10)))
        
        axchro.set_ylabel('Value',fontsize = 15)
        axchro.set_xlabel('Hue',fontsize = 15)
        #plt.setp(axchro.set_ylabel('Tuning |elevation| (deg)',fontsize = 25))
        plt.setp(axchro.get_xticklabels(), fontsize=12)
        plt.setp(axchro.get_yticklabels(), fontsize=12)
        
        #axchro.set_xlim((-45, 315))
        
        #fig.text(0.5, 0.02, 'Tuning azimuth (deg)', ha='center',fontsize = 25)
        
        #fig.savefig('mod_N2/control2_lay1_tuning.png')
        plt.show()


    else:
        for i in range(len(WCS_MAT2)):
            WCS_MAT[i] = WCS_MAT2[7-i]
        fig = plt.figure(figsize = (8,3))
        plt.imshow(np.stack((WCS_MAT,WCS_MAT,WCS_MAT),axis = 2))
        plt.xlabel('Munsell hue',fontsize = 15)
        plt.ylabel('Munsell value',fontsize = 15)
        #plt.title('Munsell chips WCS',fontsize = 18)
        #fig.savefig('Munsell_chips_WCS.png')
        fig.tight_layout
        plt.show()

def XYZ2sRGB(XYZ):
    Trans = np.array([[3.24045, -1.537138, -0.49853],[-0.9692660, 1.8760108, 0.0415560],[0.0556434, -0.2040259, 1.0572252]])
    sRGB = np.dot(XYZ,Trans.T)   
    return sRGB


def princomp(A):
 """ performs principal components analysis 
     (PCA) on the n-by-p data matrix A
     Rows of A correspond to observations, columns to variables. 

 Returns :  
  coeff :
    is a p-by-p matrix, each column containing coefficients 
    for one principal component.
  score : 
    the principal component scores; that is, the representation 
    of A in the principal component space. Rows of SCORE 
    correspond to observations, columns to components.
  latent : 
    a vector containing the eigenvalues 
    of the covariance matrix of A.
 """
 # computing eigenvalues and eigenvectors of covariance matrix
 M = (A-np.mean(A.T,axis=1)).T # subtract the mean (along columns)
 [latent,coeff] = np.linalg.eig(np.cov(M)) # attention:not always sorted
 sortedIdx = np.argsort(-latent)
 latent = latent[sortedIdx]
 explained = 100*latent/np.sum(latent)
 score = np.dot(coeff.T,M) # projection of the data in the new space
 coeff = coeff[:,sortedIdx]
 score = score[sortedIdx,:]
 return coeff,score,latent, explained

def correlations_layers(conv1):
    mean_conv1 = np.mean(conv1,axis = 2)
    std_conv1 = np.std(conv1,axis = (0,2))
    CORR = np.zeros((len(conv1), conv1.shape[1],conv1.shape[1]))
    for i in range(len(conv1)):
        CORR[i] = np.corrcoef(mean_conv1[i])
    return CORR

def scatter_muns(RESULT,title,path1,path2,RGB, LABELS = ['DIM 1','DIM 2','DIM 2']):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #ax.scatter(RESULT[10:,0], RESULT[10:,1], RESULT[10:,2], marker='o',c = WCS_MAT_sRGB[1:-1,1:].reshape((-1,3)))
    ax.scatter(RESULT[:,0], RESULT[:,1], RESULT[:,2], marker='o', c = RGB)
    ax.set_xlabel(LABELS[0],fontsize = 15)
    ax.set_ylabel(LABELS[1],fontsize = 15)
    ax.set_zlabel(LABELS[2],fontsize = 15)
    fig.text(0.5,0.95,title,ha='center',fontsize = 18)
    fig.tight_layout()
    plt.show()
    fig.savefig(path1,format='png', dpi=1200)
    plt.close()
    
    fig = plt.figure(figsize = (9,3))
    ax1 = fig.add_subplot(131)
    #ax1.scatter(RESULT[10:,0], RESULT[10:,1], marker='o',c = WCS_MAT_sRGB[1:-1,1:].reshape((-1,3)))
    ax1.scatter(RESULT[:,0], RESULT[:,1], marker='o', c = RGB)
    ax1.set_xlabel(LABELS[0],fontsize = 15)
    ax1.set_ylabel(LABELS[1],fontsize = 15)
    ax2 = fig.add_subplot(132)
    #ax2.scatter(RESULT[10:,0], RESULT[10:,2], marker='o',c = WCS_MAT_sRGB[1:-1,1:].reshape((-1,3)))
    ax2.scatter(RESULT[:,0], RESULT[:,2], marker='o', c = RGB)
    ax2.set_xlabel(LABELS[0],fontsize = 15)
    ax2.set_ylabel(LABELS[2],fontsize = 15)
    ax3 = fig.add_subplot(133)
    #ax3.scatter(RESULT[10:,1], RESULT[10:,2], marker='o',c = WCS_MAT_sRGB[1:-1,1:].reshape((-1,3)))
    ax3.scatter(RESULT[:,1], RESULT[:,2], marker='o', c = RGB)
    ax3.set_xlabel(LABELS[1],fontsize = 15)
    ax3.set_ylabel(LABELS[2],fontsize = 15)
    fig.text(0.5,0.94,title,ha='center',fontsize = 18)
    fig.tight_layout()
    plt.show()
    fig.savefig(path2,format='png', dpi=1200)
    plt.close()

def MDS(D):
    """                                                                                       
    Classical multidimensional scaling (MDS)                                                  
                                                                                               
    Parameters                                                                                
    ----------                                                                                
    D : (n, n) array                                                                          
        Symmetric distance matrix.                                                            
                                                                                               
    Returns                                                                                   
    -------                                                                                   
    Y : (n, p) array                                                                          
        Configuration matrix. Each column represents a dimension. Only the                    
        p dimensions corresponding to positive eigenvalues of B are returned.                 
        Note that each dimension is only determined up to an overall sign,                    
        corresponding to a reflection.                                                        
                                                                                               
    e : (n,) array                                                                            
        Eigenvalues of B.                                                                     
                                                                                               
    """
    # Number of points                                                                        
    n = len(D)
 
    # Centering matrix                                                                        
    H = np.eye(n) - np.ones((n, n))/n
 
    # YY^T                                                                                    
    B = -H.dot(D**2).dot(H)/2
 
    # Diagonalize                                                                             
    evals, evecs = np.linalg.eigh(B)
 
    # Sort by eigenvalue in descending order                                                  
    idx   = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:,idx]
 
    # Compute the coordinates using positive-eigenvalued components only                      
    w, = np.where(evals > 0)
    L  = np.diag(np.sqrt(evals[w]))
    V  = evecs[:,w]
    Y  = V.dot(L)
 
    return Y, evals

def rotate_via_numpy(xy, radians):
    """Use numpy to build a rotation matrix and take the dot product."""
    c, s = np.cos(radians), np.sin(radians)
    j = np.matrix([[c, s], [-s, c]])
    m = np.dot(j, xy.T)
    return m

# In[9]: LOAD ACTIVATIONS color constant

conv1 = np.load('All_muns/conv1_muns_illu.npy')
conv2 = np.load('All_muns/conv2_muns_illu.npy')
conv3 = np.load('All_muns/conv3_muns_illu.npy')
fc1 = np.load('All_muns/fc1_muns_illu.npy')
fc2 = np.load('All_muns/fc2_muns_illu.npy')
eval = np.load('All_muns/evaluation_muns_illu.npy')

np.mean(eval)

L = list()
with open("WCS_indx.txt") as f:
    for line in f:
       L.append(line.split())
      
WCS_X = [ord(char[0][0].lower()) - 97 for char in L]
WCS_Y = [int(char[0][1:]) for char in L]

WCS_X1 = [9-x for x in WCS_X]

CHROMA = list()
with open("WCS_chroma.txt") as f:
    for line in f:
       CHROMA.append(int(line.split()[0])) 

VALUE = list()
with open("WCS_val.txt") as f:
    for line in f:
       VALUE.append(int(line.split()[0])) 

CHROMA_arr = np.zeros(len(CHROMA))
VALUE_arr = np.zeros(len(CHROMA))
X_muns = np.zeros(len(CHROMA)) 
Y_muns = np.zeros(len(CHROMA))
 
HUE_arr = np.zeros(len(CHROMA))
Hue_arr = np.arange(0,360,360/40)

for i in range(len(CHROMA)):
    CHROMA_arr[i] = CHROMA[i]/2
    VALUE_arr[i] = VALUE[i]/10
    if WCS_Y[i] == 0:
        HUE_arr[i] = 0
    else:
        HUE_arr[i] = (Hue_arr[WCS_Y[i]-1])*np.pi/180
    X_muns[i] = CHROMA_arr[i]*np.cos(HUE_arr[i])
    Y_muns[i] = CHROMA_arr[i]*np.sin(HUE_arr[i])
    

with open("XYZ_WCS.txt", "rb") as fp:   #Pickling
    XYZ_WCS = pickle.load(fp)

RGB_muns = [XYZ2sRGB(XYZ) for XYZ in XYZ_WCS]
WCS_MAT_RGB = np.zeros((10,41,3))
count = 0
for i in range(len(RGB_muns)):
    WCS_MAT_RGB[WCS_X[i],WCS_Y[i]] = RGB_muns[count]
    count +=1
    
plt.imshow(((WCS_MAT_RGB[:,0]-5.2)/(88.3-5.2)).reshape((1,10,3)))
plt.show()

WCS_MAT_sRGB = (WCS_MAT_RGB - np.amin(WCS_MAT_RGB))/(np.amax(WCS_MAT_RGB)-np.amin(WCS_MAT_RGB))

RGB_muns = (RGB_muns - np.amin(RGB_muns))/(np.amax(RGB_muns)-np.amin(RGB_muns))


plt.imshow((WCS_MAT_sRGB))
plt.show()

general_mat_consistency = sio.loadmat('/home/alban/Documents/pytorch/project_color_constancy/WCS_Xp/general_consensusconsistency_map4Alban.mat')['cmap_general']
general_mat_consistency2 = np.zeros(general_mat_consistency[:-3,:].shape)
for i in range(len(general_mat_consistency2)):
    general_mat_consistency2[i] = general_mat_consistency[:-3,:][7-i]

display_munsells(general_mat_consistency2,np.amax(general_mat_consistency2))


# In[9]: measure of colo constancy per munsell
# As a measure of color constancy. we have the ratio of the representational dispertion with ditance between adjacent
# chips. representational dispertion = sqrt(sum(var))

def constancy_measure(layer,WCS_X,WCS_Y):
    mean_layer = np.mean(layer,axis = 2)
    MAT_mean_layer = np.zeros((10,10,41,layer.shape[-1]))
    MAT_disp_layer = np.zeros((10,10,41))
    for i in range(330):
        MAT_mean_layer[:,WCS_X[i],WCS_Y[i]] = mean_layer[:,i]
        MAT_disp_layer[:,WCS_X[i],WCS_Y[i]] = np.sqrt(np.sum(np.var(layer[:,i],axis =-2),axis = -1))
    
    
    MAT_mean_layer_color = MAT_mean_layer[:,1:-1,1:]
    MAT_disp_layer_color = MAT_disp_layer[:,1:-1,1:]
    
    D_MAT_mean_layer = np.zeros((10,8,40))
    for i in range(D_MAT_mean_layer.shape[1]):
        for j in range(D_MAT_mean_layer.shape[2]):
            d1 = np.sqrt(np.sum((MAT_mean_layer_color[:,i,j]-MAT_mean_layer_color[:,(i+1)%8,j])**2,axis = 1))
            d2 = np.sqrt(np.sum((MAT_mean_layer_color[:,i,j]-MAT_mean_layer_color[:,(i-1)%8,j])**2,axis = 1))
            d3 = np.sqrt(np.sum((MAT_mean_layer_color[:,i,j]-MAT_mean_layer_color[:,i,(j+1)%40])**2,axis = 1))
            d4 = np.sqrt(np.sum((MAT_mean_layer_color[:,i,j]-MAT_mean_layer_color[:,i,(j-1)%40])**2,axis = 1))
            if i == 0:
                d2 = np.zeros(10)
                d2[:] = np.nan
            elif i == 7:
                d1 = np.zeros(10)
                d1[:] = np.nan
            D_MAT_mean_layer[:,i,j] = np.nanmean((d1,d2,d3,d4), axis = 0)
    
    constancy_layer = MAT_disp_layer_color/D_MAT_mean_layer
    return D_MAT_mean_layer,constancy_layer


D_MAT_mean_conv1,constancy_conv1 = constancy_measure(conv1,WCS_X,WCS_Y)
D_MAT_mean_conv2,constancy_conv2 = constancy_measure(conv2,WCS_X,WCS_Y)
D_MAT_mean_conv3,constancy_conv3 = constancy_measure(conv3,WCS_X,WCS_Y)
D_MAT_mean_fc1,constancy_fc1 = constancy_measure(fc1,WCS_X,WCS_Y)
D_MAT_mean_fc2,constancy_fc2 = constancy_measure(fc2,WCS_X,WCS_Y)

display_munsells_inv(np.mean(constancy_conv1,axis = 0),np.amax(constancy_conv1))
display_munsells_inv(np.mean(constancy_conv2,axis = 0),np.amax(constancy_conv2))
display_munsells_inv(np.mean(constancy_conv3,axis = 0),np.amax(constancy_conv3))
display_munsells_inv(np.mean(constancy_fc1,axis = 0),np.amax(constancy_fc1))
display_munsells_inv(np.mean(constancy_fc2,axis = 0),np.amax(constancy_fc2))

CHROMA = list()
with open("WCS_chroma.txt") as f:
    for line in f:
       CHROMA.append(int(line.split()[0]))    



WCS_MAT_CHROMA = np.zeros((10,41))
count = 0
for i in range(330):
    WCS_MAT_CHROMA[WCS_X[i],WCS_Y[i]] = float(CHROMA[count])
    count +=1
display_munsells_inv(WCS_MAT_CHROMA,16)

np.corrcoef(WCS_MAT_CHROMA[1:-1,1:].flatten(),np.mean(constancy_fc2,axis = 0).flatten())

print(np.mean(constancy_conv1),np.mean(constancy_conv2),np.mean(constancy_conv3),np.mean(constancy_fc1),np.mean(constancy_fc2))
CC_interlay = np.array([np.mean(constancy_conv1),np.mean(constancy_conv2),np.mean(constancy_conv3),np.mean(constancy_fc1),np.mean(constancy_fc2)])


fig = plt.figure(figsize = (9,3))
ax1 = fig.add_subplot(111)
ax1.plot([1,2,3,4,5],CC_interlay,'k',lw =2)
ax1.set_xlabel('Layer',fontsize = 18)
ax1.set_xticks([1,2,3,4,5])
ax1.set_ylabel('Color constancy index',fontsize = 15)
#ax1.set_ylabel('Layer',fontsize = 15)
#fig.text(0.5,0.94,title,ha='center',fontsize = 18)
fig.tight_layout()
plt.show()
#fig.savefig(path2,format='png', dpi=1200)



# In[9]: idx of neuron dynamics: codes more munsell changes or illuminant changes?
# As an idx of the neurons dynamics, we take the ratio of 

def idx_dyn(layer):
    layer_illu_std = np.mean(np.std(layer,axis = -2), axis = 1)
    layer_chips_std = np.mean(np.std(layer,axis = -3),axis = 1)
    layer_all_std = np.std(layer,axis = (-2,-3))
    layer_mean = np.mean(layer,axis = (-2,-3))
    layer_mean[layer_mean==0] = np.nan
    
    layer_illu_std_norm = layer_illu_std/layer_mean
    layer_chips_std_norm = layer_chips_std/layer_mean
    layer_all_std_norm = layer_all_std/layer_mean
    
    idx_dyn_layer = layer_illu_std_norm/layer_chips_std_norm
    idx_illu_layer = layer_illu_std_norm/layer_all_std_norm
    idx_muns_layer = layer_chips_std_norm/layer_all_std_norm
    return np.round(idx_dyn_layer,2),np.round(idx_illu_layer,2),np.round(idx_muns_layer,2)

idx_dyn_conv1,idx_illu_conv1,idx_muns_conv1 = idx_dyn(conv1)
idx_dyn_conv2,idx_illu_conv2,idx_muns_conv2 = idx_dyn(conv2)
idx_dyn_conv3,idx_illu_conv3,idx_muns_conv3 = idx_dyn(conv3)
idx_dyn_fc1,idx_illu_fc1,idx_muns_fc1 = idx_dyn(fc1)
idx_dyn_fc2,idx_illu_fc2,idx_muns_fc2 = idx_dyn(fc2)

np.nanmean(idx_dyn_conv1)
np.nanmean(idx_dyn_conv2)
np.nanmean(idx_dyn_conv3)
np.nanmean(idx_dyn_fc1)
np.nanmean(idx_dyn_fc2)

np.sum(np.sum(fc1,axis = (1,-2))==0 ,axis = -1)
np.sum(np.sum(fc2,axis = (1,-2))==0 ,axis = -1)
np.sum(np.sum(conv1,axis = (1,-2))==0 ,axis = -1)
np.sum(np.sum(conv2,axis = (1,-2))==0 ,axis = -1)
np.sum(np.sum(conv3,axis = (1,-2))==0 ,axis = -1)


fig = plt.figure()
ax1 = fig.add_subplot(231)
x = idx_dyn_conv1[~np.isnan(idx_dyn_conv1)].flatten()
h = np.histogram(x,bins = np.arange(0,1.05,1.0/20),density = True)
ax1.bar(h[1][:-1]-(h[1][1]-h[1][0])/2,h[0],width = (h[1][1]-h[1][0]))

ax2 = fig.add_subplot(232,sharex=ax1)
x = idx_dyn_conv2[~np.isnan(idx_dyn_conv2)].flatten()
h = np.histogram(x,bins = np.arange(0,1.05,1.0/20),density = True)
ax2.bar(h[1][:-1]-(h[1][1]-h[1][0])/2,h[0],width = (h[1][1]-h[1][0]))

ax3 = fig.add_subplot(233, sharex=ax1, sharey=ax2)
x = idx_dyn_conv3[~np.isnan(idx_dyn_conv3)].flatten()
h = np.histogram(x,bins = np.arange(0,1.05,1.0/20),density = True)
ax3.bar(h[1][:-1]-(h[1][1]-h[1][0])/2,h[0],width = (h[1][1]-h[1][0]))

ax4 = fig.add_subplot(234, sharey=ax1)
x = idx_dyn_fc1[~np.isnan(idx_dyn_fc1)].flatten()
h = np.histogram(x,bins = np.arange(0,1.05,1.0/20),density = True)
ax4.bar(h[1][:-1]-(h[1][1]-h[1][0])/2,h[0],width = (h[1][1]-h[1][0]))

ax5 = fig.add_subplot(235, sharex=ax4, sharey=ax2)
x = idx_dyn_fc2[~np.isnan(idx_dyn_fc2)].flatten()
h = np.histogram(x,bins = np.arange(0,1.05,1.0/20),density = True)
ax5.bar(h[1][:-1]-(h[1][1]-h[1][0])/2,h[0],width = (h[1][1]-h[1][0]))

ax1.set_xticks([])
ax2.set_yticks([])
fig.text(0.5, 0.04, 'Dynamicity index', ha='center', va='center',fontsize = 14)
fig.text(0.06, 0.5, 'Density', ha='center', va='center', rotation='vertical',fontsize = 14)
#fig.tight_layout()
plt.show() 
#fig.savefig('tuning_curves/BVLC: Responsivity (normalized)', dpi = 400)

IDX_DYN = np.concatenate((idx_dyn_conv1, idx_dyn_conv2, idx_dyn_conv3, idx_dyn_fc1, idx_dyn_fc2),axis = 1)


ACTIVATIONS = np.concatenate((conv1, conv2, conv3, fc1, fc2),axis = -1)

tresh = np.nanpercentile(IDX_DYN, 75)
tresh_muns = np.nanpercentile(IDX_DYN, 25)

INDX = (np.isfinite(idx_dyn_fc1)) & (idx_dyn_fc1 > 0.75)
INDX_muns_spe = (np.isfinite(IDX_DYN)) & (IDX_DYN < tresh)

REP_illu = np.moveaxis(np.mean(fc1, axis = 1),1,-1)[INDX]

CORR_MAT_ILLU = np.corrcoef(REP_illu.T)
D_MAT_ILLU = 1 - CORR_MAT_ILLU

RESULT_MDS, stress_MDS  = MDS(D_MAT_ILLU)
#coeff1,score1,latent1,explained1 = princomp(np.mean(conv1,axis = (0,2)))

RGB_illu = np.load('RGB_illu.npy')
idx_illu = np.load('indx_illu_np.npy')

RGB_illu[[idx_illu]]

#scatter_muns(RESULT_MDS[:,:3],'Illuminant','All_muns/Illuminant_MDS_3D_classic.png','All_muns/Illuminant_MDS_2D_classic.png', RGB_illu[[idx_illu]])

COOR_ILLU_2D = RESULT_MDS[:,:3][:,:2]
COOR_ILLU_2D = rotate_via_numpy(COOR_ILLU_2D,np.pi*240/180)

fig = plt.figure(figsize = (3,3))
ax1 = fig.add_subplot(111)
#ax1.scatter(RESULT[10:,0], RESULT[10:,1], marker='o',c = WCS_MAT_sRGB[1:-1,1:].reshape((-1,3)))
ax1.scatter(COOR_ILLU_2D[:,0], COOR_ILLU_2D[:,1], marker='o', c = RGB_illu[[idx_illu]])
ax1.set_xlabel('DIM 1',fontsize = 15)
ax1.set_ylabel('DIM 2',fontsize = 15)
fig.savefig('/home/alban/Documents/pytorch/project_color_constancy/WCS_Xp/training_centered/POSTER_VSS_19/MDS_illu.png',format='png', dpi=1200)
plt.show()
plt.close()
