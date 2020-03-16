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
    for i in range(len(WCS_MAT2)):
        WCS_MAT[i] = WCS_MAT2[9-i]
    if WCS_MAT.shape == (10,41):
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

def scatter_muns2(RESULT,title,path1,path2, LABELS = ['DIM 1','DIM 2','DIM 3']):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(RESULT[:,0], RESULT[:,1], RESULT[:,2], marker='o',c = RGB_muns)
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
    ax1.scatter(RESULT[:,0], RESULT[:,1], marker='o',c = RGB_muns)
    ax1.set_xlabel(LABELS[0],fontsize = 15)
    ax1.set_ylabel(LABELS[1],fontsize = 15)
    ax2 = fig.add_subplot(132)
    ax2.scatter(RESULT[:,0], RESULT[:,2], marker='o',c = RGB_muns)
    ax2.set_xlabel(LABELS[0],fontsize = 15)
    ax2.set_ylabel(LABELS[2],fontsize = 15)
    ax3 = fig.add_subplot(133)
    ax3.scatter(RESULT[:,1], RESULT[:,2], marker='o',c = RGB_muns)
    ax3.set_xlabel(LABELS[1],fontsize = 15)
    ax3.set_ylabel(LABELS[2],fontsize = 15)
    fig.text(0.5,0.94,title,ha='center',fontsize = 18)
    fig.tight_layout()
    plt.show()
    fig.savefig(path2,format='png', dpi=1200)
    plt.close()


def scatter_muns(RESULT,title,path1,path2, LABELS = ['DIM 1','DIM 2','DIM 3']):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(RESULT[:,0], RESULT[:,1], RESULT[:,2], marker='o',c = RGB_muns)
    ax.set_xlabel(LABELS[0],fontsize = 15)
    ax.set_ylabel(LABELS[1],fontsize = 15)
    ax.set_zlabel(LABELS[2],fontsize = 15)
    fig.text(0.5,0.95,title,ha='center',fontsize = 18)
    fig.tight_layout()
    plt.show()
    fig.savefig(path1,format='png', dpi=1200)
    plt.close()
    
    fig = plt.figure(figsize = (6,3))
    ax1 = fig.add_subplot(121)
    ax1.scatter(RESULT[:,0], RESULT[:,1], marker='o',c = RGB_muns)
    ax1.set_xlabel(LABELS[0],fontsize = 15)
    ax1.set_ylabel(LABELS[1],fontsize = 15)
    ax2 = fig.add_subplot(122)
    ax2.scatter(RESULT[:,1], RESULT[:,2], marker='o',c = RGB_muns)
    ax2.set_xlabel(LABELS[1],fontsize = 15)
    ax2.set_ylabel(LABELS[2],fontsize = 15)
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



# In[9]: LOAD ACTIVATIONS color constant

conv1 = np.load('/home/alban/mnt/awesome/alban/project_color_constancy/PYTORCH/WCS/training_centered/All_muns/conv1_muns_illu.npy')
conv2 = np.load('/home/alban/mnt/awesome/alban/project_color_constancy/PYTORCH/WCS/training_centered/All_muns/conv2_muns_illu.npy')
conv3 = np.load('/home/alban/mnt/awesome/alban/project_color_constancy/PYTORCH/WCS/training_centered/All_muns/conv3_muns_illu.npy')
fc1 = np.load('/home/alban/mnt/awesome/alban/project_color_constancy/PYTORCH/WCS/training_centered/All_muns/fc1_muns_illu.npy')
fc2 = np.load('/home/alban/mnt/awesome/alban/project_color_constancy/PYTORCH/WCS/training_centered/All_muns/fc2_muns_illu.npy')

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




# In[9]: CORRELATIONs



corr_conv1 = correlations_layers(conv1)
corr_conv2 = correlations_layers(conv2)
corr_conv3 = correlations_layers(conv3)
corr_fc1 = correlations_layers(fc1)
corr_fc2 = correlations_layers(fc2)



   
mean_conv1 = np.mean(conv1,axis = 2)
std_conv1 = np.std(conv1,axis = 2)




np.mean(std_conv1/mean_conv1,axis = -1)

mean_conv2 = np.mean(conv2,axis = 2)+ 0.00001
std_conv2 = np.std(conv2,axis = 2)

np.mean(std_conv2/mean_conv2,axis = -1)

mean_conv3 = np.mean(conv3,axis = 2) + 0.00001
std_conv3 = np.std(conv3,axis = 2)

np.mean(std_conv3/mean_conv3,axis = -1)

mean_fc1 = np.mean(fc1,axis = 2) + 0.00001
std_fc1 = np.std(fc1,axis = 2)

np.mean(std_fc1/mean_fc1,axis = -1)

mean_fc2 = np.mean(fc2,axis = 2) + 0.00001
std_fc2 = np.std(fc2,axis = 2)

np.mean(std_fc2/mean_fc2,axis = -1)

len(np.mean(std_conv1,axis = -1))

M_conv1 = compute_munsells(np.mean(std_conv1,axis = -1),WCS_X,WCS_Y)
display_munsells_inv(np.mean(M_conv1,axis = 0),np.amax(M_conv1))

M_conv2 = compute_munsells(np.mean(std_conv2,axis = -1),WCS_X,WCS_Y)
display_munsells_inv(np.mean(M_conv2,axis = 0),np.amax(M_conv2))

M_conv3 = compute_munsells(np.mean(std_conv3,axis = -1),WCS_X,WCS_Y)
display_munsells_inv(np.mean(M_conv3,axis = 0),np.amax(M_conv3))

M_fc1 = compute_munsells(np.mean(std_fc1,axis = -1),WCS_X,WCS_Y)
display_munsells_inv(np.mean(M_fc1,axis = 0),np.amax(M_fc1))

M_fc2 = compute_munsells(np.mean(std_fc2,axis = -1),WCS_X,WCS_Y)
display_munsells_inv(np.mean(M_fc2,axis = 0),np.amax(M_fc2))


WCS_MAT_CHROMA = np.zeros((10,41))
count = 0
for i in range(len(CHROMA)):
    WCS_MAT_CHROMA[WCS_X[i],WCS_Y[i]] = float(CHROMA[count])
    count +=1
WCS_MAT_CHROMA = (WCS_MAT_CHROMA/16)
#plt.imshow(np.stack((WCS_MAT_CHROMA,WCS_MAT_CHROMA,WCS_MAT_CHROMA),axis = 2))
#plt.show()

general_mat_consistency = sio.loadmat('/home/alban/Documents/pytorch/project_color_constancy/WCS_Xp/general_consensusconsistency_map4Alban.mat')['cmap_general']
general_mat_consistency2 = np.zeros(general_mat_consistency[:-3,:].shape)
for i in range(len(general_mat_consistency2)):
    general_mat_consistency2[i] = general_mat_consistency[:-3,:][7-i]

display_munsells(general_mat_consistency2,np.amax(general_mat_consistency2))
np.corrcoef(np.mean(M_fc2,axis = 0)[1:-1,1:].flatten(),general_mat_consistency[:-3].flatten())

#display_munsells(np.mean(std_fc1,axis = -1))
#display_munsells(np.mean(std_fc2,axis = -1))

np.corrcoef(np.mean(M_fc2,axis = 0)[1:-1].flatten(),WCS_MAT_CHROMA[1:-1].flatten())

np.corrcoef(general_mat_consistency[:-3].flatten(),WCS_MAT_CHROMA[1:-1,1:].flatten())

mat_speakers = sio.loadmat('/home/alban/Documents/pytorch/project_color_constancy/WCS_Xp/matrix_WCS_speakers.mat')['WCS_speakers']

display_munsells(mat_speakers[1:-1].astype(float),np.amax(mat_speakers[1:-1]))
np.corrcoef(np.mean(M_fc2,axis = 0)[1:-1].flatten(),mat_speakers[1:-1].flatten())

display_munsells_inv(mat_speakers.astype(float),np.amax(mat_speakers))


### Correlations at fixed value

# In[9]: measure of colo constancy
# As a measure of color constancy. we have the ratio of the representational dispertion with ditance between adjacent
# chips. representational dispertion = sqrt(sum(var))

def constancy_measure(layer,WCS_X,WCS_Y):
    mean_layer = np.mean(layer,axis = 2)
    MAT_mean_layer = np.zeros((layer.shape[0],10,41,layer.shape[-1]))
    MAT_disp_layer = np.zeros((layer.shape[0],10,41))
    for i in range(330):
        MAT_mean_layer[:,WCS_X[i],WCS_Y[i]] = mean_layer[:,i]
        MAT_disp_layer[:,WCS_X[i],WCS_Y[i]] = np.sqrt(np.sum(np.var(layer[:,i],axis =-2),axis = -1))
    
    
    MAT_mean_layer_color = MAT_mean_layer[:,1:-1,1:]
    MAT_disp_layer_color = MAT_disp_layer[:,1:-1,1:]
    
    D_MAT_mean_layer = np.zeros((layer.shape[0],8,40))
    for i in range(D_MAT_mean_layer.shape[1]):
        for j in range(D_MAT_mean_layer.shape[2]):
            d1 = np.sqrt(np.sum((MAT_mean_layer_color[:,i,j]-MAT_mean_layer_color[:,(i+1)%8,j])**2,axis = 1))
            d2 = np.sqrt(np.sum((MAT_mean_layer_color[:,i,j]-MAT_mean_layer_color[:,(i-1)%8,j])**2,axis = 1))
            d3 = np.sqrt(np.sum((MAT_mean_layer_color[:,i,j]-MAT_mean_layer_color[:,i,(j+1)%40])**2,axis = 1))
            d4 = np.sqrt(np.sum((MAT_mean_layer_color[:,i,j]-MAT_mean_layer_color[:,i,(j-1)%40])**2,axis = 1))
            if i == 0:
                d2 = np.zeros(layer.shape[0])
                d2[:] = np.nan
            elif i == 7:
                d1 = np.zeros(layer.shape[0])
                d1[:] = np.nan
            D_MAT_mean_layer[:,i,j] = np.nanmean((d1,d2,d3,d4), axis = 0)
    
    constancy_layer = MAT_disp_layer_color/D_MAT_mean_layer
    return D_MAT_mean_layer,constancy_layer


D_MAT_mean_conv1,constancy_conv1 = constancy_measure(conv1,WCS_X,WCS_Y)
D_MAT_mean_conv2,constancy_conv2 = constancy_measure(conv2,WCS_X,WCS_Y)
D_MAT_mean_conv3,constancy_conv3 = constancy_measure(conv3,WCS_X,WCS_Y)
D_MAT_mean_fc1,constancy_fc1 = constancy_measure(fc1,WCS_X,WCS_Y)
D_MAT_mean_fc2,constancy_fc2 = constancy_measure(fc2,WCS_X,WCS_Y)

print(np.mean(constancy_conv1),np.mean(constancy_conv2),np.mean(constancy_conv3),np.mean(constancy_fc1),np.mean(constancy_fc2))


## In[9]: idx of neuron dynamics: codes more munsell changes or illuminant changes?
## As an idx of the neurons dynamics, we take the ratio of 
#
#def idx_dyn(layer):
#    layer_illu_std = np.mean(np.std(layer,axis = -2), axis = 1)
#    layer_chips_std = np.mean(np.std(layer,axis = -3),axis = 1)
#    
#    layer_illu_std_norm = np.mean(np.std(layer,axis = -2), axis = 1)/np.mean(np.mean(layer,axis = -2), axis = 1)
#    layer_chips_std_norm = np.mean(np.std(layer,axis = -3),axis = 1)/np.mean(np.mean(layer,axis = -3),axis = 1)
#    
#    idx_dyn_layer = layer_illu_std_norm/layer_chips_std_norm
#    return np.round(idx_dyn_layer,2)
#
#idx_dyn_conv1 = idx_dyn(conv1)
#idx_dyn_conv2 = idx_dyn(conv2)
#idx_dyn_conv3 = idx_dyn(conv3)
#idx_dyn_fc1 = idx_dyn(fc1)
#idx_dyn_fc2 = idx_dyn(fc2)
#
#np.nanmean(idx_dyn_conv1)
#np.nanmean(idx_dyn_conv2)
#np.nanmean(idx_dyn_conv3)
#np.nanmean(idx_dyn_fc1)
#np.nanmean(idx_dyn_fc2)
#
#fig = plt.figure(figsize = (9,3))
#ax1 = fig.add_subplot(231)
#ax1.scatter(RESULT[10:,0], RESULT[10:,1], marker='o',c = WCS_MAT_sRGB[1:-1,1:].reshape((-1,3)))
#ax1.set_xlabel(LABELS[0],fontsize = 15)
#ax1.set_ylabel(LABELS[1],fontsize = 15)
#ax2 = fig.add_subplot(232)
#ax2.scatter(RESULT[10:,0], RESULT[10:,2], marker='o',c = WCS_MAT_sRGB[1:-1,1:].reshape((-1,3)))
#ax2.set_xlabel(LABELS[0],fontsize = 15)
#ax2.set_ylabel(LABELS[2],fontsize = 15)
#ax3 = fig.add_subplot(233)
#ax3.scatter(RESULT[10:,1], RESULT[10:,2], marker='o',c = WCS_MAT_sRGB[1:-1,1:].reshape((-1,3)))
#ax3.set_xlabel(LABELS[1],fontsize = 15)
#ax3.set_ylabel(LABELS[2],fontsize = 15)
#fig.text(0.5,0.94,title,ha='center',fontsize = 18)
#fig.tight_layout()
#plt.show()
#fig.savefig(path2,format='png', dpi=1200)
#plt.close()

# In[9]: DISSIMILARITY MATRICES 

DM_conv1 = 1 - np.mean(corr_conv1,axis = 0)
DM_conv2 = 1 - np.mean(corr_conv2,axis = 0)
DM_conv3 = 1 - np.mean(corr_conv3,axis = 0)
DM_fc1 = 1 - np.mean(corr_fc1,axis = 0)
DM_fc2 = 1 - np.mean(corr_fc2,axis = 0)

np.amax(DM_conv3)
np.amin(DM_conv3)

plt.imshow(np.stack((DM_conv3,DM_conv3,DM_conv3),axis = 2)/np.amax(DM_conv3))
plt.show()

'''
WCS_CORR_3D = np.zeros((np.mean(corr_conv3,axis = 0).shape[0],np.mean(corr_conv3,axis = 0).shape[1],3))
for j in range(WCS_CORR_3D.shape[0]):
    for i in range(WCS_CORR_3D.shape[1]):
        c = np.mean(corr_conv3,axis = 0)[j,i]
        if c < 0:
            WCS_CORR_3D[j,i] = [c,0,0]
        else:
            WCS_CORR_3D[j,i] = [0,c,0]
plt.imshow(WCS_CORR_3D)
plt.show()

color_bar = np.zeros((1,10,3))
min_corr = np.amin(corr_conv3)
max_corr = np.amax(corr_conv3)
color_code = np.arange(min_corr,max_corr,(max_corr - min_corr)/9)
for i in range(9):
    if color_code[i]<0:
        color_bar[0,i] = [color_code[i],0,0]
    else:
        color_bar[0,i] = [0,color_code[i],0]'''

# In[9]: MDS color constant


RESULTconv1, stress_conv1  = MDS(DM_conv1)
coeff1,score1,latent1,explained1 = princomp(np.mean(conv1,axis = (0,2)))


RESULTconv2, stress_conv2  = MDS(DM_conv2)
coeff2,score2,latent2,explained2 = princomp(np.mean(conv2,axis = (0,2)))

RESULTconv3, stress_conv3 = MDS(DM_conv3)
coeff3,score3,latent3,explained3 = princomp(np.mean(conv3,axis = (0,2)))

RESULTfc1, stress_fc1 = MDS(DM_fc1)
coeff4,score4,latent4,explained4 = princomp(np.mean(fc1,axis = (0,2)))

RESULTfc2, stress_fc2 = MDS(DM_fc2)
coeff5,score5,latent5,explained5 = princomp(RESULTfc2)

np.sum((stress_conv1/np.sum(stress_conv1[stress_conv1>0]))[0:3])
np.sum((stress_conv2/np.sum(stress_conv2[stress_conv2>0]))[0:3])
np.sum((stress_conv3/np.sum(stress_conv3[stress_conv3>0]))[0:3])
np.sum((stress_fc1/np.sum(stress_fc1[stress_fc1>0]))[0:3])
np.sum((stress_fc2/np.sum(stress_fc2[stress_fc2>0]))[0:5])

# In[9]: plot MDS color constant

    
#scatter_muns(RESULTconv1,'CONV 1')
scatter_muns(RESULTconv1[:,:3],'CONV 1','MDS/conv1_MDS_3D_classic.png','All_muns/conv1_MDS_2D_classic.png')
#scatter_muns(RESULTconv2)
scatter_muns(RESULTconv2[:,:3],'CONV 2','MDS/conv2_MDS_3D_classic.png','All_muns/conv2_MDS_2D_classic.png')
#scatter_muns(RESULTconv3)
scatter_muns(RESULTconv3[:,:3],'CONV 3','MDS/conv3_MDS_3D_classic.png','All_muns/conv3_MDS_2D_classic.png')
#scatter_muns(RESULTfc1)
scatter_muns(RESULTfc1[:,:3],'FC 1','MDS/fc1_MDS_3D_classic.png','All_muns/fc1_MDS_2D_classic.png')
#scatter_muns(RESULTfc2)
scatter_muns(RESULTfc2[:,:3],'FC 2','MDS/fc2_MDS_3D_classic.png','All_muns/fc2_MDS_2D_classic.png')


# In[9]: Test for constant value


# In[9]: Test for constant value

WCS_X[123]
WCS_Y[124]

corr_conv1_v5 = correlations_layers(conv1[:,125:164])
corr_conv2_v5= correlations_layers(conv2[:,125:164])
corr_conv3_v5= correlations_layers(conv3[:,125:164])
corr_fc1_v5= correlations_layers(fc1[:,125:164])
corr_fc2_v5= correlations_layers(fc2[:,125:164])

DM_conv1_v5 = 1 - np.mean(corr_conv1_v5,axis = 0)
DM_conv2_v5 = 1 - np.mean(corr_conv2_v5,axis = 0)
DM_conv3_v5 = 1 - np.mean(corr_conv3_v5,axis = 0)
DM_fc1_v5 = 1 - np.mean(corr_fc1_v5,axis = 0)
DM_fc2_v5 = 1 - np.mean(corr_fc2_v5,axis = 0)



def scatter_muns_2D(RESULT,L,title,path1,path2):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(RESULT[:,0], RESULT[:,1], marker='o',c = WCS_MAT_sRGB[L,1:].reshape((-1,3)))
    ax.set_xlabel('DIM 1',fontsize = 15)
    ax.set_ylabel('DIM 2',fontsize = 15)
    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)
    fig.text(0.5,0.97,title,ha='center',fontsize = 18)
    fig.tight_layout()
    plt.show()
    fig.savefig(path1)
    plt.close()



# In[9]: Similarity analysis (procrustes)

def procrustes(X, Y, scaling=True, reflection='best'):
    """
    A port of MATLAB's `procrustes` function to Numpy.

    Procrustes analysis determines a linear transformation (translation,
    reflection, orthogonal rotation and scaling) of the points in Y to best
    conform them to the points in matrix X, using the sum of squared errors
    as the goodness of fit criterion.

        d, Z, [tform] = procrustes(X, Y)

    Inputs:
    ------------
    X, Y    
        matrices of target and input coordinates. they must have equal
        numbers of  points (rows), but Y may have fewer dimensions
        (columns) than X.

    scaling 
        if False, the scaling component of the transformation is forced
        to 1

    reflection
        if 'best' (default), the transformation solution may or may not
        include a reflection component, depending on which fits the data
        best. setting reflection to True or False forces a solution with
        reflection or no reflection respectively.

    Outputs
    ------------
    d       
        the residual sum of squared errors, normalized according to a
        measure of the scale of X, ((X - X.mean(0))**2).sum()

    Z
        the matrix of transformed Y-values

    tform   
        a dict specifying the rotation, translation and scaling that
        maps X --> Y

    """

    n,m = X.shape
    ny,my = Y.shape

    muX = X.mean(0)
    muY = Y.mean(0)

    X0 = X - muX
    Y0 = Y - muY

    ssX = (X0**2.).sum()
    ssY = (Y0**2.).sum()

    # centred Frobenius norm
    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)

    # scale to equal (unit) norm
    X0 /= normX
    Y0 /= normY

    if my < m:
        Y0 = np.concatenate((Y0, np.zeros(n, m-my)),0)

    # optimum rotation matrix of Y
    A = np.dot(X0.T, Y0)
    U,s,Vt = np.linalg.svd(A,full_matrices=False)
    V = Vt.T
    T = np.dot(V, U.T)

    if reflection is not 'best':

        # does the current solution use a reflection?
        have_reflection = np.linalg.det(T) < 0

        # if that's not what was specified, force another reflection
        if reflection != have_reflection:
            V[:,-1] *= -1
            s[-1] *= -1
            T = np.dot(V, U.T)

    traceTA = s.sum()

    if scaling:

        # optimum scaling of Y
        b = traceTA * normX / normY

        # standarised distance between X and b*Y*T + c
        d = 1 - traceTA**2

        # transformed coords
        Z = normX*traceTA*np.dot(Y0, T) + muX

    else:
        b = 1
        d = 1 + ssY/ssX - 2 * traceTA * normY / normX
        Z = normY*np.dot(Y0, T) + muX

    # transformation matrix
    if my < m:
        T = T[:my,:]
    c = muX - b*np.dot(muY, T)

    #transformation values 
    tform = {'rotation':T, 'scale':b, 'translation':c}

    return d, Z, tform

disp_layer1,score1_D65_stan,tform_layer1 = procrustes(RESULTconv1[:,:3].T, RESULTconv1_D65[:,:3].T)
scatter_muns(score1_D65_stan.T,'CONV 1','mod_N2/conv1_stand_MDS_3D.png','mod_N2/conv1_stand_MDS_2D.png')

disp_layer2,score2_D65_stan,tform_layer2 = procrustes(RESULTconv2[:,:3].T, RESULTconv2_D65[:,:3].T)
disp_layer3,score3_D65_stan,tform_layer3 = procrustes(RESULTconv3[:,:3].T, RESULTconv3_D65[:,:3].T)
disp_layer4,score4_D65_stan,tform_layer4 = procrustes(RESULTfc1[:,:3].T, RESULTfc1_D65[:,:3].T)
disp_layer5,score5_D65_stan,tform_layer5 = procrustes(RESULTfc2[:,:3].T, RESULTfc2_D65[:,:3].T)


scatter_muns(score5_D65_stan.T,'FC 2','mod_N2/fc2_stand_MDS_3D.png','mod_N2/fc2_stand_MDS_2D.png')
score5_D65_stan2 = score5_D65_stan.copy() 
score5_D65_stan2[0] = -score5_D65_stan2[0]
scatter_muns(score5_D65_stan2.T,'FC 2','mod_N2/fc2_stand_MDS_3D.png','mod_N2/fc2_stand_MDS_2D.png')

Diff_lay_1 = np.sqrt(np.sum((score1_D65_stan - RESULTconv1[:,:3].T)**2,axis = 0))
Diff_lay_1 = Diff_lay_1.reshape(1,-1)
Diff_lay_1_mat = compute_munsells(Diff_lay_1,WCS_X,WCS_Y)
display_munsells_inv(Diff_lay_1_mat[0],np.amax(Diff_lay_1_mat))
plt.show()

Diff_lay_2 = np.sqrt(np.sum((score2_D65_stan - RESULTconv2[:,:3].T)**2,axis = 0))
Diff_lay_2 = Diff_lay_2.reshape(1,-1)
Diff_lay_2_mat = compute_munsells(Diff_lay_2,WCS_X,WCS_Y)
display_munsells_inv(Diff_lay_2_mat[0],np.amax(Diff_lay_2_mat))
plt.show()

Diff_lay_3 = np.sqrt(np.sum((score3_D65_stan - RESULTconv3[:,:3].T)**2,axis = 0))
Diff_lay_3 = Diff_lay_3.reshape(1,-1)
Diff_lay_3_mat = compute_munsells(Diff_lay_3,WCS_X,WCS_Y)
display_munsells_inv(Diff_lay_3_mat[0],np.amax(Diff_lay_3_mat))
plt.show()

Diff_lay_4 = np.sqrt(np.sum((score4_D65_stan - RESULTfc1[:,:3].T)**2,axis = 0))
Diff_lay_4 = Diff_lay_4.reshape(1,-1)
Diff_lay_4_mat = compute_munsells(Diff_lay_4,WCS_X,WCS_Y)
display_munsells_inv(Diff_lay_4_mat[0],np.amax(Diff_lay_4_mat))
plt.show()

Diff_lay_5 = np.sqrt(np.sum((score5_D65_stan - RESULTfc2[:,:3].T)**2,axis = 0))
Diff_lay_5 = Diff_lay_5.reshape(1,-1)
np.mean(Diff_lay_5)
Diff_lay_5_mat = compute_munsells(Diff_lay_5,WCS_X,WCS_Y)
display_munsells_inv(Diff_lay_5_mat[0],np.amax(Diff_lay_5_mat))
plt.show()




# In[9]: similarity analysis (procrustes) with Munsell space


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
    
WCS_MUNS_XYValue_coord = np.array([X_muns,Y_muns,VALUE_arr])

scatter_muns(WCS_MUNS_XYValue_coord.T,'Munsell Coordinates','Munsell_coord_3D.png','Munsell_coord_2D.png', ['Green-Red', 'Blue-Yellow', 'Value' ])

disp_MUNS_1,score_MUNS_1,tform_MUNS_1 = procrustes(WCS_MUNS_XYValue_coord, RESULTconv1[:,:3].T)
disp_MUNS_2,score_MUNS_2,tform_MUNS_1 = procrustes(WCS_MUNS_XYValue_coord, RESULTconv2[:,:3].T)
disp_MUNS_3,score_MUNS_3,tform_MUNS_1 = procrustes(WCS_MUNS_XYValue_coord, RESULTconv3[:,:3].T)
disp_MUNS_4,score_MUNS_4,tform_MUNS_1 = procrustes(WCS_MUNS_XYValue_coord, RESULTfc1[:,:3].T)
disp_MUNS_5,score_MUNS_5,tform_MUNS_1 = procrustes(WCS_MUNS_XYValue_coord, RESULTfc2[:,:3].T)

disp_MUNS_1,score_MUNS_1,tform_MUNS_1 = procrustes(WCS_MUNS_XYValue_coord, RESULTconv1[:,:3].T)
disp_MUNS_2,score_MUNS_2,tform_MUNS_2 = procrustes(WCS_MUNS_XYValue_coord, RESULTconv2[:,:3].T)
disp_MUNS_3,score_MUNS_3,tform_MUNS_3 = procrustes(WCS_MUNS_XYValue_coord, RESULTconv3[:,:3].T)
disp_MUNS_4,score_MUNS_4,tform_MUNS_4 = procrustes(WCS_MUNS_XYValue_coord, RESULTfc1[:,:3].T,scaling = False)
disp_MUNS_5,score_MUNS_5,tform_MUNS_5 = procrustes(WCS_MUNS_XYValue_coord, RESULTfc2[:,:3].T)

scatter_muns(score_MUNS_4.T,'FC 2','All_muns/fc1_procrustes_muns_3D.png','All_muns/fc1_procrustes_muns_2D.png')
