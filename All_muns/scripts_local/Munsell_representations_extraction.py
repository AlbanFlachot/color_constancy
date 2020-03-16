#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 17:11:40 2019

@author: alban
"""

import numpy as np
import matplotlib.pyplot as plt

import pickle



# In[9]: FUNCTIONS

def compute_munsells(EVAL,WCS_X,WCS_Y):
    WCS_MAT = np.zeros((len(EVAL),10,41))
    for j in range(len(EVAL)):
        for i in range(len(EVAL[0])):
            WCS_MAT[j,WCS_X[i],WCS_Y[i]] = EVAL[j][i]
    return WCS_MAT


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
    CORR = np.zeros((len(conv1), conv1.shape[1],conv1.shape[1]))
    for i in range(len(conv1)):
        CORR[i] = np.corrcoef(mean_conv1[i])
    return CORR

def scatter_muns2(RESULT,title,path1,path2, LABELS = ['DIM 1','DIM 2','DIM 3']):
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


# In[9]: CORRELATIONs

fc2 = np.load(ADDR)

with open("XYZ_WCS.txt", "rb") as fp:   #Pickling
    XYZ_WCS = pickle.load(fp)


# In[9]: CORRELATIONs

corr_fc2 = correlations_layers(fc2)


# In[9]: DISSIMILARITY MATRICES

DM_fc2 = 1 - np.mean(corr_fc2,axis = 0)


# In[9]: MDS color constant

RESULTfc2, stress_fc2 = MDS(DM_fc2)
coeff5,score5,latent5,explained5 = princomp(RESULTfc2)

np.sum((stress_fc2/np.sum(stress_fc2[stress_fc2>0]))[0:5])

# In[9]: plot MDS color constant

scatter_muns(RESULTfc2[:,:3],'FC 2','MDS/result_MDS_3D.png','MDS/result_MDS_2D.png')

