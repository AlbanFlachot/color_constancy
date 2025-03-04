#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 12:03:14 2019

@author: alban
"""

import numpy as np
import sys


sys.path.append('../../')

txt_dir_path = '../../txt_files/'
npy_dir_path = '../../npy_files/'

def compute_WCS_Munsells_categories():
	'''
	Function that maps WCS munsells onto the 1600 munsells indexes (e.g WCS muns 0 = muns 1521)
	'''

	## list of Munsells used in the World Color Survey
	WCS_muns = list()
	with open(txt_dir_path + "WCS_muns.txt") as f:
		for line in f:
		   WCS_muns.append(line.split()[0])

	## list of 1600 Munsells
	All_muns = list()
	with open(txt_dir_path + "munsell_labels.txt") as f:
		for line in f:
		   All_muns.append(line.split()[0])

	return np.asarray([All_muns.index(WCS_muns[i]) for i in range(len(WCS_muns))]) ## Position of the WCS munsells among the 1600 munsells


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

def correlations_layers(layer, mod_axis = 0, mean_axis = 2):
	'''
	Correlation of activations along a chosen axis within a given layer.

	Inputs:
		- layer: matrix of activations. First axis is for the model training instance
		  (10 models were trained in my case.)
		- main_axis: axis corresponding to the activations dim we want to correlates
		  (e.g. object, munsells)
		- mean_axis: axis along which we will average the activation to have a mean activation pattern.
		  Can also be a list (e.g (1,2))
	Outputs:
		- CORR: Matrix of coorelations.
	'''

	if len(layer.shape) < 3:
		layer = layer[np.newaxis,:]
	layer = np.moveaxis(layer, mod_axis,0)
	mean_layer = np.mean(layer,axis = mean_axis)
	CORR = np.zeros((len(layer), layer.shape[1], layer.shape[1]))
	for i in range(len(layer)):
		CORR[i] = np.corrcoef(mean_layer[i])
	return CORR

def similarity_layers(layer, mod_axis = 0, mean_axis = 2):
	'''
	Similarity of activations along a chosen axis within a given layer.

	Inputs:
		- layer: matrix of activations. First axis is for the model training instance
		  (10 models were trained in my case.)
		- main_axis: axis corresponding to the activations dim we want to correlates
		  (e.g. object, munsells)
		- mean_axis: axis along which we will average the activation to have a mean activation pattern.
		  Can also be a list (e.g (1,2))
	Outputs:
		- DIST: Dissimilarity matrix.
	'''

	if len(layer.shape) < 3:
		layer = layer[np.newaxis,:]
	layer = np.moveaxis(layer, mod_axis,0)
	#import pdb; pdb.set_trace()
	layer = np.moveaxis(layer, mean_axis,0)
	layer = layer/np.nanmax(np.absolute(layer),axis = 0)
	#layer = layer/np.nanstd(layer,axis = 0)
	layer = np.moveaxis(layer, 0,-1)
	mean_layer = np.nanmean(layer, axis = -1)
	mean_layer[~np.isfinite(mean_layer)] = 0
	DIST = np.zeros((len(layer), layer.shape[1], layer.shape[1]))
	for i in range(len(layer)):
	        for muns in range(layer.shape[1]):
	                for muns2 in range(layer.shape[1]):
		                DIST[i,muns,muns2] = np.linalg.norm(mean_layer[i,muns] - mean_layer[i,muns2])
	#import pdb; pdb.set_trace()                
	return DIST


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
    #import pdb; pdb.set_trace()
    return Y, evals


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

    # centered Frobenius norm
    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)

    # scale to equal (unit) norm
    X0 /= normX
    Y0 /= normY

    if my < m:
        Y0 = np.concatenate((Y0, np.zeros(n, m-my)),0)

    # optimum rotation matrix of Y
    A = np.dot(X0.T, Y0)
    U,s,Vt = np.linalg.svd(A,full_matrices=True)
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
    #print(s)
    traceTA = s.sum()

    if scaling:

        # optimum scaling of Y
        b = traceTA * normX / normY

        # transformed coords
        Z = normY * b * np.dot(Y0, T) + muX
        
        # standarised distance between X and b*Y*T + c
        d = np.sum((X0 - np.dot(Y0, T))**2)/np.sum(X0**2)
        #if d > 1:
        #        print('Procrustes analysis failed to converge')
        #print ((X0**2).sum())

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

def save_pickle(path, dict):
    import pickle
    f = open(path,"wb")
    pickle.dump(dict,f)
    f.close()

def load_pickle(path):
    import pickle
    f = open(path,"rb")
    return pickle.load(f)

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
