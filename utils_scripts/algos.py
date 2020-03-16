#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 12:03:14 2019

@author: alban
"""

import numpy as np

CC_dir_path = '/home/alban/Documents/pytorch/project_color_constancy/WCS_Xp/'

def compute_WCS_Munsells_categories():
	'''
	Function that maps WCS munsells onto the 1600 munsells indexes (e.g WCS muns 0 = muns 1521)
	'''
	
	## list of Munsells used in the World Color Survey
	WCS_muns = list()
	with open(CC_dir_path + "WCS_muns.txt") as f:
		for line in f:
		   WCS_muns.append(line.split()[0])

	## list of 1600 Munsells	
	All_muns = list()
	with open(CC_dir_path + "munsell_labels.txt") as f:
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
