#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 16:03:24 2019

@author: alban
"""
import numpy as np


def XYZ2Lab(XYZ,white = 0.0):
	'''
    Convert from XYZ to CIELab coordinates using the matrix given by a linear regression
    Inputs:
        XYZ: Matrix of XYZ values (could be an image)
		White: Self explanatory. If not specified, the maximum value of the XYZ input.
    Outputs:
        Lab: Matrix of CIELab values
    '''
	if XYZ.shape[0] != 3:
		raise ValueError('Firt dimension of XYZ must be 3')
	if white.shape == 1:
		raise ValueError('Please define the white point')

	#XYZtemp = XYZ.reshape(3,-1)

	epsi = 0.008856
	K = 903.3

	Xr = XYZ[0]/white[0]
	Yr = XYZ[1]/white[1]
	Zr = XYZ[2]/white[2]
	fx = Xr**(1./3.)
	fy = Yr**(1./3.)
	fz = Zr**(1./3.)

	fx[Xr<epsi] = (K*Xr[Xr<epsi]+16)/116
	fy[Yr<epsi] = (K*Yr[Yr<epsi]+16)/116
	fz[Zr<epsi] = (K*Zr[Zr<epsi]+16)/116

	Lab = np.zeros(XYZ.shape)
	Lab[0] = 116 * fy - 16
	Lab[1] = 500 * (fx - fy)
	Lab[2] = 200 * (fy - fz)
	return Lab

def LMS2XYZ(LMS):
    '''
    Convert from LMS to XYZ coordinates using the matrix given by a linear regression
    Input:
        LMS: Matrix of LMS values (could be an image)
    Outputs:
        XYZ: Matrix of XYZ values
    '''
    if LMS.shape[-1] != 3:
        raise ValueError('Last dimension of XYZ must be 3')
    M = np.array([[ 4.51420115e+01, -2.68211814e+01,  4.25120051e+00],
       [ 1.59927663e+01,  6.60496090e+00,  1.26892433e-07],
       [-5.03167761e-07, -3.30469228e-07,  2.25500896e+01]])
    #XYZtemp = XYZ.reshape(3,-1)
    XYZ = np.dot(LMS,M.T)
    return XYZ


def Sharpening(x):
    M_diag = np.array([[ 1.6934   , -1.5335   ,  0.075    ],
       [-0.5341875,  1.3293125, -0.1401875],
       [ 0.0215   , -0.0432   ,  1.0169   ]])
    Sharp_x = np.dot(x, M_diag.T)
    return Sharp_x

def Unsharpening(x):
    M_diag = np.array([[ 0.92806228,  1.07319983,  0.07950096],
       [ 0.37254386,  1.18645912,  0.13608609],
       [-0.0037953 ,  0.02771289,  0.98748122]])
    Unsharp_x = np.dot(x, M_diag.T)
    return Unsharp_x

def XYZ2sRGB(XYZ):
    Trans = np.array([[3.24045, -1.537138, -0.49853],[-0.9692660, 1.8760108, 0.0415560],[0.0556434, -0.2040259, 1.0572252]])
    sRGB = np.dot(XYZ,Trans.T)
    return sRGB
