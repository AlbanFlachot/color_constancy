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
	if XYZ.shape[-1] != 3:
		raise ValueError('First dimension of XYZ must be 3')
	if white.shape == 1:
		raise ValueError('Please define the white point')
	
	#XYZtemp = XYZ.reshape(3,-1)
	
	epsi = 0.008856
	K = 903.3
	
	XYZr = (XYZ/white)
	Fxyz = XYZr**(1./3.) 
	Fxyz[XYZr<epsi] = (K*XYZr[XYZr<epsi] + 16)/116
	
	Fxyz_flat = Fxyz.reshape((-1,3))
	Lab = np.zeros(Fxyz_flat.shape)
	Lab[:,0] = 116 * Fxyz_flat[:,1] - 16
	Lab[:,1] = 500 * (Fxyz_flat[:,0] - Fxyz_flat[:,1])
	Lab[:,2] = 200 * (Fxyz_flat[:,1] - Fxyz_flat[:,2])
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

def XYZ2LMS(XYZ):
    '''
    Convert from LMS to XYZ coordinates using the matrix given by a linear regression
    Input:
        LMS: Matrix of LMS values (could be an image)
    Outputs:
        XYZ: Matrix of XYZ values
    '''
    if XYZ.shape[-1] != 3:
        raise ValueError('Last dimension of XYZ must be 3')
    M = np.linalg.inv(np.array([[ 4.51420115e+01, -2.68211814e+01,  4.25120051e+00],
       [ 1.59927663e+01,  6.60496090e+00,  1.26892433e-07],
       [-5.03167761e-07, -3.30469228e-07,  2.25500896e+01]]))
    #XYZtemp = XYZ.reshape(3,-1)
    LMS = np.dot(XYZ,M.T)
    return LMS

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
    
def sRGB2XYZ(RGB):
    Trans = np.linalg.inv(np.array([[3.24045, -1.537138, -0.49853],[-0.9692660, 1.8760108, 0.0415560],[0.0556434, -0.2040259, 1.0572252]]))
    XYZ = np.dot(RGB,Trans.T)
    return XYZ

def cart2sph(x,y,z):
	'''
	Function to convert from cartesians to spherical coordinates
	'''
	XsqPlusYsq = x**2 + y**2
	r = np.sqrt(XsqPlusYsq + z**2)# r
	elev = np.arctan(z/np.sqrt(XsqPlusYsq)) # theta
	az = np.arctan2(y,x) # phi
	return np.array([r, az, elev])

def VXY2VHC(VXY, muns = 'True'):
    '''
    Fuction that converts Munsell representation from cardinal (Value X, Y) to cylindrical (Value, Hue, Chroma)
    '''

    shape = VXY.shape
    if shape == 3:
        VXY = VXY.reshape(1,3)
    else:
        VXY = VXY.reshape(-1,3)

    VHC = VXY.copy()
    VHC[:,-1] = np.linalg.norm(VXY[:,1:], axis = -1)
    #import dbg; dbg.set_trace()
    VHC[:,1] = (np.arccos(VXY[:,1]/VHC[:,-1])*np.sign(VXY[:,2]))
    VHC[VHC[:,-1] == 0,1] = 0
    if muns:
        VHC[:,1] = VHC[:,1]*180/np.pi/4.5
    VHC = VHC.reshape(shape)
    return VHC
