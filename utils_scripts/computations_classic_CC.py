#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 17:45:58 2019

@author: alban
"""
import numpy as np
import os
import matplotlib.pyplot as plt

def save_cc_images(IMG, names_algo, path, img_name):
	'''
	Function that loads full images with their illuimant compensated by some algo, and saves them.

	Inputs:
		- IMG: Matrix of images. shape = [height, width, color channels, nb_algos]
		- names_algo: list of algos names. Length should be = to nb_algos
	Outputs:
		- OBJ: List of the
	'''

	if len(names_algo) != IMG.shape[-2]:
		print ('The numbers of specified algos does not match the number given as input.')

	for i in range(IMG.shape[-2]):
		if not os.path.exists( path + names_algo[i] + '/'):
			os.mkdir( path + names_algo[i] + '/')
		np.save(path + names_algo[i] + '/' + img_name + '.npy', IMG[:,:,i])


def from_img_2_obj(im_paths_mat, path2save, mat_classic_cc_mat, shape, save_path, condition = 'compensated'):
	'''
	Function that loads full images, their mask, extracts the object area and saves these.

	Inputs:
		- im_paths_mat: Matrix of paths to images. shape = [muns, illus, exp]
		- path2save: path to dir where to save the selected object area
		- mat_classic_cc_mat: matrix of illuminants estimation resulting from constancy algo.
		- shape: shape that the output should have
	Outputs:
		- OBJ: List of the croped object.
	'''
	wh  = np.array([17.95504939, 18.96455292, 20.36406225])
	wh = wh/np.mean(wh)
	OBJ = list()
	for muns in range(shape[0]):
	#for muns in range(1):
		Obj = list()
		for illu in range(shape[1]):
			obj = list()
			for exp in range(shape[2]):
				#print im_paths_mat[muns,illu,exp]
				image = np.load(im_paths_mat[muns,illu,exp][:12]+'mnt/awesome/alban/'+im_paths_mat[muns,illu,exp][12:])
				image_norm = image/np.amax(image)
				image_norm = np.expand_dims(image_norm,axis = 2)
				if mat_classic_cc_mat.size == 3:
					image_norm_ntr = np.repeat(image_norm,shape[3],axis = 2)/mat_classic_cc_mat
				elif mat_classic_cc_mat.size == range(shape[1])*3:
					image_norm_ntr = np.repeat(image_norm,shape[3],axis = 2)/mat_classic_cc_mat[illu]
				else:
					image_norm_ntr = np.repeat(image_norm,shape[3],axis = 2)/mat_classic_cc_mat[muns,illu,exp,:,:]
				#print image_norm_ntr.shape
				#save_cc_images(image_norm_ntr*np.amax(image)*wh, ['gw', 'wp', 'edg1', 'edg2', 'contrast'], save_path, im_paths_mat[muns,illu,exp][79:-4] )
				mask = np.load(im_paths_mat[muns,illu,exp][:12]+'mnt/awesome/alban/'+im_paths_mat[muns,illu,exp][12:-4] + '_mask' + '.npy')
				mask_1 = np.mean(mask, axis = -1)
				mask_1[mask_1>0.2] = 1
				object = (image_norm_ntr.T*mask_1.T).T
				object = object.reshape((-1, shape[3], 3))
				obj.append(object[object[:,0,0]>0])
			Obj.append(obj)
		OBJ.append(Obj)

	for muns in range(len(im_paths_mat)):
		np.save(path2save%muns, OBJ[muns])

	return OBJ

def predicted_chromaticity(im_paths_mat, mat_classic_cc_mat, shape, max_pxl, correction = True):
	'''
	Function that loads full images, their mask, extracts the object area and saves these.

	Inputs:
		- im_paths_mat: Matrix of paths to images. shape = [muns, illus, exp]
		- path2save: path to dir where to save the selected object area
		- mat_classic_cc_mat: matrix of illuminants estimation resulting from constancy algo.
		- shape: shape that the output should have [nb_muns, nb_illu, nb_exp, nb_models_CC, nb_channels]
	Outputs:
		- OBJ: List of the croped object.
	'''
	CHROMA_OBJ_MEAN = np.zeros(shape)
	CHROMA_OBJ_LUM = np.zeros(shape)
	CHROMA_OBJ_MEDIAN = np.zeros(shape)
	for muns in range(shape[0]):
	#for muns in range(1):
		for illu in range(shape[1]):
			for exp in range(shape[2]):
				#print im_paths_mat[muns,illu,exp]
				image = np.load(im_paths_mat[muns,illu,exp]) # load image
				image_norm = image/max_pxl.max() # normalize it
				image_norm = np.expand_dims(image_norm,axis = 2)

				#import pdb; pdb.set_trace()
				if mat_classic_cc_mat.size == 3:
					image_norm_ntr = image_norm/(mat_classic_cc_mat/np.mean(mat_classic_cc_mat))
				elif mat_classic_cc_mat.size == shape[1]*shape[-1]:
					image_norm_ntr = image_norm/(mat_classic_cc_mat[illu]/np.mean(mat_classic_cc_mat[illu]))
				else:
					image_norm_ntr = (np.repeat(image_norm,shape[3],axis = 2)/
                    (mat_classic_cc_mat[muns,illu,exp,:,:]/np.mean(mat_classic_cc_mat[muns,illu,exp,:,:])))
				if correction:
					image_norm_ntr = image_norm_ntr*np.array([0.95595725, 1.00115551, 1.04288724]) # D65 chromaticity
				mask = np.load(im_paths_mat[muns,illu,exp][:-4] + '_mask' + '.npy')
				mask_1 = np.mean(mask, axis = -1)
				mask_1[mask_1>0.2] = 1
				object = (image_norm_ntr.T*mask_1.T).T
				object = object.reshape((-1, shape[3], 3))
				object = object[object[:,0,0]>0] # we consider only the croped objct pixels
				chroma_obj_mean = np.mean(object,axis = 0)
				chroma_obj_median = np.median(object,axis = 0)
				most_bright = np.argsort(np.sum(object, axis = -1),axis = 0)[-10:]
				chroma_obj_lum = np.mean(object[most_bright[:,0],:],axis = 0)
				CHROMA_OBJ_MEAN[muns,illu,exp] = chroma_obj_mean
				CHROMA_OBJ_LUM[muns,illu,exp] = chroma_obj_lum
				CHROMA_OBJ_MEDIAN[muns,illu,exp] = chroma_obj_median
	return CHROMA_OBJ_MEAN, CHROMA_OBJ_LUM, CHROMA_OBJ_MEDIAN


def white_balance_image(im_path, max_pxl, illu, chroma):
    '''
    Function that corrects images.
    Inputs:
        - im_path: path to image.
        - max_pxl: nromalization value
        - illut: illuminant estimation resulting from constancy algo.
        - chroma: theoretical chromaticity
    Outputs:
        - chromaticity found for object.
    '''
    image = np.load(im_path) # load image
    image_norm = image/max_pxl.max() # normalize it
    image_norm_ntr = image_norm/(illu/np.mean(illu))
    mask = np.load(im_path[:-4] + '_mask' + '.npy')
    mask_1 = np.mean(mask, axis = -1)
    mask_1[mask_1>0.2] = 1
    object = (image_norm_ntr.T*mask_1.T).T
    object = object.reshape((-1, 3))
    object = object[object[:,0]>0] # we consider only the cropped objct pixels
    CHROMA_OBJ_MEAN = np.mean(object,axis = 0)
    #CHROMA_OBJ_MEAN = CHROMA_OBJ_MEAN/(np.linalg.norm(CHROMA_OBJ_MEAN)/np.linalg.norm(chroma))
    CHROMA_OBJ_MEAN = scaling(CHROMA_OBJ_MEAN, MUNSELLS_LMS)
    CHROMA_OBJ_MEDIAN = np.median(object,axis = 0)
    CHROMA_OBJ_MEDIAN = CHROMA_OBJ_MEDIAN/(np.linalg.norm(CHROMA_OBJ_MEDIAN)/np.linalg.norm(chroma))
    most_bright = np.argsort(np.sum(object, axis = -1))[-10:]
    CHROMA_OBJ_LUM = np.mean(object[most_bright[:],:],axis = 0)
    fig = plt.figure()
    ax1 = plt.subplot(2,2,1)
    ax1.imshow(image_norm/image_norm.max())
    ax2 = plt.subplot(2,2,2)
    ax2.imshow(image_norm_ntr/image_norm_ntr.max())
    ax3 = plt.subplot(2,2,3)
    ax3.scatter(object[:,0],object[:,1],color = 'k')
    ax3.scatter(CHROMA_OBJ_MEAN[0],CHROMA_OBJ_MEAN[1], s = 200,color = 'b')
    ax3.scatter(CHROMA_OBJ_MEDIAN[0],CHROMA_OBJ_MEDIAN[1],s = 200,color = 'r')
    ax3.scatter(chroma[0],chroma[1],s = 200,color = 'g')
    ax4 = plt.subplot(2,2,4)
    #import pdb; pdb.set_trace()
    ax4.scatter(object[:,1],object[:,2],color = 'k')
    ax4.scatter(CHROMA_OBJ_MEAN[1],CHROMA_OBJ_MEAN[2],color = 'b',s = 200)
    ax4.scatter(CHROMA_OBJ_MEDIAN[1],CHROMA_OBJ_MEDIAN[2],color = 'r',s = 200)
    ax4.scatter(chroma[1],chroma[2],color = 'g',s = 200)
    plt.show()
    return CHROMA_OBJ_MEAN, CHROMA_OBJ_LUM, CHROMA_OBJ_MEDIAN

def Obj_chromaticity(path2obj, shape, ground_truth_LMS):
	'''
	Function that loads object areas and extracts their LMS chromaticities to be afterwards compared.

	Inputs:
		- path2obj: path to dir where to load the selected object areas
		- shape: shape that the output should have
		- ground_truth_LMS: Matrix of the Munsells theoretical LMS values. shape = [muns, 3]
	Outputs:
		- CHROMA_OBJ_MEAN, CHROMA_OBJ_LUM, CHROMA_OBJ_MEDIAN: Array of the mean, higher luminance and median chromaticities of the munsells/objects
	'''
	CHROMA_OBJ_MEAN = np.zeros(shape)
	CHROMA_OBJ_LUM = np.zeros(shape)
	CHROMA_OBJ_MEDIAN = np.zeros(shape)
	for muns in range(shape[0]):
	#for muns in range(1):
		OBJECT = np.load(path2obj %muns)
		for illu in range(shape[1]):
			Object = OBJECT[illu]
			#OBJECT = OBJ[muns]
			for exp in range(shape[2]):
				object = Object[exp]
				chroma_obj_mean = np.mean(object,axis = 0)
				chroma_obj_median = np.median(object,axis = 0)
				most_bright = np.argsort(np.sum(object, axis = -1),axis = 0)[-10:]
				chroma_obj_lum = np.mean(object[most_bright[:,0],:],axis = 0)
				chroma_obj_mean = (chroma_obj_mean.T/(np.linalg.norm(chroma_obj_mean,axis = -1)/np.linalg.norm(ground_truth_LMS[muns]))).T
				chroma_obj_lum = (chroma_obj_lum.T/(np.linalg.norm(chroma_obj_lum,axis = -1)/np.linalg.norm(ground_truth_LMS[muns]))).T
				chroma_obj_median = (chroma_obj_median.T/(np.linalg.norm(chroma_obj_median,axis = -1)/np.linalg.norm(ground_truth_LMS[muns]))).T
				#chroma_obj = (chroma_obj.T/(np.amax(chroma_obj,axis = -1)/np.amax(MUNSELLS_LMS[muns]))).T
				CHROMA_OBJ_MEAN[muns,illu,exp] = chroma_obj_mean
				CHROMA_OBJ_LUM[muns,illu,exp] = chroma_obj_lum
				CHROMA_OBJ_MEDIAN[muns,illu,exp] = chroma_obj_median
	return CHROMA_OBJ_MEAN, CHROMA_OBJ_LUM, CHROMA_OBJ_MEDIAN

def DE_error_all(CHROMA_OBJ, MUNSELL_LAB):
	'''
    Compute DelatE error forall predicitons given by the classsic algo of color constancy
        CHROMA_OBJ: Predictions given by the alogrithms. shape = [munsell, examplars (illu), algo, color dim]
		MUNSELL_LAB: Correct CIELab coordinates for the munsells
    Outputs:
        DE: Delta E error. shape = [munsell, illu, algo]
    '''
	DE = np.zeros(CHROMA_OBJ.shape[:-1])
	#import pdb;pdb.set_trace()
	for muns in range(CHROMA_OBJ.shape[0]):
		DE[muns] = DE_error(CHROMA_OBJ[muns], MUNSELL_LAB[muns])
	return DE

def DE_error(chroma_obj, munsell_lab):
	'''
    Compute DelatE error for some predicitons given by the classsic algo of color constancy
        CHROMA_OBJ: Predictions given by the alogrithms. shape = [examplars (illu), algo, color dim]
		MUNSELL_LAB: Correct CIELab coordinates for the munsell
    Outputs:
        dist: Delta E error. shape = [illu, algo]
    '''
	dist = np.linalg.norm(chroma_obj - munsell_lab, axis = -1)
	return dist

def scaling(X, Y):
    '''
    Function to scale X according to Y
    '''

    #maxX = np.amax(X, axis = ax)
    #minX = np.amin(X, axis = ax)
    maxX = X[0].max()
    minX = X[-1].min()

    maxY = Y[0].max()
    minY = Y[-1].min()

    #maxY = np.amax(Y, axis = ax)
    #minY = np.amin(Y, axis = ax)

    return (X - minX)*((maxY-minY)/(maxX-minX)) + minY
