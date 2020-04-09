#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 14:04:32 2019

@author: alban
"""

import numpy as np
import matplotlib.pyplot as plt
import sklearn
import pickle
import scipy.io as sio
import scipy


# In[9]: FUNCTIONS
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


def DE_error_all(CHROMA_OBJ, MUNSELL_LAB):
	'''
    Compute DelatE error forall predicitons given by the classsic algo of color constancy
        CHROMA_OBJ: Predictions given by the alogrithms. shape = [munsell, examplars (illu), algo, color dim]
		MUNSELL_LAB: Correct CIELab coordinates for the munsells
    Outputs:
        DE: Delta E error. shape = [munsell, examplars (illu), algo]
    '''
	DE = np.zeros(CHROMA_OBJ.shape[:-1])
	for muns in range(CHROMA_OBJ.shape[0]):
		DE[muns] = DE_error(CHROMA_OBJ[muns], MUNSELL_LAB[muns])
	return DE

def DE_error(chroma_obj, munsell_lab):
	'''
    Compute DelatE error for some predicitons given by the classsic algo of color constancy
        CHROMA_OBJ: Predictions given by the alogrithms. shape = [examplars (illu), algo, color dim]
		MUNSELL_LAB: Correct CIELab coordinates for the munsell
    Outputs:
        dist: Delta E error. shape = [examplars (illu), algo]
    '''
	dist = np.linalg.norm(chroma_obj - munsell_lab, axis = -1)
	return dist

def scatter_LAB(LAB, RGB):
	'''
	Function to plot CIE lab coordinates and display point with some RGB colors
	'''
	fig = plt.figure(figsize = (7,6))
	ax = fig.add_subplot(111)
	ax.scatter(LAB.T[:,1], LAB.T[:,2],marker = 'o',color=RGB,s = 40)
	#ax.set_title('CIELab values under %s'%ill,fontsize = 18)
	ax.set_xlim(-100,100)
	ax.set_ylim(-100,100)
	ax.set_xlabel('a*',fontsize = 25)
	ax.set_ylabel('b*',fontsize = 25)
	plt.xticks(range(-100,110, 50),fontsize = 20)
	plt.yticks(range(-100,110, 50),fontsize = 20)
	fig.tight_layout()
	plt.show()
	#fig.savefig('/home/alban/Documents/pytorch/project_color_constancy/WCS_Xp/training_centered/POSTER_VSS_19/CIELab_MUNS_ab.png', dpi=800)
	plt.close()

	fig = plt.figure(figsize = (7,6))
	ax = fig.add_subplot(111)
	ax.scatter(LAB.T[:,1], LAB.T[:,0],marker = 'o',color=RGB,s = 40)
	#ax.set_title('CIELab values under %s'%ill,fontsize = 18)
	ax.set_xlim(-100,100)
	ax.set_ylim(0,100)
	ax.set_xlabel('a*',fontsize = 25)
	ax.set_ylabel('L*',fontsize = 25)
	plt.xticks(range(-100,110, 50),fontsize = 20)
	plt.yticks(range(0,110, 50),fontsize = 20)
	fig.tight_layout()
	plt.show()
	#fig.savefig('/home/alban/Documents/pytorch/project_color_constancy/WCS_Xp/training_centered/POSTER_VSS_19/CIELab_MUNS_aL.png', dpi=800)
	plt.close()

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



# In[9]: COMPUTATIONS

im_paths = list()
with open("im_paths.txt") as f:
    for line in f:
       im_paths.append(line.split())

import re

train_labels = [int(re.search('object(.*?)/', addr[0]).group(1)) for addr in im_paths]
Illu = [re.search('illu_(.*).npy', addr[0]).group(1) for addr in im_paths]
#val_labels = [int(re.search('object(.*)_', addr).group(1)[:-2]) for addr in val_addrs]
#val_labels = [int(re.search('object(.*?)_', addr).group(1)) for addr in val_addrs]

mat_classic_cc = np.load('NPY_files/all_luminanes.npy')
mat_classic_cc_mat = np.empty((1600,28,5,3))
im_paths_mat = np.empty((1600,28),dtype=np.object)
count = 0
for muns in range(len(im_paths_mat)):
	for exp in range(im_paths_mat.shape[1]):
		im_paths_mat[train_labels[count],exp] = im_paths[count][0]
		mat_classic_cc_mat[train_labels[count],exp] = mat_classic_cc[count]
		count +=1


MUNSELLS_LMS = np.load('/home/alban/Documents/project_color_constancy/MUNSELLS_LMS.npy')

wh  = np.array([17.95504939, 18.96455292, 20.36406225])
XYZ_MUNS = LMS2XYZ(MUNSELLS_LMS).T

MUNS_LAB = XYZ2Lab(XYZ_MUNS,white = wh)

RGB_1600_muns = np.load('/home/alban/Documents/project_color_constancy/RGB_1600_muns.npy')
scatter_LAB(MUNS_LAB, RGB_1600_muns)



CHROMA_OBJ_MEAN = np.zeros((1600,28,5,3))
CHROMA_OBJ_LUM = np.zeros((1600,28,5,3))


OBJ = list()
for muns in range(len(im_paths_mat)):
#for muns in range(1):
	obj = list()
	for exp in range(im_paths_mat.shape[1]):
		image = np.load(im_paths_mat[muns,exp][:12]+'mnt/awesome/alban/'+im_paths_mat[muns,exp][12:])
		image_norm = image/np.amax(image)
		image_norm = np.expand_dims(image_norm,axis = 2)
		image_norm_ntr = np.repeat(image_norm,5,axis = 2)/mat_classic_cc_mat[muns,exp,:,:]
		mask = np.load(im_paths_mat[muns,exp][:12]+'mnt/awesome/alban/'+im_paths_mat[muns,exp][12:-4] + '_mask' + '.npy')
		mask_1 = np.mean(mask, axis = -1)
		mask_1[mask_1>0.2] = 1
		object = (image_norm_ntr.T*mask_1.T).T
		object = object.reshape((-1,5,3))
		obj.append(object[object[:,0,0]>0])
	OBJ.append(obj)

#np.save('/home/alban/mnt/DATA/project_color_constancy/obj_segmented.npy',OBJ)

import pickle
file_write = open('/home/alban/mnt/awesome/alban/project_color_constancy/PYTORCH/WCS/train_centered/classic_color_constancy/NPY_files/OBJ_val.npy', 'wb')
pickle.dump(OBJ,file_write)
file_write.close()
print "Saving over."

for muns in range(len(im_paths_mat)):
	np.save('/home/alban/mnt/DATA/project_color_constancy/classic_constancy/obj_segmented/munsell_%i.npy'%muns,OBJ[muns])


for muns in range(len(im_paths_mat)):
#for muns in range(1):
	OBJECT = np.load('/home/alban/mnt/DATA/project_color_constancy/classic_constancy/obj_segmented/munsell_%i.npy'%muns)
	#OBJECT = OBJ[muns]
	for exp in range(im_paths_mat.shape[1]):
		object = OBJECT[exp]
		chroma_obj_mean = np.mean(object,axis = 0)
		most_bright = np.argsort(np.sum(object, axis = -1),axis = 0)[-10:]
		chroma_obj_lum = np.mean(object[most_bright[:,0],:],axis = 0)
		chroma_obj_mean = (chroma_obj_mean.T/(np.linalg.norm(chroma_obj_mean,axis = -1)/np.linalg.norm(MUNSELLS_LMS[muns]))).T
		chroma_obj_lum = (chroma_obj_lum.T/(np.linalg.norm(chroma_obj_lum,axis = -1)/np.linalg.norm(MUNSELLS_LMS[muns]))).T
		#chroma_obj = (chroma_obj.T/(np.amax(chroma_obj,axis = -1)/np.amax(MUNSELLS_LMS[muns]))).T
		CHROMA_OBJ_MEAN[muns,exp] = chroma_obj_mean
		CHROMA_OBJ_LUM[muns,exp] = chroma_obj_lum




#PREDICTION_XYZ = LMS2XYZ(CHROMA_OBJ_MEAN).T
#PREDICTION_LAB = XYZ2Lab(PREDICTION_XYZ,white = wh)

#np.save('PREDICTION_LAB_lumnorm_mean.npy',PREDICTION_LAB)
PREDICTION_LAB = np.load('PREDICTION_LAB_lumnorm_mean.npy')

scatter_LAB(PREDICTION_LAB[:,-1,10,:], RGB_1600_muns)
DELTAE = DE_error_all(PREDICTION_LAB.T, MUNS_LAB.T)
np.mean(DELTAE,axis = (0,1))




#PREDICTION_XYZ = LMS2XYZ(CHROMA_OBJ_LUM).T
#PREDICTION_LAB = XYZ2Lab(PREDICTION_XYZ,white = wh)

#np.save('PREDICTION_LAB_lumnorm_lum.npy',PREDICTION_LAB)
PREDICTION_LAB = np.load('PREDICTION_LAB_lumnorm_lum.npy')

scatter_LAB(PREDICTION_LAB[:,-1,10,:], RGB_1600_muns)
DELTAE = DE_error_all(PREDICTION_LAB.T, MUNS_LAB.T)
np.mean(DELTAE,axis = (0,1))
np.median(DELTAE,axis = (0,1))
np.amin(DELTAE)
idx_min = np.unravel_index(np.argmin(DELTAE),[1600,28])


image = np.load(im_paths_mat[idx_min[0],idx_min[1]][:12]+'mnt/awesome/alban/'+im_paths_mat[idx_min[0],idx_min[1]][12:])
image_norm = image/np.amax(image)
image_norm = np.expand_dims(image_norm,axis = 2)
image_norm_ntr = np.repeat(image_norm,5,axis = 2)/mat_classic_cc_mat[idx_min[0],idx_min[1],:,:]

plt.imshow(image_norm[:,:,0,:])
plt.show()
plt.imshow((image_norm_ntr)[:,:,0,:])
plt.show()


# In[9]: Grey patch background
CHROMA_OBJ_MEAN_GP = np.zeros((1600,28,3))

'''
OBJ_GP = list()
for muns in range(len(im_paths_mat)):
#for muns in range(1):
	obj_gp = list()
	for exp in range(im_paths_mat.shape[1]):
		image = np.load(im_paths_mat[muns,exp][:12]+'mnt/awesome/alban/'+im_paths_mat[muns,exp][12:])
		image_norm = image/np.amax(image)
		image_norm = np.expand_dims(image_norm,axis = 2)
		GP = image_norm[10:17,54:62]
		ILLU_GP = np.mean(GP,axis = (0,1))
		ILLU_GP = ILLU_GP/np.amax(ILLU_GP)
		image_norm_ntr = image_norm/ILLU_GP
		mask = np.load(im_paths_mat[muns,exp][:12]+'mnt/awesome/alban/'+im_paths_mat[muns,exp][12:-4] + '_mask' + '.npy')
		mask_1 = np.mean(mask, axis = -1)
		mask_1[mask_1>0.2] = 1
		object_gp = (image_norm_ntr.T*mask_1.T).T
		object_gp = object_gp.reshape((-1,3))
		obj_gp.append(object_gp[object_gp[:,0]>0])
	OBJ_GP.append(obj_gp)

for muns in range(len(im_paths_mat)):
	np.save('/home/alban/mnt/DATA/project_color_constancy/classic_constancy/obj_segmented/munsell_%i_GP.npy'%muns,OBJ_GP[muns])
'''

for muns in range(len(im_paths_mat)):
#for muns in range(1):
	OBJECT_GP = np.load('/home/alban/mnt/DATA/project_color_constancy/classic_constancy/obj_segmented/munsell_%i_GP.npy'%muns)
	#OBJECT = OBJ[muns]
	for exp in range(im_paths_mat.shape[1]):
		object_GP = OBJECT_GP[exp]
		chroma_obj_mean = np.mean(object_GP,axis = 0)
		chroma_obj_mean = (chroma_obj_mean.T/(np.linalg.norm(chroma_obj_mean,axis = -1)/np.linalg.norm(MUNSELLS_LMS[muns]))).T
		CHROMA_OBJ_MEAN_GP[muns,exp] = chroma_obj_mean

PREDICTION_XYZ_GP = LMS2XYZ(CHROMA_OBJ_MEAN_GP).T
PREDICTION_LAB_GP = XYZ2Lab(PREDICTION_XYZ_GP,white = wh)

np.save('PREDICTION_LAB_mean_GP.npy',PREDICTION_LAB_GP)
PREDICTION_LAB_GP = np.load('PREDICTION_LAB_mean_GP.npy')

scatter_LAB(PREDICTION_LAB_GP[:,10,:], RGB_1600_muns)
DELTAE_GP = DE_error_all(PREDICTION_LAB_GP.T, MUNS_LAB.T)
np.mean(DELTAE_GP,axis = (0,1))
np.median(DELTAE_GP,axis = (0,1))


# In[9]: Grey patch background with sharpening
CHROMA_OBJ_MEAN_GP_sharpening = np.zeros((1600,28,3))


OBJ_GP = list()
for muns in range(len(im_paths_mat)):
#for muns in range(1):
	obj_gp = list()
	for exp in range(im_paths_mat.shape[1]):
		image = np.load(im_paths_mat[muns,exp][:12]+'mnt/awesome/alban/'+im_paths_mat[muns,exp][12:])
		image_norm = image/np.amax(image)
		GP = image_norm[10:17,54:62]
		ILLU_GP = np.mean(GP,axis = (0,1))
		ILLU_GP = ILLU_GP/np.amax(ILLU_GP)
		image_norm_ntr = Unsharpening(Sharpening(image_norm)/Sharpening(ILLU_GP))
		mask = np.load(im_paths_mat[muns,exp][:12]+'mnt/awesome/alban/'+im_paths_mat[muns,exp][12:-4] + '_mask' + '.npy')
		mask_1 = np.mean(mask, axis = -1)
		mask_1[mask_1>0.2] = 1
		object_gp = (image_norm_ntr.T*mask_1.T).T
		object_gp = object_gp.reshape((-1,3))
		obj_gp.append(object_gp[object_gp[:,0]>0])
	OBJ_GP.append(obj_gp)

for muns in range(len(im_paths_mat)):
	np.save('/home/alban/mnt/DATA/project_color_constancy/classic_constancy/obj_segmented/munsell_%i_GP_sharpening.npy'%muns,OBJ_GP[muns])


for muns in range(len(im_paths_mat)):
#for muns in range(1):
	OBJECT_GP = np.load('/home/alban/mnt/DATA/project_color_constancy/classic_constancy/obj_segmented/munsell_%i_GP_sharpening.npy'%muns)
	#OBJECT = OBJ[muns]
	for exp in range(im_paths_mat.shape[1]):
		object_GP = OBJECT_GP[exp]
		chroma_obj_mean = np.mean(object_GP,axis = 0)
		chroma_obj_mean = (chroma_obj_mean.T/(np.linalg.norm(chroma_obj_mean,axis = -1)/np.linalg.norm(MUNSELLS_LMS[muns]))).T
		CHROMA_OBJ_MEAN_GP_sharpening[muns,exp] = chroma_obj_mean

PREDICTION_XYZ_GP_sharpening = LMS2XYZ(CHROMA_OBJ_MEAN_GP_sharpening).T
PREDICTION_LAB_GP_sharpening = XYZ2Lab(PREDICTION_XYZ_GP_sharpening,white = wh)

#np.save('PREDICTION_LAB_mean_GP.npy',PREDICTION_LAB_GP_sharpening)
#PREDICTION_LAB_GP_sharpening = np.load('PREDICTION_LAB_mean_GP.npy')

scatter_LAB(PREDICTION_LAB_GP_sharpening[:,10,:], RGB_1600_muns)
DELTAE_GP_sharpening = DE_error_all(PREDICTION_LAB_GP_sharpening.T, MUNS_LAB.T)
np.mean(DELTAE_GP_sharpening,axis = (0,1))
np.median(DELTAE_GP_sharpening,axis = (0,1))

