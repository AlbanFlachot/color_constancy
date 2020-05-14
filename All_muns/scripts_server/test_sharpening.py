from __future__ import print_function, division

import numpy as np
import re

import sys
sys.path.append('../../')

from color_scripts import color_transforms as CT # scripts with a bunch of functions for color transformations

from utils_scripts import algos
from utils_scripts import computations_classic_CC as comp

import warnings

warnings.filterwarnings("ignore")


# In[9]: path to files

txt_dir_path = '../../txt_files/'
npy_dir_path = '../../npy_files/'
pickles_dir_path = '../pickles/'
figures_dir_path = '../figures/'

# In[9]: INITIALIZATIONS

im_paths = list()
with open(txt_dir_path + "5_luminants_im_paths.txt") as f:
    for line in f:
       im_paths.append(line.split())


train_labels = [int(re.search('object(.*?)/', addr[0]).group(1)) for addr in im_paths]


nb_muns = 330
nb_test_muns = 330
nb_illu = 5
nb_exp = 5
nb_algos = 5

im_paths2 = [('/home/alban/DATA/IM_CC/'+ p[0][35:]) for p in im_paths]

mat_classic_cc = np.load(npy_dir_path + '5_luminants.npy')
mat_classic_cc_mat = np.empty((nb_muns,nb_illu, nb_exp, nb_algos,3))
im_paths_mat = np.empty((nb_muns,nb_illu,nb_exp),dtype=np.object)
count = 0
for muns in range(nb_muns):
	for illu in range(nb_illu):
		for exp in range(nb_exp):
			im_paths_mat[train_labels[count],illu,exp] = im_paths2[count]
			mat_classic_cc_mat[train_labels[count], illu, exp] = mat_classic_cc[count]
			count +=1

mat_classic_cc_mat = ((mat_classic_cc_mat.T)/np.mean((mat_classic_cc_mat.T), axis = 0)).T

#list_WCS_labels = algos.compute_WCS_Munsells_categories()

#mat_classic_cc_WCS = mat_classic_cc_mat[list_WCS_labels]
#im_paths_mat_WCS = im_paths_mat[list_WCS_labels]

 

MUNSELLS_LMS = np.load(npy_dir_path +'LMS_WCS_D65.npy')

#extract_max_pix = np.array([0,0,0])
#for i in im_paths:
#    extract_max_pix = np.amax((extract_max_pix, np.amax(np.load(i[0][:12]+'mnt/awesome/alban/'+i[0][12:]), axis = (0,1))), axis = 0)

#extract_min_pix = np.array([3,3,3])
#for i in im_paths:
#    extract_min_pix = np.amin((extract_min_pix, np.amin(np.load(i[0][:12]+'mnt/awesome/alban/'+i[0][12:]), axis = (0,1))), axis = 0)


max_pxl = np.array([22.2250821 , 25.19385033, 25.49259802])
min_pxl = np.array([0.00260734, 0.00236998, 0.00086242])

wh  = np.array([17.95504939, 18.96455292, 20.36406225])
XYZ_MUNS = CT.LMS2XYZ(MUNSELLS_LMS).T

MUNS_LAB = CT.XYZ2Lab(XYZ_MUNS,white = wh)

RGB_WCS = np.load(npy_dir_path + 'RGB_WCS.npy')
#dis.scatter_LAB(MUNS_LAB, RGB_WCS)


# In[9]: COMPUTATIONS

def scaling(X, Y):
    '''
    Function to scale X according to Y
    '''
    
    maxX = X[0].max()
    minX = X[-1].min()
    
    maxY = Y[0].max()
    minY = Y[-1].min()
    
    return (X - minX)*((maxY-minY)/(maxX-minX)) + minY

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
				if correction:
					image_norm = Sharpening(image_norm)
					mat_classic_cc_mat_sharp = Sharpening(mat_classic_cc_mat)
					if mat_classic_cc_mat.size == 3:
						image_norm_ntr = image_norm/(mat_classic_cc_mat_sharp/np.mean(mat_classic_cc_mat_sharp))
					elif mat_classic_cc_mat.size == shape[1]*shape[-1]:
						image_norm_ntr = image_norm/(mat_classic_cc_mat_sharp[illu]/np.mean(mat_classic_cc_mat_sharp[illu]))
					else:
						image_norm_ntr = (np.repeat(image_norm,shape[3],axis = 2)/(mat_classic_cc_mat_sharp[muns,illu,exp,:,:]/np.mean(mat_classic_cc_mat_sharp[muns,illu,exp,:,:])))
					image_norm_ntr = Unsharpening(image_norm_ntr*Sharpening(np.array([0.95595725, 1.00115551, 1.04288724]))) # D65 chromaticity
				mask = np.load(im_paths_mat[muns,illu,exp][:-4] + '_mask' + '.npy')
				mask_1 = np.mean(mask, axis = -1)
				mask_1[mask_1>0.2] = 1
				if correction:
					object = (image_norm_ntr.T*mask_1.T).T
				else:
					object = (image_norm.T*mask_1.T).T
				object = object.reshape((-1, shape[3], 3))
				object = object[object[:,0,0]!=0] # we consider only the croped objct pixels
				chroma_obj_mean = np.mean(object,axis = 0)
				if np.isnan(chroma_obj_mean[0,0]):
					import pdb; pdb.set_trace()
				chroma_obj_median = np.median(object,axis = 0)
				most_bright = np.argsort(np.sum(object, axis = -1),axis = 0)[-10:]
				chroma_obj_lum = np.mean(object[most_bright[:,0],:],axis = 0)
				CHROMA_OBJ_MEAN[muns,illu,exp] = chroma_obj_mean
				CHROMA_OBJ_LUM[muns,illu,exp] = chroma_obj_lum
				CHROMA_OBJ_MEDIAN[muns,illu,exp] = chroma_obj_median
	return CHROMA_OBJ_MEAN, CHROMA_OBJ_LUM, CHROMA_OBJ_MEDIAN

def Sharpening2(x):
    M_diag = np.array([[ 0.7408369 , -0.37109953,  0.02111892],
       [-0.67088307,  0.92347209, -0.0424343 ],
       [ 0.03281137, -0.09738812,  0.99887603]])
    Sharp_x = np.dot(x, M_diag)
    return Sharp_x

def Sharpening(x):
    M_diag = np.array([[ 1.6934   , -1.5335   ,  0.075    ],
       [-0.5341875,  1.3293125, -0.1401875],
       [ 0.0215   , -0.0432   ,  1.0169   ]])
    Sharp_x = np.dot(x, M_diag.T)
    return Sharp_x

def Unsharpening2(x):
    M_diag = np.array([[ 2.1213585 ,  0.85155825, -0.00867528],
       [ 1.54484144,  1.70787505,  0.03989194],
       [ 0.0809355 ,  0.13854166,  1.00529958]])
    Unsharp_x = np.dot(x, M_diag)
    return Unsharp_x

def Unsharpening(x):
    M_diag = np.array([[ 0.92806228,  1.07319983,  0.07950096],
       [ 0.37254386,  1.18645912,  0.13608609],
       [-0.0037953 ,  0.02771289,  0.98748122]])
    Unsharp_x = np.dot(x, M_diag.T)
    return Unsharp_x


def computations_chromaticities(im_paths_mat, illus, shape, MUNSELLS_LMS, max_pxl, correction = True , algo = 0):
    CHROMA_OBJ_MEAN, CHROMA_OBJ_LUM, CHROMA_OBJ_MEDIAN = predicted_chromaticity(im_paths_mat,illus, shape, max_pxl, correction)

    chroma_mean = scaling(np.mean(CHROMA_OBJ_MEAN, axis = (2)), MUNSELLS_LMS)
    
    #import pdb; pdb.set_trace()
    PREDICTION_XYZ = CT.LMS2XYZ(chroma_mean).T
    PREDICTION_LAB = CT.XYZ2Lab(PREDICTION_XYZ,white = wh)
    
    DELTAE = comp.DE_error_all(PREDICTION_LAB[:,algo].T, MUNS_LAB.T)
    print(np.mean(DELTAE,axis = (0)))
    print(np.median(DELTAE,axis = (0)))
    return {'mean_LMS': CHROMA_OBJ_MEAN, 'median_LMS': CHROMA_OBJ_MEDIAN, 'bright_LMS': CHROMA_OBJ_LUM, 'XYZ': PREDICTION_XYZ, 'LAB': PREDICTION_LAB}



# no correction and no sharpening

vs = np.array([0.8,0.8,0.8])

u = np.array([0.95595725, 1.00115551, 1.04288724])
#u = np.array([1, 1, 1])

S = vs/u


# no correction and sharpening

vs_sharp = Sharpening(vs)

u_sharp = Sharpening(u)

S_sharp = vs_sharp/u_sharp

vs2 = Unsharpening(S_sharp*u_sharp)

# In[9]: ref: chromaticity under D65

DICT_ref = computations_chromaticities(im_paths_mat[:,3].reshape((nb_test_muns,1,nb_exp)), np.array([1,1,1]), (nb_test_muns, 1, nb_exp,1,3), MUNSELLS_LMS, max_pxl, correction = False)
#DICT_ref = np.load('DICT_ref.npy',allow_pickle = True)[True][0]

# In[9]: Controle: perfect estimation illu with von kries

LMS_4illu = np.array([[0.8608155 , 0.87445185, 0.77403174],[0.78124027, 0.84468902, 1.04376177], [0.87937024, 0.95460385, 0.97006115],[0.77854056, 0.79059459, 0.86517277]])

DICT_vK = computations_chromaticities(im_paths_mat[:,[0,1,2,4]], LMS_4illu, (nb_test_muns, 4, nb_exp, 1,3), MUNSELLS_LMS, max_pxl)

#np.save('DICT_vK.npy', DICT_vK)
#DICT_vK = np.load('DICT_vK.npy',allow_pickle = True)[True][0]

print('Error von Kries:')
print(np.nanmedian(np.linalg.norm(DICT_vK['LAB'][:,:,0] - DICT_ref['LAB'][:,:,0], axis = 0)))
error_vK = np.nanmedian(np.linalg.norm(DICT_vK['LAB'] - DICT_vK['LAB'][:,0], axis = 0))

