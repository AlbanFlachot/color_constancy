
# Script that computes correlations in activation between munsells, dissimilarity matrices and computes MDS

import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys
import re
import os
from scipy.spatial.transform import Rotation as R


import sys
sys.path.append('../../')

from color_scripts import color_transforms as CT # scripts with a bunch of functions for color transformations
from error_measures_scripts import Error_measures as EM
from display_scripts import display as dis
from utils_scripts import algos


# Ignore warnings
import warnings

warnings.filterwarnings("ignore")

# In[9]: path to files

txt_dir_path = '../../txt_files/'
npy_dir_path = '../../npy_files/'
pickles_dir_path = '../pickles/'
figures_dir_path = '../figures/'


root_dir = '../../'


# In[9]: Function
def LMS2Opp(x):
	M = np.array([[ 0.67,  1.0,  0.5],[ 0.67,  -1,  0.5],[0.67,  0,  -1]])
	return np.dot(x,M)

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
#compute_and_display_MDS(layer)

def affine_transform(X, Y):
    # Normalize coordinates
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
    #import pdb; pdb.set_trace()
    # Compute the affine transformation using homogenous coordinates
    hom_X0 = np.vstack([X0.T, np.ones(len(X0))]).T
    hom_Y0 = np.vstack([Y0.T, np.ones(len(Y0))]).T
    
    affine_mat, R, rank, S = np.linalg.lstsq(hom_X0, hom_Y0, rcond=None)
    
    
    return affine_mat, R, S


def apply_affine_transform(affine_mat, pts):
    hom_pts = np.vstack([pts.T, np.ones(len(pts))]).T
    #import pdb; pdb.set_trace()
    tmp = np.dot(hom_pts, affine_mat)
    #out_pts = np.array([x[:-1] / x[-1] for x in tmp.T])
    
    return tmp

def compute_and_display_MDS(layer,save_path):
	'''
	Function that compute a similarity analysis on the activations of a given DNN layer
	'''
	corr_layer = algos.correlations_layers(layer)
	#import pdb; pdb.set_trace()

	DM_layer = 1 - np.mean(corr_layer,axis = 0)

	RESULTlayer, stress_layer = algos.MDS(DM_layer)
	coeff,score,latent,explained = algos.princomp(RESULTlayer)

	np.sum((stress_layer/np.sum(stress_layer[stress_layer>0]))[0:5])

	#dis.scatter_MDS_vert(RESULTlayer[:,:4],' ', save_path + '3D.png',save_path +'2D.png',RGB_muns)
	return RESULTlayer, stress_layer, explained, score

# In[9]: LOAD ACTIVATIONS color constant

nb_models = 10

WCS_LAB = np.load(npy_dir_path + 'WCS_LAB.npy') # Used for procrustes analysis
RGB_muns = np.load(npy_dir_path + 'RGB_WCS.npy') # Used for display
inds_330 = np.load(npy_dir_path + 'WCS_muns_index.npy') # index of the 330 WCS muns amoung 1600

WCS_MUNS_XYValue_coord = np.load(npy_dir_path + 'MUNSELL_COORDINATES.npy') # Used for procrustes analysis
#dis.scatter_MDS2(WCS_MUNS_XYValue_coord.T,'','figures/procrustes/WCS_Muns_Coor_3D.png','figures/procrustes/WCS_Muns_Coor_2D.png', RGB_muns)

list_activation_path = '/home/alban/mnt/awesome/works/color_constancy/All_muns/scripts_server/layers/'

activations_name = '%s_Original_CC_D65_masks_WCS_%s.npy'

MUNSELLS_LMS = np.load(npy_dir_path + 'MUNSELLS_LMS.npy')

MUNSELLS_LGN = LMS2Opp(MUNSELLS_LMS-MUNSELLS_LMS.mean(axis = 0))


conditions = ['normal','no_patch', 'wrong_illu', 'no_back', 'D65']

layers = ['fc2','fc1', 'conv3', 'conv2', 'conv1']

EXPLAINED = {}
PROCRUSTES_LAB = {}
PROCRUSTES_LGN = {}
PROCRUSTES_LMS = {}
PROCRUSTES_MUNS = {}
AFFINE_LGN = {}
AFFINE_LAB = {}
TRANSFO_LAB = {}

def extract_procrustes( ref_coord, result_MDS):
        disp_layer, score_procrustes, tform_procrustes = algos.procrustes( ref_coord, result_MDS[:,:3] )
        return disp_layer, tform_procrustes
        
        
count = 0
for layer in layers:
    PROCRUSTES_LAB[layer] = {}
    PROCRUSTES_LGN[layer] = {}
    PROCRUSTES_LMS[layer] = {}
    PROCRUSTES_MUNS[layer] = {}
    EXPLAINED[layer] = {}
    AFFINE_LGN[layer] = {}
    AFFINE_LAB[layer] = {}
    TRANSFO_LAB[layer] = {}
    for condition in conditions[:1]:

        activations = np.load(list_activation_path + activations_name %(layer, condition))

        #activations = activations.mean(axis=(-3,-2))

        num_samples = 20
        #f_size = np.array(activations.shape[1:])
        input_to_mds = activations.reshape(nb_models, 330,num_samples,-1)


        result_MDS, stress, explained, score_MDS = compute_and_display_MDS(input_to_mds, figures_dir_path +'MDS/'+layer +'_' + condition)

        PROCRUSTES_LAB[layer][condition],TRANSFO_LAB[layer][condition] = extract_procrustes( WCS_LAB.T, result_MDS[:,:3] )
        affine_mat_LAB, residuals_LAB, S_LAB = affine_transform( result_MDS[:,:3], WCS_LAB.T )
        AFFINE_LAB[layer][condition] = (residuals_LAB).sum()
        dis.scatter_MDS2(apply_affine_transform( affine_mat_LAB, result_MDS[:,:3]),'',figures_dir_path +'/affine/'+layer +'_' + condition + '_LAB_3D.png',figures_dir_path +'affine/'+layer +'_' + condition + '_LAB_2D.png', RGB_muns, display = False)
        
        #dis.scatter_MDS2(score_procrustes,'',figures_dir_path +'/procrustes/'+layer +'_' + condition + '_LAB_3D.png',figures_dir_path +'procrustes/'+layer +'_' + condition + '_LAB_2D.png', RGB_muns, display = False)
        #print('Result procrustes is: %f' %disp_layer)
        
        PROCRUSTES_LGN[layer][condition] = extract_procrustes( MUNSELLS_LGN[algos.compute_WCS_Munsells_categories()], result_MDS[:,:3])[0]
        
        PROCRUSTES_LMS[layer][condition] = extract_procrustes( MUNSELLS_LMS[algos.compute_WCS_Munsells_categories()], result_MDS[:,:3])[0]
        
        PROCRUSTES_MUNS[layer][condition] = extract_procrustes( WCS_MUNS_XYValue_coord.T, result_MDS[:,:3])[0]

        affine_mat_LGN, residuals_LGN, S_LGN = affine_transform(result_MDS[:,:3], MUNSELLS_LGN[algos.compute_WCS_Munsells_categories()] )
        AFFINE_LGN[layer][condition] = residuals_LGN.sum()

        EXPLAINED[layer][condition] = explained[:4]
        count +=1


dis.DEFINE_PLT_RC(type = 1)

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot([1,2,3,4,5],[PROCRUSTES_LAB[layer]['normal'] for layer in layers[::-1]],'k', label = 'CIELab')
ax1.plot([1,2,3,4,5],[PROCRUSTES_LGN[layer]['normal'] for layer in layers[::-1]],color = [0.5,0.5,0.5], label = 'LGN')
ax1.plot([1,2,3,4,5],[PROCRUSTES_MUNS[layer]['normal'] for layer in layers[::-1]],color = 'b', label = 'Munsell')
ax1.plot([1,2,3,4,5],[PROCRUSTES_LMS[layer]['normal'] for layer in layers[::-1]],color = [0.25,0.25,0.25], label = 'LMS')
ax1.set_xticks([1,2,3,4,5])
ax1.set_xticklabels(layers[::-1])
plt.xlabel('Layers')
plt.ylabel('R$^2$')
#plt.ylim(0,0.4)
#ax1.set_yticks(np.arange(0,0.41,0.2))
#ax1.set_ylabel('Median CCI')
plt.legend()
fig.tight_layout()
fig.savefig(figures_dir_path +'procrustes.png', dpi=400)
plt.show()

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot([1,2,3,4,5],[AFFINE_LAB[layer]['normal'] for layer in layers[::-1]],'k', label = 'CIELab')
#ax1.plot([1,2,3,4,5],[AFFINE_LGN[layer]['normal'] for layer in layers[::-1]],color = [0.6,0.6,0.6], label = 'LGN')
ax1.set_xticks([1,2,3,4,5])
ax1.set_xticklabels(layers[::-1])
plt.xlabel('Layers')
plt.ylabel('R$^2$')
#plt.xlim(-1,40)
#ax1.set_ylabel('Median CCI')
#plt.legend()
fig.tight_layout()
fig.savefig(figures_dir_path +'affine.png', dpi=400)
plt.show()


for i in [TRANSFO_LAB[layer]['normal'] for layer in layers[::-1]]:
        #print(np.round(i['rotation'],2))
        M = np.round(i['rotation'],2)
        
        Mr = R.from_matrix(M.T).as_euler('xyz', degrees = True)
        print(Mr.astype(int))
        
        permutation = np.argmax(np.absolute(M), axis = 0)
        idx = np.empty_like(permutation)
        idx[permutation] = np.arange(len(permutation))
        
        M_pos = M[:,idx]
        M_neg = -M[:,idx]
        Mr_pos = R.from_matrix(M_pos.T).as_euler('xyz', degrees = True)
        Mr_neg = R.from_matrix(M_neg.T).as_euler('xyz', degrees = True)
        print(Mr_pos.astype(int), Mr_neg.astype(int))
        print(np.absolute(Mr_pos).sum().astype(int), np.absolute(Mr_neg).sum().astype(int))










'''

WCS_MUNS_XYValue_coord = np.load(npy_dir_path + 'MUNSELL_COORDINATES.npy')


disp_layer, score_procrustes, tform_procrustes = algos.procrustes( WCS_MUNS_XYValue_coord.T, result_MDS[:,:3])
dis.scatter_MDS2(score_procrustes,'','figures/procrustes_'+layer_name +'_' + network_name + '3D.png','figures/procrustes_'+layer_name +'_' + network_name+ '2D.png', RGB_muns)

print('Result procrustes is: %f' %disp_layer)

'''

