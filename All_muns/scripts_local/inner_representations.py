
# Script that computes correlations in activation between munsells, dissimilarity matrices and computes MDS

import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys
import re
import os
import scipy.stats as stats


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

	dis.scatter_MDS_vert(RESULTlayer[:,:4],' ', save_path + '3D.png',save_path +'2D.png',RGB_muns)
	return RESULTlayer, stress_layer, explained, score

# In[9]: LOAD ACTIVATIONS color constant

nb_models = 10

WCS_LAB = np.load(npy_dir_path + 'WCS_LAB.npy') # Used for procrustes analysis
RGB_muns = np.load(npy_dir_path + 'RGB_WCS.npy') # Used for display
inds_330 = np.load(npy_dir_path + 'WCS_muns_index.npy') # index of the 330 WCS muns amoung 1600

WCS_MUNS_XYValue_coord = np.load(npy_dir_path + 'MUNSELL_COORDINATES.npy') # Used for procrustes analysis
#dis.scatter_MDS2(WCS_MUNS_XYValue_coord.T,'','figures/procrustes/WCS_Muns_Coor_3D.png','figures/procrustes/WCS_Muns_Coor_2D.png', RGB_muns)

list_activation_path = '/home/alban/mnt/awesome/works/color_constancy/All_muns/scripts_servers/layers/'

activations_name = '%s_Original_CC_4illu_WCS_%s.npy'

MUNSELLS_LMS = np.load(npy_dir_path + 'MUNSELLS_LMS.npy')

MUNSELLS_LGN = LMS2Opp(MUNSELLS_LMS-MUNSELLS_LMS.mean(axis = 0))


conditions = ['normal','no_patch', 'wrong_illu', 'no_back', 'D65']

layers = ['fc2','fc1', 'conv3', 'conv2', 'conv1']

EXPLAINED = {}
PROCRUSTES_LAB = {}
PROCRUSTES_LGN = {}
PROCRUSTES_LMS = {}

count = 0
for layer in layers:
    PROCRUSTES_LAB[layer] = {}
    PROCRUSTES_LGN[layer] = {}
    PROCRUSTES_LMS[layer] = {}
    EXPLAINED[layer] = {}
    for condition in conditions[:2]:

    	activations = np.load(list_activation_path + activations_name %(layer, condition))

    	#activations = activations.mean(axis=(-3,-2))

    	num_samples = 40
    	#f_size = np.array(activations.shape[1:])
    	input_to_mds = activations.reshape(nb_models, 330,num_samples,-1)


    	result_MDS, stress, explained, score_MDS = compute_and_display_MDS(input_to_mds, figures_dir_path +'/MDS/'+layer +'_' + condition)


        # procrustes analysis with respect to munsell space or LAB space
        ###----------------------------------------------------------------------------------------------------------
    	disp_layer, score_procrustes, tform_procrustes = algos.procrustes( WCS_LAB.T, result_MDS[:,:3])
    	dis.scatter_MDS2(score_procrustes,'',figures_dir_path +'/procrustes/'+layer +'_' + condition + '_LAB_3D.png',figures_dir_path +'/procrustes/'+layer +'_' + condition + '_LAB_2D.png', RGB_muns, display = False)
    	#print('Result procrustes is: %f' %disp_layer)

    	PROCRUSTES_LAB[layer][condition] = disp_layer

        disp_layer_LGN, score_procrustes_LGN, tform_procrustes_LGN = algos.procrustes( MUNSELLS_LGN[algos.compute_WCS_Munsells_categories()], result_MDS[:,:3])
        PROCRUSTES_LGN[layer][condition] = disp_layer_LGN

        disp_layer_LMS, score_procrustes_LMS, tform_procrustes_LMS = algos.procrustes( MUNSELLS_LMS[algos.compute_WCS_Munsells_categories()], result_MDS[:,:3])
        PROCRUSTES_LMS[layer][condition] = disp_layer_LMS


    	EXPLAINED[layer][condition] = explained[:4]
    	count +=1


dis.DEFINE_PLT_RC(type = 1)

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot([1,2,3,4,5],[PROCRUSTES_LAB[layer]['normal'] for layer in layers[::-1]],'k', label = 'CIELab')
#ax1.plot([1,2,3,4,5],[PROCRUSTES_LMS[layer]['normal'] for layer in layers[::-1]],color = [0.6,0.6,0.6], label = 'LMS')
ax1.set_xticks([1,2,3,4,5])
ax1.set_xticklabels(layers[::-1])
plt.xlabel('Layers')
plt.ylabel('R$^2$')
#plt.xlim(-1,40)
#ax1.set_ylabel('Median CCI')
#plt.legend()
fig.tight_layout()
#fig.savefig(figures_dir_path +'CCI_vs_hue.png', dpi=400)
plt.show()


'''
WCS_MUNS_XYValue_coord = np.load(npy_dir_path + 'MUNSELL_COORDINATES.npy')


disp_layer, score_procrustes, tform_procrustes = algos.procrustes( WCS_MUNS_XYValue_coord.T, result_MDS[:,:3])
dis.scatter_MDS2(score_procrustes,'','figures/procrustes_'+layer_name +'_' + network_name + '3D.png','figures/procrustes_'+layer_name +'_' + network_name+ '2D.png', RGB_muns)

print('Result procrustes is: %f' %disp_layer)'''



