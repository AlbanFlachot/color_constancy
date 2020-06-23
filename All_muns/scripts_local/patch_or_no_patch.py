
from __future__ import print_function, division

import matplotlib.pyplot as plt
import numpy as np
import pickle
import scipy.io as sio
import os



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


def LMS2Opp(x):
	M = np.array([[ 0.67,  1.0,  0.5],[ 0.67,  -1,  0.5],[0.67,  0,  -1]])
	return np.dot(x,M)


nb_models = 10

WCS_LAB = np.load(npy_dir_path + 'WCS_LAB.npy') # Used for procrustes analysis
RGB_muns = np.load(npy_dir_path + 'RGB_WCS.npy') # Used for display
inds_330 = np.load(npy_dir_path + 'WCS_muns_index.npy') # index of the 330 WCS muns amoung 1600

WCS_MUNS_XYValue_coord = np.load(npy_dir_path + 'MUNSELL_COORDINATES.npy') # Used for procrustes analysis
#dis.scatter_MDS2(WCS_MUNS_XYValue_coord.T,'','figures/procrustes/WCS_Muns_Coor_3D.png','figures/procrustes/WCS_Muns_Coor_2D.png', RGB_muns)

list_activation_path = '/home/alban/mnt/awesome/works/color_constancy/All_muns/scripts_server/layers/all/'

activations_name = '%s_Original_CC_D65_masks_WCS_%s.npy'

MUNSELLS_LMS = np.load(npy_dir_path + 'MUNSELLS_LMS.npy')

MUNSELLS_LGN = LMS2Opp(MUNSELLS_LMS-MUNSELLS_LMS.mean(axis = 0))


conditions = ['normal','no_patch', 'wrong_illu', 'no_back', 'D65']

layers = ['fc2','fc1', 'conv3', 'conv2', 'conv1']

DIFF = {}
for layer in layers:
        activations_patch = np.load(list_activation_path + activations_name %(layer, conditions[0]))
        activations_nopatch = np.load(list_activation_path + activations_name %(layer, conditions[1]))
        
        DIFF[layer] = (activations_patch - activations_nopatch)/np.amax((activations_patch,activations_nopatch), axis = 0)



fig = plt.figure()
ax1 = fig.add_subplot(111)
[plt.hist(DIFF[layer].mean(axis = (1,2,3))[0], label = layer) for layer in layers[::-1]]
plt.xlabel('Difference')
plt.ylabel('count')
#plt.xlim(-1,40)
#ax1.set_ylabel('Median CCI')
plt.legend()
fig.tight_layout()
#fig.savefig(figures_dir_path +'affine.png', dpi=400)
plt.show()

        
        
