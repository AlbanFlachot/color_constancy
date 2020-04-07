# In[1]:



from __future__ import print_function, division

import matplotlib.pyplot as plt
import numpy as np
import pickle
import scipy.io as sio
import os
import torch


import sys
sys.path.append('../../')

from color_scripts import color_transforms as CT # scripts with a bunch of functions for color transformations
from error_measures_scripts import Error_measures as EM
from display_scripts import display as dis
from utils_scripts import algos
from DL_testing_functions_scripts import DL_testing_functions as DLtest
from DL_training_functions_scripts import MODELS as M
# Ignore warnings
import warnings

warnings.filterwarnings("ignore")

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

# In[38] PARSER definition
import argparse

parser = argparse.ArgumentParser(description='Parsing variables for rendering images')

parser.add_argument('--training_set', default='CC', type=str, metavar='str',
                    help='to distiguish between CC and D65')

parser.add_argument('--NetType', default='Original', type=str, metavar='str',
                    help='type of model to analyze')


parser.add_argument('--path2weights', default='/home/alban/mnt/awesome/alban/project_color_constancy/PYTORCH/WCS/train_centered/All_muns/', type=str, metavar='str',
                    help='path to directory where activations (npy files) are')


args = parser.parse_args()


# In[9]: path to files

txt_dir_path = '../../txt_files/'
npy_dir_path = '../../npy_files/'
pickles_dir_path = '../pickles/'
figures_dir_path = '../figures/'

# In[9]: Initialization variables


NetType = args.NetType


DIR_LOAD = args.path2weights
#path2activations = '/home/alban/mnt/awesome/arash/Software/repositories/kernelphysiology/python/src/kernelphysiology/dl/pytorch/activations/Test_4_illu/fc/wcs_lms_1600/original/'
#path2activations = '/home/alban/mnt/awesome/arash/Software/repositories/kernelphysiology/python/src/kernelphysiology/dl/pytorch/activations/fc/wcs_lms_1600/original/'


TRAINING_CURV, EPOCHMAX = DLtest.training_curves(DIR_LOAD + 'INST_%s/'%(args.training_set),args.training_set, 90)
nb_models = len(TRAINING_CURV)

if args.NetType == 'Ref':
    net = M.Ref()
elif args.NetType == 'Original':
    net = M.Net2_4ter_norm()


W = np.zeros((nb_models,net.conv1.weight.size()[0],net.conv1.weight.size()[2],net.conv1.weight.size()[2],3))

for m in range(nb_models):
    print('Evaluation of model %i' %(m+1))
    weights = DIR_LOAD +'INST_%s/inst_%d_%s/epoch_%i.pt' %((args.training_set,m,args.training_set,EPOCHMAX[m]))
    net.load_state_dict(torch.load(weights, map_location='cpu'))
    net.to(device)
    net.eval()

    w1 = net.conv1.weight
    b1 = net.conv1.bias
    w1 = w1.cpu().detach().numpy()
    b1 = b1.cpu().detach().numpy()

    #for j in range(len(b1)):
    #    w1[j] = w1[j] + b1[j]
    w1 = np.moveaxis(w1,1,-1)
    W[m] = w1



# In[9]: VISUALIZE FILTERS 1st LAYER

for m in range(0,nb_models):
    data = dis.vis_square(W[m],figures_dir_path + 'filters_inst_%s_%i' %(args.training_set, m))


# In[9]:
## Plot az and el layer 1

def LMS2Opp(x):
	M = np.array([[ 0.67,  1.0,  0.5],[ 0.67,  -1,  0.5],[ 0.67,  0,  -1]])
	return np.dot(x,M)

WEIGHTS1 = np.zeros((W.shape[0],W.shape[1],3))
EXPLAINED1 = np.zeros((W.shape[0],W.shape[1]))

for i in range(len(W)):
    weights_filt = np.zeros((len(w1),3))
    expl_filt = np.zeros(len(w1))
    for f in range(0,w1.shape[0]):
        fil2 = W[i][f].reshape((net.conv1.weight.size()[2]*net.conv1.weight.size()[2],3))
        coeff,score,latent,explained = algos.princomp(fil2)
        weights_filt[f] = coeff[:,0]
        expl_filt[f] = explained[0]
    WEIGHTS1[i] = weights_filt
    EXPLAINED1[i] = expl_filt

DWEIGHTS1 = WEIGHTS1.reshape((-1,WEIGHTS1.shape[-1]))

dweights_filt = LMS2Opp(DWEIGHTS1)

rho, az, el = CT.cart2sph(dweights_filt[:,1], dweights_filt[:,2], dweights_filt[:,0])
az = az*180/np.pi
el = el * 180/np.pi

'''
AZIMUTH[Fi] = az.reshape((1,az.size))
ELEVATION[Fi] = el.reshape((1,el.size))
if Fi == 0:
	azimuth = az.reshape((1,az.size))
	elevation = el.reshape((1,el.size))
else:
	azimuth = np.concatenate((azimuth,az.reshape((1,az.size))), axis = 0)
	elevation = np.concatenate((elevation,el.reshape((1,el.size))), axis = 0)'''


## In[9]:
### Figure visualization weights
#
#w1_opp = LMS2Opp(w1).reshape((w1.shape[0],-1,3))
#	## Figure scatter plot preferred elevation and azimuth
#fig = plt.figure(figsize=(6,5.4))
#ax=plt.subplot(1,1,1)
#p=plt.plot(w1_opp[1,:,1],w1_opp[1,:,2],'o',color = [0.25,0.25,0.25])
#coeff,score,latent,explained = princomp(w1_opp[1])
#mean_dis = np.mean(w1_opp[1],0)
#plt.plot([mean_dis[1]-coeff[:,0][1],mean_dis[1]+coeff[:,0][1]],[mean_dis[2]-coeff[:,0][2],mean_dis[2]+coeff[:,0][2]],lw = 3,color = 'orange')
#ax.axhline(0, xmin=-1, xmax=1, color='k', ls='-',linewidth = 1, label='Tresh')
#ax.axvline(0, ymin=-1, ymax=1, color='k', ls='-',linewidth = 1, label='Tresh')
#plt.setp(ax.get_xticklabels(), fontsize=20)
#plt.setp(ax.get_yticklabels(), fontsize=20)
#plt.xlim(-0.5,0.5)
#plt.ylim(-0.5,0.5)
##plt.xlabel('Epoch',fontsize = 25)
##plt.ylabel('Accuracy validation (%)',fontsize = 25)
##plt.xticks((np.arange(-1,1.1,0.5)))
##plt.yticks((np.arange(-1,1.1,0.5)))
#fig.tight_layout()
#fig.text(0.92, 0.51, 'L-M', ha='center',color = 'k', fontsize = 20)
#fig.text(0.56, 0.92, 'L+M-S', ha='center',color = 'k', fontsize = 20)
#plt.show()
# In[9]: VISUALIZE FILTERS 1st LAYER

#fig.savefig('project_color_constancy/mod_N2/PC_weights1.png')
#plt.close()
#
#
#w1_opp = LMS2Opp(w1).reshape((w1.shape[0],-1,3))
#	## Figure scatter plot preferred elevation and azimuth
#fig = plt.figure(figsize=(6,5.4))
#ax=plt.subplot(1,1,1)
#p=plt.plot(w1_opp[1,:,0],w1_opp[1,:,1],'o',color = [0.25,0.25,0.25])
#coeff,score,latent,explained = princomp(w1_opp[1])
#mean_dis = np.mean(w1_opp[1],0)
#plt.plot([mean_dis[0]-coeff[:,0][0],mean_dis[0]+coeff[:,0][0]],[mean_dis[1]-coeff[:,0][1],mean_dis[1]+coeff[:,0][1]],lw = 3,color = 'orange')
#ax.axhline(0, xmin=-1, xmax=1, color='k', ls='-',linewidth = 1, label='Tresh')
#ax.axvline(0, ymin=-1, ymax=1, color='k', ls='-',linewidth = 1, label='Tresh')
#plt.setp(ax.get_xticklabels(), fontsize=20)
#plt.setp(ax.get_yticklabels(), fontsize=20)
#plt.xlim(-0.5,0.5)
#plt.ylim(-0.5,0.5)
##plt.xlabel('Epoch',fontsize = 25)
##plt.ylabel('Accuracy validation (%)',fontsize = 25)
##plt.xticks((np.arange(-1,1.1,0.5)))
##plt.yticks((np.arange(-1,1.1,0.5)))
#fig.tight_layout()
#fig.text(0.92, 0.51, 'L-M', ha='center',color = 'k', fontsize = 20)
#fig.text(0.56, 0.92, 'L+M+S', ha='center',color = 'k', fontsize = 20)
#plt.show()
#fig.savefig('project_color_constancy/mod_N2/PC_weights0.png')
#plt.close()

# In[9]:

#	## Figure scatter plot preferred elevation and azimuth
#fig = plt.figure(figsize=(12,12))
#ax=plt.subplot(1,1,1)
#p=plt.plot(az.reshape(az.size),np.absolute(el.reshape(el.size)),'o',color = '#343837')
#ax.axhline(45, xmin=0, xmax=180, color='k', ls='--',linewidth = 6, label='Tresh')
#ax.text(90, 40, 'Color specific', ha = 'center',va = 'center', weight = 'bold',fontsize = 44)
#ax.text(90, 50, 'Color agnostic', ha = 'center',va = 'center', weight = 'bold',fontsize = 44)
#plt.setp(ax.get_xticklabels(), fontsize=30)
#plt.setp(ax.get_yticklabels(), fontsize=30)
#plt.xlabel('Preferred azimuth (mod 180$^o$)',fontsize = 35)
#plt.ylabel('Preferred |elevation| (degree)',fontsize = 35)
#plt.xticks((np.arange(0,360,30)))
#plt.yticks((np.arange(0,100,30)))
#fig.tight_layout()
#plt.show()
##fig.savefig('az&el_model0.png')
#plt.close()

fig = plt.figure(figsize = (6,5))
ax = fig.add_subplot(111)
h = ax.hist(EXPLAINED1.reshape(-1),color ='#343837',bins = 25)
ax.axvline(np.mean(EXPLAINED1), ymin=0, ymax=np.amax(h[0]), color='k', ls='-',linewidth = 5, label='Mean')
ax.axvline(np.percentile(EXPLAINED1,50), ymin=0, ymax=np.amax(h[0]), color='k', ls='--',linewidth = 5, label='Median')
ax.axvline(np.percentile(EXPLAINED1,10), ymin=0, ymax=np.amax(h[0]), color='k', ls=':',linewidth = 5, label='10$^{th}$ percentile')
plt.setp(ax.get_xticklabels(), fontsize=20)
plt.setp(ax.get_yticklabels(), fontsize=20)
plt.legend(loc='best',fontsize = 15)
plt.xlabel('% of variance explained',fontsize = 25)
plt.ylabel('Number of kernels',fontsize = 25)
fig.tight_layout()
plt.xlim(0,100)
fig.savefig(figures_dir_path +'filters_var_exp_%s.png' %(args.training_set))
plt.show()

import matplotlib.patheffects as PathEffects
# definitions for the axes
left, width = 0.13, 0.72
bottom, height = 0.11, 0.72
left_h = left + width + 0.005
bottom_h = bottom + height + 0.005

rect_scatter1 = [left, bottom, width, height]
rect_histy1 = [left_h, bottom, 0.165, height]
rect_histy2 = [left , bottom_h, width, 0.165]

dis.DEFINE_PLT_RC(type = 1)

fig = plt.figure(figsize = (7,7))

axScatter1 = plt.axes(rect_scatter1)
axHisty1 = plt.axes(rect_histy1)
axHisty2 = plt.axes(rect_histy2)

az[az<-45] = az[az<-45] + 360
# the scatter plot:
axScatter1.plot(az,np.absolute(el.reshape(el.size)),'o',ms=10,markeredgewidth=0.1,color = '#343837')
#axScatter1.axhline(45, xmin=-45, xmax=135, color='k', ls='--',linewidth = 6, label='Tresh')
#txt1 = axScatter1.text(135, 40, 'Color kernels', ha = 'center',color = 'k',va = 'center', weight = 'bold',fontsize = 20)
#txt2 = axScatter1.text(135, 50, 'Luminance kernels', ha = 'center',color = 'k',va = 'center', weight = 'bold',fontsize = 20)

#txt1.set_path_effects([PathEffects.withStroke(linewidth=2.5, foreground='w')])
#txt2.set_path_effects([PathEffects.withStroke(linewidth=2.5, foreground='w')])

h = axHisty1.hist(np.absolute(el.reshape(el.size)), bins=30, orientation='horizontal',color = '#343837') #343837
#axHisty1.axhline(45, xmin=0, xmax=np.amax(h[0]), color='k', ls='--',linewidth = 6, label='Tresh')

axScatter1.set_xticks((np.arange(-45,325,45)))
axScatter1.set_yticks((np.arange(0,100,45)))
axHisty1.set_xticks([])
axHisty1.set_yticks([])

#plt.setp(axScatter1.set_xlabel('Preferred azimuth (mod 180$^o$)',fontsize = 35))
plt.setp(axScatter1.set_ylabel('Tuning |elevation| (deg)'))
plt.setp(axScatter1.get_xticklabels())
plt.setp(axScatter1.get_yticklabels())

axScatter1.set_xlim((-45, 315))

#h2 = axHisty2.hist(az[np.absolute(el.reshape(el.size))<45], bins=30,color = '#343837')
h2 = axHisty2.hist(az, weights = np.cos(el.reshape(el.size)*np.pi/180), bins=30,color = '#343837')

axHisty2.set_xlim(axScatter1.get_xlim())
axHisty2.set_xticks([])
axHisty2.set_yticks([])

fig.text(0.5, 0.02, 'Tuning azimuth (deg)', ha='center')

fig.savefig(figures_dir_path +'filters_tuning_%s.png'%(args.training_set))
plt.show()

