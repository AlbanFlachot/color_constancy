# -*- coding: utf-8 -*-
'''

Script to select the model with highest accuracy

'''

import numpy as np
import glob
import matplotlib.pyplot as plt
import sys
import torch
import os

#sys.path.append('/home/alban/mnt/awesome/alban/project_color_constancy/PYTORCH/WCS/Resnets/python/src/kernelphysiology/dl/pytorch')
#sys.path.append('/home/alban/mnt/awesome/alban/project_color_constancy/PYTORCH/WCS/Resnets/python/src/')

#from models import utils
os.environ["CUDA_VISIBLE_DEVICES"]="2"
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

nb_epochs = 90

def info_conv_parameters(state_dict):
    num_kernels = 0
    num_parameters = 0
    for layer in state_dict.keys():
        if '.weight' in layer:
            current_layer = state_dict[layer].cpu().numpy()
            #import pdb; pdb.set_trace()
            num_kernels += current_layer.shape[0]
            num_parameters += current_layer.size
    return num_kernels, num_parameters

list_training_dir = glob.glob('*train_curve.npy')

for i in list_training_dir:
	if 'tristimulus' in i:
		list_training_dir.remove(i)
	if 'D65' in i:
		list_training_dir.remove(i)

for i in list_training_dir:
	if 'tristimulus' in i:
		list_training_dir.remove(i)
	if 'D65' in i:
		list_training_dir.remove(i)

TRAIN_CURVES = np.zeros((len(list_training_dir), nb_epochs))
DIFF_train_val = np.zeros((len(list_training_dir), nb_epochs))
NB_PARAM = np.zeros((len(list_training_dir), 2))

import re
models = [re.search('(.*?)train_curve.npy', addr).group(1) for addr in list_training_dir]

count = 0
for add in list_training_dir:
	tran_curv = np.load(add)
	TRAIN_CURVES[count] = tran_curv
	weights = 'INST/%s/epoch_%i.pt' %(models[count], np.argmax(TRAIN_CURVES[count])+1)
	state_dict = torch.load(weights)
	num_kernels, num_param = info_conv_parameters(state_dict)
	NB_PARAM[count, 0] = num_kernels
	NB_PARAM[count, 1] = num_param
	count +=1


INDX = np.unravel_index(np.argmax(TRAIN_CURVES), (len(list_training_dir), nb_epochs))

print('The model with the highest performance is: %s \nat epoch %i \nwith a performance of %d percents' %(list_training_dir[INDX[0]], INDX[1], TRAIN_CURVES[INDX]))

sort_IDX = np.argsort(-np.amax(TRAIN_CURVES, axis = 1))

np.amax(TRAIN_CURVES, axis = 1)[sort_IDX[:10]]
#list_training_dir[list(sort_IDX[:10])]
for i in sort_IDX[:10]:
	print(list_training_dir[i])

np.save('nb_param_convnet.npy',NB_PARAM)
np.save('train_curves_convnet.npy', TRAIN_CURVES)


'''
fig = plt.figure()
p = plt.plot(TRAIN_CURVES.T, color= 'black', lw = 2)
plt.ylim(0,100)
plt.xlabel('epoch',fontsize = 15)
plt.ylabel('accuracy',fontsize = 15)
#plt.xticks(np.arange(0,101,25),fontsize = 14)
plt.yticks(fontsize = 14)
plt.xticks(fontsize = 14)

fig.tight_layout
plt.show()'''
