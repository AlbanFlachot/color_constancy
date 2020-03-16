
# coding: utf-8

# In[35]:


from __future__ import print_function, division
import os
import torch

from skimage import io, transform
import numpy as np
from torch.utils import data
from torchvision import transforms, utils
import pickle
import MODELS as M
import CLASSES

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

os.environ["CUDA_VISIBLE_DEVICES"]="1"
# 
# First load the dataset lists

# In[36]:


with open("labels_train.txt", "rb") as fp:   # Unpickling
    train_lab = pickle.load(fp)

with open("ima_train.txt", "rb") as fp:   # Unpickling
    train_im = pickle.load(fp)
    
with open("labels_validation.txt", "rb") as fp:   # Unpickling
    val_lab = pickle.load(fp)

with open("ima_validation.txt", "rb") as fp:   # Unpickling
    val_im = pickle.load(fp)

MUNSELL_CIELAB = np.load('MUNSELL_LAB.npy')/100
#MUNSELL_COOR = np.load('MUNSELL_coor.npy')/100
#MUNSELL_COOR[:,1:] = MUNSELL_COOR[:,1:] + 1


#LAB_WCS = np.load('WCS_LAB.npy').T
#LAB_WCS = LAB_WCS/100
#LAB_WCS[:,1:] = LAB_WCS[:,1:] + 1


train_cielab = MUNSELL_CIELAB[train_lab]
val_cielab = MUNSELL_CIELAB[val_lab]
#train_coor = MUNSELL_COOR[train_lab]
#val_coor = MUNSELL_COOR[val_lab]



# In[38]:


import torch.nn as nn
import torch.nn.functional as F

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
#cudnn.benchmark = True



# In[39]:



# Parameters
params = {'batch_size': 800,
          'shuffle': True,
          'num_workers': 10}


# Datasets

partition = {'train': train_im, 'validation': val_im}

labels = {train_im[i]: train_cielab[i] for i in range(len(train_im))}
for i in range(len(val_im)):
    labels[val_im[i]] = val_cielab[i]

global OUTPUTS
global LABELS

def TRAINING_FUNC(net, partition, labels, params, epoch, chkpt_dir):

	if not os.path.exists(chkpt_dir):
		os.mkdir(chkpt_dir)

	# Generators
	training_set = CLASSES.Dataset(partition['train'], labels)
	training_generator = data.DataLoader(training_set, **params)

	validation_set = CLASSES.Dataset(partition['validation'], labels)
	validation_generator = data.DataLoader(validation_set, **params)


	# Optimizer
	import torch.optim as optim


	criterion = nn.PairwiseDistance(p=2.0, eps=1e-06, keepdim=False)
	#optimizer = optimizer.to(device)
	#optimizer = optim.Adam(net.parameters(), lr=0.0005)
	optimizer = optim.SGD(net.parameters(), lr=0.01, momentum = 0.9)

	from torch.optim.lr_scheduler import StepLR
	scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

	max_epochs = epoch

	train_curve = np.zeros(max_epochs)

	step = 0
	# Loop over epochs
	for epoch in range(max_epochs):
		running_loss = 0.0
		scheduler.step()
		# Training
		for local_batch, local_labels in training_generator:

			# Transfer to GPU
			local_batch, local_labels = local_batch.to(device), local_labels.to(device)

			
			# Model computations
			# zero the parameter gradients
			optimizer.zero_grad()

			# forward + backward + optimize
			outputs = net(local_batch)
			#print(outputs['out'].shape)
			#print(local_labels.shape)
			#loss = criterion()
			loss = criterion(outputs['out'].float(), local_labels.float()).mean()
			global OUTPUTS
			OUTPUTS = outputs['out'].float()
			global LABELS
			LABELS = local_labels.float()
			loss = loss.to(device)
			loss.backward()
			optimizer.step()
			#print(loss.item())
			outputs_np = outputs['out'].cpu().detach().numpy()
			labels_np = local_labels.cpu().detach().numpy()
			MSE_np = np.sqrt((outputs_np[:,0] - labels_np[:,0])**2 + (outputs_np[:,1] - labels_np[:,1])**2 + (outputs_np[:,2] - labels_np[:,2])**2)

			# print statistics
			running_loss += loss.item()
			step += 1
			if step % 20 == 0:    # print every 2000 mini-batches
				print('[%d, %5d] loss: %.3f batch_mean_dE: %.3f batch_median_dE: %.3f' %
					  (epoch + 1, step + 1, running_loss / 20 ,np.mean(MSE_np)*100, np.median(MSE_np)*100))
				running_loss = 0.0
	
		print('Epoch %d complete' %(epoch +1))
		torch.save(net.state_dict(), './' + chkpt_dir+'/model_3conv_epoch_%s.pt' %str(epoch+1))
	
		# Validation

		count = 0
		average = 0
		with torch.set_grad_enabled(False):
			for local_batch, local_labels in validation_generator:
			    # Transfer to GPU
			    local_batch, local_labels = local_batch.to(device), local_labels.to(device)

			    # Model computations
			    
			    outputs = net(local_batch)
			    outputs_np = outputs['out'].cpu().detach().numpy()
			    labels_np = local_labels.cpu().detach().numpy()
			    MSE_np = np.sqrt((outputs_np[:,0] - labels_np[:,0])**2 + (outputs_np[:,1] - labels_np[:,1])**2 + (outputs_np[:,2] - labels_np[:,2])**2) 
			    #print(MSE_np.shape)
			    average += np.mean(MSE_np)
			    #print(average)
			    count += 1
			train_curve[epoch] = average/float(count)
			print(average/float(count))
			#print('Mean deltaE of the network is: %f' %average/float(count))
		np.save(chkpt_dir+'train_curve', train_curve)
	

for inst_nb in range(5,10):
	net = M.Net_tristimulus_norm()
	net.to(device)
	TRAINING_FUNC(net, partition, labels, params, 90, 'inst%i_tristimulus'%inst_nb)
	
'''
for idx in np.arange(0,80000,200).astype(int):
	x = torch.load(partition['train'][idx])
	im = x.cpu().detach().numpy()
	im = np.moveaxis(im,0,-1)
	im = im/2 + 0.5
	print(partition['train'][idx])
	print(labels[partition['train'][idx]])'''



