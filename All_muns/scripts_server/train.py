
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

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

os.environ["CUDA_VISIBLE_DEVICES"]="3"
# 
# First load the dataset lists

# In[36]:


with open("labels_train_D65.txt", "rb") as fp:   # Unpickling
    train_lab = pickle.load(fp)

with open("ima_train_D65.txt", "rb") as fp:   # Unpickling
    train_im = pickle.load(fp)
    
with open("labels_validation_D65.txt", "rb") as fp:   # Unpickling
    val_lab = pickle.load(fp)

with open("ima_validation_D65.txt", "rb") as fp:   # Unpickling
    val_im = pickle.load(fp)


# In[37]:


def _get_padding(size, kernel_size, stride, dilation):
    padding = (kernel_size - stride)//2
    return padding

print(_get_padding(128, 9, 4, 1))


# In[38]:


import torch.nn as nn
import torch.nn.functional as F

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
#cudnn.benchmark = True



# In[39]:


class Dataset(data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, list_IDs, labels):
        'Initialization'
        self.labels = labels
        self.list_IDs = list_IDs

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]
        #Load data and get label
        if ID[-3:] == 'png':
            I = np.array(io.imread(ID)).astype(float)
            if np.amax(I) > 2:
            	I = (I/255)
            	#print((np.amax(I),np.amin(I)))
            else:
            	I = (I)
            #I = transform.resize(I,(224,224))
            I = np.moveaxis(I,-1,0)
            X = torch.from_numpy(I)
        elif ID[-3:] == 'npy':
        	I = np.load(ID).astype(float)
        	I = np.moveaxis(I,-1,0)
        	I = I*(0.75 + 0.5*np.random.random(1))
        	I = I - 3
        	X = torch.from_numpy(I)
        else:
            X = torch.load(ID)
        X = X.type(torch.FloatTensor)
        #X = torch.load(ID)
        #X = transforms.CenterCrop(224)
        y = self.labels[ID]
        return X, y


# Parameters
params = {'batch_size': 800,
          'shuffle': True,
          'num_workers': 10}


# Datasets
partition = {'train': train_im, 'validation': val_im}

labels = {train_im[i]: train_lab[i] for i in range(len(train_im))}
for i in range(len(val_im)):
    labels[val_im[i]] = val_lab[i]


def TRAINING_FUNC(net, partition, labels, params, epoch, chkpt_dir):
	
	if not os.path.exists(chkpt_dir):
		os.mkdir(chkpt_dir)
	
	# Generators
	training_set = Dataset(partition['train'], labels)
	training_generator = data.DataLoader(training_set, **params)

	validation_set = Dataset(partition['validation'], labels)
	validation_generator = data.DataLoader(validation_set, **params)


	# Optimizer
	import torch.optim as optim

	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(net.parameters(), lr=0.001)
	#optimizer = optimizer.to(device)

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
			#optimizer = optim.Adam(net.parameters(), lr=0.0005)
			# Model computations
			# zero the parameter gradients
			optimizer.zero_grad()

			# forward + backward + optimize
			outputs = net(local_batch)
			loss = criterion(outputs['out'], local_labels)
			loss = loss.to(device)

			loss.backward()
			optimizer.step()
			_, predicted_train = torch.max(outputs['out'].data, 1)
			correct_train = (predicted_train == local_labels).sum().item()
			Acc_train = 100 * correct_train / local_batch.shape[0]
			# print statistics
			running_loss += loss.item()
			step += 1
			if step % 20 == 0:    # print every 2000 mini-batches
				print('[%d, %5d] loss: %.3f Accuracy batch: %.3f' %
					  (epoch + 1, step + 1, running_loss / 20, Acc_train))
				running_loss = 0.0
		
		print('Epoch %d complete' %(epoch +1))
		torch.save(net.state_dict(), './' + chkpt_dir+'/epoch_%s.pt' %str(epoch+1))

		# Validation
		correct = 0
		total = 0
		with torch.set_grad_enabled(False):
			for local_batch, local_labels in validation_generator:
				# Transfer to GPU
				local_batch, local_labels = local_batch.to(device), local_labels.to(device)

				# Model computations
				
				outputs = net(local_batch)
				_, predicted = torch.max(outputs['out'].data, 1)
				total += local_labels.size(0)
				correct += (predicted == local_labels).sum().item()
			train_curve[epoch] = 100 * correct / total
			print('Accuracy of the network on the validation dataset: %d %%' % (100 * correct / total))
		np.save(chkpt_dir+'train_curve', train_curve)

for inst_nb in range(0,10):
	net = M.Net2_4ter_norm()
	net.to(device)
	TRAINING_FUNC(net, partition, labels, params, 90, 'INST_D65/inst%i'%inst_nb)
	



