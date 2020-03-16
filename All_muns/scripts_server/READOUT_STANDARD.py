#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 15:40:21 2019

@author: alban
"""


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
from skimage import color
# Ignore warnings
import warningsscheduler.step()
warnings.filterwarnings("ignore")
import argparse

os.environ["CUDA_VISIBLE_DEVICES"]="0"


# In[36]:


with open("../train_labels.txt", "rb") as fp:   # Unpickling
    train_lab = pickle.load(fp)

with open("../train_ima.txt", "rb") as fp:   # Unpickling
    train_im = pickle.load(fp)

with open("../val_labels.txt", "rb") as fp:   # Unpickling
    val_lab = pickle.load(fp)

with open("../val_ima.txt", "rb") as fp:   # Unpickling
    val_im = pickle.load(fp)



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


parser = argparse.ArgumentParser(description='Training parameters')
parser.add_argument()
# In[39]:

# Parameters
params = {'batch_size': 256,
          'shuffle': True,
          'num_workers': 20}


# Datasets

partition = {'train': train_im, 'validation': val_im}

labels = {train_im[i]: train_lab[i] for i in range(len(train_im))}
for i in range(len(val_im)):
    labels[val_im[i]] = val_lab[i]

#class _Loss(Module):
#    def __init__(self, size_average=None, reduce=None, reduction='mean'):
#        super(_Loss, self).__init__()
#        if size_average is not None or reduce is not None:
#            self.reduction = _Reduction.legacy_get_string(size_average, reduce)
#        else:
#            self.reduction = reduction

# In[7]: PLOT TRAINING CURVES
nb_mod = 10
nb_epoch = 40
nb_obj = 330

## Xp set
####---------------------------------------------------------------------------------------------------------------------
Training_curv = np.zeros((nb_mod,nb_epoch))
Training_curv_D65 = np.zeros((nb_mod,nb_epoch))
epochmax = np.zeros((nb_mod))
epochmax_D65 = np.zeros((nb_mod))

for i in range(nb_mod):
    Training_curv[i] = np.load('../mod_N2/inst%i_rand_lum_norm_steptrain_curve.npy' %((i+1)))
    epochmax[i] = np.argmax(Training_curv[i])
    print('inst %i achieves max accuracy of %d at epoch %i' %(i+1, np.amax(Training_curv[i]),np.argmax(Training_curv[i])))

class Readout_net(nn.Module):
    def __init__(self):
        super(Readout_net, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16*63*63, 330)
	
    def forward(self, x):
        POOL = self.pool(x)
        POOL_flat = POOL.view(-1, 16*63*63)
        FC1 = self.fc1(POOL_flat)
        FEATURES = {'out': FC1}
        return FEATURES


# In[7]: FINETUNING FUNCITON

#initialization

i = 0


# In[7]: FINETUNING FUNCITON
def FINETUNE_FUNC(net, R1, partition, labels, params, epoch, chkpt_dir):

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
    optimizer = optim.Adam(R1.parameters(), lr=0.0005)
    from torch.optim.lr_scheduler import StepLR
	scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    #
    
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
            #LR = 0.0005
            outputs = net(local_batch)
            
            Read = R1(outputs['conv1'])
            
            #optimizer = optimizer.to(device)
            # Model computations
            # zero the parameter gradients
            optimizer.zero_grad()
            
            loss = criterion(Read['out'], local_labels)
            loss = loss.to(device)
            loss.backward()
            optimizer.step()
            
            #print(loss.item())
            
            # print statistics
            running_loss += loss.item()
            step += 1
            
            if step % 20 == 0:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, step + 1, running_loss / 20))
                
                running_loss = 0.0
        
        print('Epoch %d complete' %(epoch +1))
        torch.save(R1.state_dict(), './' + chkpt_dir+'/epoch_%s.pt' %str(epoch+1))
        
        # Validation
        correct = 0
        total = 0
        with torch.set_grad_enabled(False):
            for local_batch, local_labels in validation_generator:
                # Transfer to GPU
                local_batch, local_labels = local_batch.to(device), local_labels.to(device)
                
                # Model computations
                
                outputs = net(local_batch)
                Read = R1(outputs['conv1'])
                #outR1 = R1(outputs['fc1'])
                _, predicted = torch.max(Read['out'].data, 1)
                total += local_labels.size(0)
                correct += (predicted == local_labels).sum().item()
                train_curve[epoch] = 100 * correct / total
        print('Accuracy of the network on the validation dataset: %d %%' % (100 * correct / total))
        np.save(chkpt_dir+'/train_curve', train_curve)


for inst_nb in range(10):
    weights = '../mod_N2/inst_%d/model_3conv_epoch_%i.pt' %(inst_nb+1,epochmax[inst_nb])
    net = M.Net2_4ter()
    net.load_state_dict(torch.load(weights))
    for param in net.parameters():
        param.requires_grad = False
    net.to(device)
    R1 = Readout_net()
    R1.to(device)
    if not os.path.exists('WCS_muns/inst_%i'%inst_nb):
        os.mkdir('WCS_muns/inst_%i'%inst_nb)
    FINETUNE_FUNC(net, R1, partition, labels, params, 30, 'WCS_muns/inst_%i/readout_conv1_rand_lum_norm' %inst_nb)
