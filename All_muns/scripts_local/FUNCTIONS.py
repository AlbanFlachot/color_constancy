# In[1]:



from __future__ import print_function, division
import os
import torch
import matplotlib.pyplot as plt
from skimage import io, transform
import numpy as np
from torch.utils import data
from torchvision import transforms, utils
import pickle
import matplotlib.patheffects as PathEffects
import scipy.io as sio
from random import randint

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


import torch.nn as nn
import torch.nn.functional as F


# In[2]:


# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
#cudnn.benchmark = True


# In[3]: FUNCTIONS


def princomp(A):
 """ performs principal components analysis 
     (PCA) on the n-by-p data matrix A
     Rows of A correspond to observations, columns to variables. 

 Returns :  
  coeff :
    is a p-by-p matrix, each column containing coefficients 
    for one principal component.
  score : 
    the principal component scores; that is, the representation 
    of A in the principal component space. Rows of SCORE 
    correspond to observations, columns to components.
  latent : 
    a vector containing the eigenvalues 
    of the covariance matrix of A.
 """
 # computing eigenvalues and eigenvectors of covariance matrix
 M = (A-np.mean(A.T,axis=1)).T # subtract the mean (along columns)
 [latent,coeff] = np.linalg.eig(np.cov(M)) # attention:not always sorted
 sortedIdx = np.argsort(-latent)
 latent = latent[sortedIdx]
 explained = 100*latent/np.sum(latent)
 score = np.dot(coeff.T,M) # projection of the data in the new space
 coeff = coeff[:,sortedIdx]
 score = score[sortedIdx,:]
 return coeff,score,latent, explained

def LMS2Opp(x):
	""" Converts from LMS tristimulus to opponent space.

	Returns :  
	Matrix of in opponent coordinates.
	"""
	M = np.array([[ 0.67,  1.0,  0.5],[ 0.67,  -1,  0.5],[ 0.67,  0,  -1]])
	return np.dot(x,M)

def cart2sph(x,y,z):
    """ Converts for cartesian coordinates to spherical coordinates.
    
    Returns :  
    Matrix in spherical coordinates radius, azimuth and elevation.
    """
    XsqPlusYsq = x**2 + y**2
    r = np.sqrt(XsqPlusYsq + z**2)               # r
    elev = np.arctan(z/np.sqrt(XsqPlusYsq))     # theta
    az = np.arctan2(y,x)                           # phi
    return np.array([r, az, elev])


def vis_square(data, name_fig):
    '''Take an array of shape (n, height, width) or (n, height, width, 3)
       #and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)'''
    
    # normalize data for display
    data = (data - data.min()) / (data.max() - data.min())
    
    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = (((0, n ** 2 - data.shape[0]),
	       (0, 1), (0, 1))                 # add some space between filters
	       + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)
    data = np.pad(data, padding, mode='constant', constant_values=1)  # pad with ones (white)
    
    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    
    plt.imshow(data)
    plt.show()
    plt.imsave(name_fig,data)
    return data

def testing_image_feed(ID):
    '''Function to test image loading'''
    
    if ID[-3:] == 'png':
        I = np.array(io.imread(ID)).astype(float)
        if np.amax(I) > 2:
            I = (I/255 - 0.5)*2
        else:
            I = (I - 0.5)*2
        I = np.moveaxis(I,-1,0)
        X = torch.from_numpy(I)
    elif ID[-3:] == 'npy':
        I = np.load(ID)
        I = np.moveaxis(I,-1,0)
        I = I - 3
        x = torch.from_numpy(I)
    else:
        X = torch.load(ID)
    X = X.type(torch.FloatTensor)
    X  = X.unsqueeze(0)
    
    X = X.to(device)
    im = X.cpu().detach().numpy()
    im = np.moveaxis(im,0,-1)
    im = im/2 + 0.5
    np.amax(im)
    plt.imshow(im)
    plt.show()



def evaluation(predictions,label):
    s = np.sum(predictions == label)
    return 100*s/len(predictions)

## TEST MODEL
def TESTING_FUNC(datatype,model,weights,list_WCS_labels = None):
        
    with open("val_labels" +datatype, "rb") as fp:   # Unpickling
        val_lab = pickle.load(fp)
    
    with open("val_ima" +datatype, "rb") as fp:   # Unpickling
        val_im = pickle.load(fp)
    
    LIST = [[None]] * (max(val_lab)+1)
    for count in range(len(val_lab)):
        LIST[val_lab[count]] = LIST[val_lab[count]]+[val_im[count]]
    
    
    net = model
    
    net.load_state_dict(torch.load(weights))
    net.to(device)
    net.eval()
    
    PREDICTION = np.ones((len(LIST),50))
    for l in range(len(LIST)):
        PREDICTION[l] = testing(net, LIST[l][1:51],'npy')
    
    EVAL = np.zeros(len(LIST))
    for i in range(len(EVAL)):
        if list_WCS_labels == None:
            EVAL[i] = evaluation(PREDICTION[i],i)
        else:
            EVAL[i] = evaluation(PREDICTION[i],list_WCS_labels[i])
    return PREDICTION,EVAL


def TESTING_FUNC_all(datatype,model,weights):
    with open("train_labels" +datatype, "rb") as fp:   # Unpickling
        train_lab = pickle.load(fp)
    
    with open("train_ima" +datatype, "rb") as fp:   # Unpickling
        train_im = pickle.load(fp)
        
    with open("val_labels" +datatype, "rb") as fp:   # Unpickling
        val_lab = pickle.load(fp)
    
    with open("val_ima" +datatype, "rb") as fp:   # Unpickling
        val_im = pickle.load(fp)    
    
    net = model
    
    net.load_state_dict(torch.load(weights))
    net.to(device)
    net.eval()
    
    PREDICTION = np.ones((len(LIST),50))
    for l in range(len(LIST)):
        PREDICTION[l] = testing(net, LIST[l][1:51],'npy')
    
    EVAL = np.zeros(len(LIST))
    for i in range(len(EVAL)):
        EVAL[i] = evaluation(PREDICTION[i],i)
    return PREDICTION,EVAL


def testing(net,list_obj1,type):
    count = 0
    #print(type)
    predictions = np.zeros(len(list_obj1))
    for i in list_obj1:
        if type == 'png':
            I = np.array(io.imread(i)).astype(float)
            if np.amax(I) > 2:
                I = (I/255)
            else:
                I = I
            #I = transform.resize(I,(224,128))
            I = np.moveaxis(I,-1,0)
            x = torch.from_numpy(I)
            x = x.type(torch.FloatTensor)
        elif type == 'npy':
            I = np.load(i).astype(float)
            I = np.moveaxis(I,-1,0)
            I = I - 3
            x = torch.from_numpy(I)
        else:
            x = torch.load(i)
        x  = x.unsqueeze(0)
        x = x.to(device)
        if x.size()[2] < 220:
            continue
        outputs = net(x)
        _, predictions[count] = torch.max(outputs['out'].data, 1)
        count +=1
    return predictions

def evaluation(predictions,label):
    s = np.sum(predictions == label)
    return 100*s/len(predictions)

def retrieve_activations(net,img,type):
    if type == 'png':
        I = np.array(io.imread(img)).astype(float)
        if np.amax(I) > 2:
            I = (I/255)
        else:
            I = I
        #I = transform.resize(I,(224,128))
        I = np.moveaxis(I,-1,0)
        x = torch.from_numpy(I)
        x = x.type(torch.FloatTensor)
    elif type == 'npy':
        I = np.load(img).astype(float)
        I = np.moveaxis(I,-1,0)
        I = I - 3
        x = torch.from_numpy(I)
        x = x.type(torch.FloatTensor)
    else:
        x = torch.load(img)
    x  = x.unsqueeze(0)
    x = x.to(device)
    outputs = net(x)
    conv1 = np.amax(outputs['conv1'].cpu().detach().numpy(),axis = (2,3))
    conv2 = np.amax(outputs['conv2'].cpu().detach().numpy(),axis = (2,3))
    conv3 = np.amax(outputs['conv3'].cpu().detach().numpy(),axis = (2,3))
    fc1 = outputs['fc1'].cpu().detach().numpy()
    fc2 = outputs['fc2'].cpu().detach().numpy()
    _, p = torch.max(outputs['out'].data, 1)
    return conv1[0],conv2[0],conv3[0],fc1[0],fc2[0], (p.cpu().detach().numpy())[0]


def compute_munsells(EVAL,WCS_X,WCS_Y):
    WCS_MAT = np.zeros((len(EVAL),10,41))
    for j in range(len(EVAL)):
        for i in range(len(EVAL[0])):
            WCS_MAT[j,WCS_X[i],WCS_Y[i]] = EVAL[j][i]
    return WCS_MAT

def display_munsells(WCS_MAT, norm):
    WCS_MAT = (WCS_MAT/norm)
    if WCS_MAT.shape == (10,41):
        WCS_MAT_achro = WCS_MAT[:,0]
        WCS_MAT_chro = WCS_MAT[1:-1,1:]
        # definitions for the axes
        left, width = 0.05, 0.8
        bottom, height = 0.1, 0.8
        left_h = left + width + 0.05
        bottom_h = 0
        
        rect_chro = [left, bottom, width, height]
        rect_achro = [left_h, bottom_h, 0.8/40, 1]
        
        fig = plt.figure(1, figsize=(8, 3))
        
        axchro = plt.axes(rect_chro)
        axachro = plt.axes(rect_achro)
        
        # the scatter plot:
        axchro.imshow(np.stack((WCS_MAT_chro,WCS_MAT_chro,WCS_MAT_chro),axis = 2))
        axachro.imshow(np.stack((WCS_MAT_achro,WCS_MAT_achro,WCS_MAT_achro),axis = 2))
        #axchro.set_xticks((np.arange(0,40)))
        #axchro.set_yticks((np.arange(1,9)))
        axachro.set_xticks([])
        #axachro.set_yticks((np.arange(0,10)))
        
        axchro.set_ylabel('Value',fontsize = 15)
        axchro.set_xlabel('Hue',fontsize = 15)
        #plt.setp(axchro.set_ylabel('Tuning |elevation| (deg)',fontsize = 25))
        plt.setp(axchro.get_xticklabels(), fontsize=12)
        plt.setp(axchro.get_yticklabels(), fontsize=12)
        
        #axchro.set_xlim((-45, 315))
        
        #fig.text(0.5, 0.02, 'Tuning azimuth (deg)', ha='center',fontsize = 25)
        
        #fig.savefig('mod_N2/control2_lay1_tuning.png')
        plt.show()

    else:
        fig = plt.figure(figsize = (8,3))
        plt.imshow(np.stack((WCS_MAT,WCS_MAT,WCS_MAT),axis = 2))
        plt.xlabel('Munsell hue',fontsize = 15)
        plt.ylabel('Munsell value',fontsize = 15)
        #plt.title('Munsell chips WCS',fontsize = 18)
        #fig.savefig('Munsell_chips_WCS.png')
        fig.tight_layout
        plt.show()

def display_munsells_inv(WCS_MAT, norm):
    WCS_MAT2 = (WCS_MAT/norm)
    WCS_MAT = np.zeros(WCS_MAT2.shape)

    if WCS_MAT.shape == (10,41):
        for i in range(len(WCS_MAT2)):
            WCS_MAT[i] = WCS_MAT2[9-i]
        WCS_MAT_achro = WCS_MAT[:,0].reshape(10,1)
        WCS_MAT_chro = WCS_MAT[1:-1,1:]
        # definitions for the axes
        left, width = 0.05, 0.8
        bottom, height = 0.1, 0.8
        left_h = left + width + 0.05
        bottom_h = 0
        
        rect_chro = [left, bottom, width, height]
        rect_achro = [left_h, bottom_h, 0.8/40, 1]
        
        fig = plt.figure(1, figsize=(8, 3))
        
        axchro = plt.axes(rect_chro)
        axachro = plt.axes(rect_achro)
        
        # the scatter plot:
        axchro.imshow(np.stack((WCS_MAT_chro,WCS_MAT_chro,WCS_MAT_chro),axis = 2))
        axachro.imshow(np.stack((WCS_MAT_achro,WCS_MAT_achro,WCS_MAT_achro),axis = 2))
        #axchro.set_xticks((np.arange(0,40)))
        #axchro.set_yticks((np.arange(1,9)))
        axachro.set_xticks([])
        #axachro.set_yticks((np.arange(0,10)))
        
        axchro.set_ylabel('Value',fontsize = 15)
        axchro.set_xlabel('Hue',fontsize = 15)
        #plt.setp(axchro.set_ylabel('Tuning |elevation| (deg)',fontsize = 25))
        plt.setp(axchro.get_xticklabels(), fontsize=12)
        plt.setp(axchro.get_yticklabels(), fontsize=12)
        
        #axchro.set_xlim((-45, 315))
        
        #fig.text(0.5, 0.02, 'Tuning azimuth (deg)', ha='center',fontsize = 25)
        
        #fig.savefig('mod_N2/control2_lay1_tuning.png')
        plt.show()


    else:
        for i in range(len(WCS_MAT2)):
            WCS_MAT[i] = WCS_MAT2[7-i]
        fig = plt.figure(figsize = (8,3))
        plt.imshow(np.stack((WCS_MAT,WCS_MAT,WCS_MAT),axis = 2))
        plt.xlabel('Munsell hue',fontsize = 15)
        plt.ylabel('Munsell value',fontsize = 15)
        #plt.title('Munsell chips WCS',fontsize = 18)
        #fig.savefig('Munsell_chips_WCS.png')
        fig.tight_layout
        plt.show()

# Accuracy as WCS munsell distance

def PREDICTION_WCS(PREDICTION,WCS_X,WCS_Y):
    PREDICTION_WCS = np.zeros((PREDICTION.shape[0],10,41,PREDICTION.shape[-1]))
    for i in range(PREDICTION.shape[1]):
        PREDICTION_WCS[:,WCS_X[i],WCS_Y[i]] = PREDICTION[:,i]
    
    WCS_X_arr = np.asarray(WCS_X)
    WCS_Y_arr = np.asarray(WCS_Y)
    
    PREDICTION_WCS_color = PREDICTION_WCS[:,1:-1,1:].copy()
    PREDICTION_WCS_achro = PREDICTION_WCS[:,:,0].copy()
    for m in range(PREDICTION.shape[0]):
        for v in range(8):
            for h in range(40):
                dist = np.sqrt((v+1 - WCS_X_arr[PREDICTION_WCS_color[m,v,h].astype(int).tolist()])**2
                + (h+1 - WCS_Y_arr[PREDICTION_WCS_color[m,v,h].astype(int).tolist()])**2)
                dist[dist>20] = 40 - dist[dist>20]
                PREDICTION_WCS_color[m,v,h] = dist 
     
    for m in range(PREDICTION.shape[0]):
        for v in range(10):           
            dist = np.sqrt((v - WCS_X_arr[PREDICTION_WCS_achro[m,v].astype(int).tolist()])**2)
            PREDICTION_WCS_achro[m,v] = dist
            
    PREDICTION_ERROR_WCS = np.zeros(PREDICTION_WCS.shape)
    PREDICTION_ERROR_WCS[:,1:-1,1:] = PREDICTION_WCS_color
    PREDICTION_ERROR_WCS[:,:,0] = PREDICTION_WCS_achro
    return PREDICTION_ERROR_WCS


# Accuracy as LAB distance

def PREDICTION_LAB(PREDICTION,WCS_X,WCS_Y,LAB_WCS):
    PREDICTION_WCS = np.zeros((PREDICTION.shape[0],10,41,PREDICTION.shape[-1]))
    for i in range(PREDICTION.shape[1]):
        PREDICTION_WCS[:,WCS_X[i],WCS_Y[i]] = PREDICTION[:,i]
    
    WCS_X_arr = np.asarray(WCS_X)
    WCS_Y_arr = np.asarray(WCS_Y)
    
    PREDICTION_WCS_color = PREDICTION_WCS[:,1:-1,1:].copy()
    PREDICTION_WCS_achro = PREDICTION_WCS[:,:,0].copy()
    for m in range(PREDICTION.shape[0]):
        for v in range(8):
            for h in range(40):
                dist = np.sqrt((v+1 - WCS_X_arr[PREDICTION_WCS_color[m,v,h].astype(int).tolist()])**2
                + (h+1 - WCS_Y_arr[PREDICTION_WCS_color[m,v,h].astype(int).tolist()])**2)
                dist[dist>20] = 40 - dist[dist>20]
                PREDICTION_WCS_color[m,v,h] = dist 
     
    for m in range(PREDICTION.shape[0]):
        for v in range(10):           
            dist = np.sqrt((v - WCS_X_arr[PREDICTION_WCS_achro[m,v].astype(int).tolist()])**2)
            PREDICTION_WCS_achro[m,v] = dist
            
    PREDICTION_ERROR_WCS = np.zeros(PREDICTION_WCS.shape)
    PREDICTION_ERROR_WCS[:,1:-1,1:] = PREDICTION_WCS_color
    PREDICTION_ERROR_WCS[:,:,0] = PREDICTION_WCS_achro
    return PREDICTION_ERROR_WCS

# Accuracy as munsell distance

def PREDICTION_munsell(PREDICTION,WCS_X,WCS_Y):
    PREDICTION_WCS = np.zeros((PREDICTION.shape[0],10,41,PREDICTION.shape[-1]))
    for i in range(PREDICTION.shape[1]):
        PREDICTION_WCS[:,WCS_X[i],WCS_Y[i]] = PREDICTION[:,i]
    
    WCS_X_arr = np.asarray(WCS_X)
    WCS_Y_arr = np.asarray(WCS_Y)
    
    PREDICTION_WCS_color = PREDICTION_WCS[:,1:-1,1:].copy()
    PREDICTION_WCS_achro = PREDICTION_WCS[:,:,0].copy()
    for m in range(PREDICTION.shape[0]):
        for v in range(8):
            for h in range(40):
                dist = np.sqrt((v+1 - WCS_X_arr[PREDICTION_WCS_color[m,v,h].astype(int).tolist()])**2
                + (h+1 - WCS_Y_arr[PREDICTION_WCS_color[m,v,h].astype(int).tolist()])**2)
                dist[dist>20] = 40 - dist[dist>20]
                PREDICTION_WCS_color[m,v,h] = dist 
     
    for m in range(PREDICTION.shape[0]):
        for v in range(10):           
            dist = np.sqrt((v - WCS_X_arr[PREDICTION_WCS_achro[m,v].astype(int).tolist()])**2)
            PREDICTION_WCS_achro[m,v] = dist
            
    PREDICTION_ERROR_WCS = np.zeros(PREDICTION_WCS.shape)
    PREDICTION_ERROR_WCS[:,1:-1,1:] = PREDICTION_WCS_color
    PREDICTION_ERROR_WCS[:,:,0] = PREDICTION_WCS_achro
    return PREDICTION_ERROR_WCS




def TESTING_FUNC_no_patches(datatype,model,weights):
        
    with open("val_labels" +datatype, "rb") as fp:   # Unpickling
        val_lab = pickle.load(fp)
    
    with open("val_ima" +datatype, "rb") as fp:   # Unpickling
        val_im = pickle.load(fp)    
    
    LIST = [[None]] * (max(val_lab)+1)
    for count in range(len(val_lab)):
        LIST[val_lab[count]] = LIST[val_lab[count]]+[val_im[count]]
     
    net = model
    
    net.load_state_dict(torch.load(weights))
    net.to(device)
    net.eval()
    
    net = model
    
    net.load_state_dict(torch.load(weights))
    net.to(device)
    net.eval()
    
    PREDICTION = np.ones((len(LIST),50))
    for l in range(len(LIST)):
        PREDICTION[l] = testing_no_patches(net, LIST[l][1:51],'npy')
    
    EVAL = np.zeros(len(LIST))
    for i in range(len(EVAL)):
        EVAL[i] = evaluation(PREDICTION[i],i)
    return PREDICTION,EVAL

def testing_no_patches(net,list_obj1,type):
    count = 0
    #print(type)
    predictions = np.zeros(len(list_obj1))
    for i in list_obj1:
        if type == 'png':
            I = np.array(io.imread(i)).astype(float)
            if np.amax(I) > 2:
                I = (I/255)
            else:
                I = I
            #I = transform.resize(I,(224,128))
            I = np.moveaxis(I,-1,0)
            x = torch.from_numpy(I)
            x = x.type(torch.FloatTensor)
        elif type == 'npy':
            I = np.load(i).astype(float)
            trans_im = I.copy()
            local_mean = np.mean(trans_im[0:8,27:100],axis = (0))
            band = np.tile(local_mean[np.newaxis,:,:],(11,1,1))
            local_std= np.std(trans_im[0:8,10:115])
            #fill_in_shape = trans_im[8:19,27:100].shape
            lum_noise = np.random.normal(0,local_std/10,(11,73))
            #band = np.zeros(fill_in_shape)
            #trans_im[8:19,27:100] = np.stack((lum_noise,lum_noise,lum_noise),axis = -1) + np.stack(local_mean,local_mean,local_mean)
            trans_im[8:19,27:100] = band+ np.tile(lum_noise[:,:,np.newaxis],(1,1,3))
            I = trans_im.copy()
            I = np.moveaxis(I,-1,0)
            I = I - 3
            x = torch.from_numpy(I)
            x = x.type(torch.FloatTensor)
        else:
            x = torch.load(i)
        x  = x.unsqueeze(0)
        x = x.to(device)
        outputs = net(x)
        #print(outputs['conv1'].size())
        #print(outputs['conv2'].size())
        #print(outputs['conv3'].size())
        #print(outputs['pool3'].size())
        _, predictions[count] = torch.max(outputs['out'].data, 1)
        count +=1
    return predictions
