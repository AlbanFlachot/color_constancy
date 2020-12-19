import numpy as np
import torch
import cv2
from random import randint

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

def transform(img, scene, patch_nb = 0, testing = 'normal', type = 'npy', preprocessing='alban'):
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
		if (testing == 'wrong_illu') | (testing == 'no_back'):
			I_mask = np.load(img[:-4] + '_mask.npy')
			MASK = np.mean(I_mask, axis = -1)
			MASK[MASK > np.amax(MASK)/255] = 1
			MASK[MASK != 1] = 0
		else:
			MASK = 0
		if testing == 'no_patch':
			trans_im = I.copy()
			local_mean = np.mean(trans_im[0:8,27 + int(12*patch_nb) : 27+ int(12*(patch_nb+1))+1],axis = (0))
			band = np.tile(local_mean[np.newaxis,:,:],(11,1,1))
			local_std= np.std(trans_im[0:8,10:115])
			lum_noise = np.random.normal(0,local_std/10,(11,13))
			trans_im[8:19, 27 + int(12*patch_nb) : 27+ int(12*(patch_nb+1))+1] = band+ np.tile(lum_noise[:,:,np.newaxis],(1,1,3))
			I = trans_im.copy()
		elif testing == 'wrong_illu':
			SCENE = np.load(scene)                     
			SCENE[MASK==1] = I[MASK==1]         
			I = SCENE.astype(float)
		elif testing == 'no_back': 
			SCENE = np.zeros(I.shape)
			SCENE[MASK==1] = I[MASK==1]         
			I = SCENE.astype(float)
		if preprocessing == 'alban':
			I = np.moveaxis(I,-1,0)
			I = I - 3
		elif preprocessing == 'arash':
			mean = [0.485, 0.456, 0.406]
			std = [0.229, 0.224, 0.225]
			I = I/I.max()
			I = (I - mean)/std
			I = np.moveaxis(I,-1,0)
		x = torch.from_numpy(I)
		x = x.type(torch.FloatTensor)
	else:
		x = torch.load(img)
	x  = x.unsqueeze(0)
	x = x.to(device)
	return x, MASK


def receptive_field(layer, stride, window):
        '''
        Function that computes receptive field of a convolutional kernel
        '''
        rf = 1
        for count in range(layer):
                rf += np.power(2,count)*stride[count]*(window[count] - 1)
        return rf


def relevant_kernel_map(mask, layer, dim , stride , window , im_size):
        '''
        Function which finds which kernel of convolutional layer are detecting colored object within scene.
        '''
        
        rf = window[0]
        rf = receptive_field(layer, stride, window)	
        relevance = cv2.GaussianBlur(mask.astype('float32'), (rf, rf),0)[::np.power(2,layer-1),::np.power(2,layer-1)]
        ind = (relevance.shape[0] - dim[layer-1])//2
        relevance = relevance[ind:dim[layer-1]+ind,ind:dim[layer-1]+ind]
        return np.where(relevance > 0.33)



def retrieve_activations(net,img, val_im_empty_scenes, testing = 'normal', type = 'npy', focus = 'Munsells', prep = 'alban'):
	index_illu = randint(0,len(val_im_empty_scenes)-1)
	scene = val_im_empty_scenes[index_illu]
	x, mask = transform(img,scene, testing, type, preprocessing = prep)
	outputs = net(x)
	# Update: values here are for Original net
	if focus == 'Munsells':
	        relevant_conv1 = relevant_kernel_map(mask, 1, np.array([126,61,28]),np.array([1,1,1]), [5,3,3], 128)*1
	        relevant_conv2 = relevant_kernel_map(mask, 2, np.array([126,61,28]),np.array([1,1,1]), [5,3,3], 128)*1
	        relevant_conv3 = relevant_kernel_map(mask, 3, np.array([126,61,28]),np.array([1,1,1]), [5,3,3], 128)*1
	        conv1 = np.amax(outputs['conv1'].cpu().detach().numpy()[:,:,relevant_conv1[0],relevant_conv1[1]],axis = -1)
	        conv2 = np.amax(outputs['conv2'].cpu().detach().numpy()[:,:,relevant_conv2[0],relevant_conv2[1]],axis = -1)
	        conv3 = np.amax(outputs['conv3'].cpu().detach().numpy()[:,:,relevant_conv3[0],relevant_conv3[1]],axis = -1)
	else:
		conv1 = np.amax(outputs['conv1'].cpu().detach().numpy(),axis = (2,-1))
		conv2 = np.amax(outputs['conv2'].cpu().detach().numpy(),axis = (2,-1))
		conv3 = np.amax(outputs['conv3'].cpu().detach().numpy(),axis = (2,-1))
	fc1 = outputs['fc1'].cpu().detach().numpy()
	fc2 = outputs['fc2'].cpu().detach().numpy()
	_, p = torch.max(outputs['out'].data, 1)
	out = outputs['out'].cpu().detach().numpy()
	return conv1[0],conv2[0],conv3[0],fc1[0],fc2[0], out[0], (p.cpu().detach().numpy())[0]


def compute_outputs(net,img, val_im_empty_scenes, testing = 'normal', type = 'npy', focus = 'Munsells', prep = 'alban', layer = ''):
	index_illu = randint(0,len(val_im_empty_scenes)-1)
	scene = val_im_empty_scenes[index_illu]
	x, mask = transform(img, scene, testing, type, preprocessing = prep)
	outputs = net(x)
	if len(layer) >0:
	        out = np.amax(outputs[0].cpu().detach().numpy(),axis = (1,2))
	else:
	        out = outputs[0].cpu().detach().numpy()
	return out, np.argmax(out)



def evaluation_Readouts(net, img,  val_im_empty_scenes, readout_net, patch_nb=0, layer = 'fc2',testing = 'normal',type='npy'):
	index_illu = randint(0,len(val_im_empty_scenes)-1)
	scene = val_im_empty_scenes[index_illu]
	x, _ = transform(img,scene, patch_nb, testing, type)
	outputs = net(x)
	RC = readout_net(outputs[layer])
	_, p = torch.max(RC['out'].data, 1)
	#import pdb; pdb.set_trace()
	if testing == 'wrong_illu':
		return (RC['out'].cpu().detach().numpy())[0], (p.cpu().detach().numpy())[0], scene
	else:
		return (RC['out'].cpu().detach().numpy())[0], (p.cpu().detach().numpy())[0], 0
	

def evaluation(predictions, label):
    s = np.sum(predictions == label)
    return 100*s/predictions.size

def training_curves(training_dir, training_set, nb_epoch, Readout = False, layer = ''):
	import glob
	nb_mod = len(glob.glob(training_dir+'inst_*_%s'%training_set))
	Training_curv = np.zeros((nb_mod,nb_epoch))
	epochmax = np.zeros((nb_mod))
	for i in range(nb_mod):
		if Readout:
			Training_curv[i] = np.load( training_dir +'inst_%d_%s/readout_'%((i+1,training_set)) + layer + '/train_curve.npy' )
		else:
			Training_curv[i] = np.load( training_dir +'inst%dtrain_curve.npy' %((i)))
		epochmax[i] = np.argmax(Training_curv[i])
	return Training_curv, epochmax

import torch.nn as nn

class IntermediateModel(nn.Module):
    def __init__(self, original_model, layer_name):
        super(IntermediateModel, self).__init__()

        for i, (key, val) in enumerate(original_model.named_parameters()):
            if key == layer_name:
                layer_num = i
                break
        self.features = nn.Sequential(
            *list(original_model.features.children())[:layer_num]
        )

    def forward(self, x):
        x = self.features(x)
        return x
