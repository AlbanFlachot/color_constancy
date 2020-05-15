import numpy as np
import torch
import cv2

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

def transform(img,testing = 'normal', type = 'npy'):
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
		I_mask = np.load(img[:-4] + '_mask.npy')
		MASK = np.mean(I_mask, axis = -1)
		MASK[MASK > np.amax(MASK)/255] = 1
		MASK[MASK != 1] = 0
		if testing == 'no_patch':
			trans_im = I.copy()
			local_mean = np.mean(trans_im[0:8,27:100],axis = (0))
			band = np.tile(local_mean[np.newaxis,:,:],(11,1,1))
			local_std= np.std(trans_im[0:8,10:115])
			lum_noise = np.random.normal(0,local_std/10,(11,73))
			trans_im[8:19,27:100] = band+ np.tile(lum_noise[:,:,np.newaxis],(1,1,3))
			I = trans_im.copy()
		elif testing == 'wrong_illu':  
			scene = val_im_empty_scenes[index_illu]
			SCENE = np.load(scene)                     
			SCENE[MASK==1] = I[MASK==1]         
			I = SCENE.astype(float)
		I = np.moveaxis(I,-1,0)
		I = I - 3
		x = torch.from_numpy(I)
		x = x.type(torch.FloatTensor)
	else:
		x = torch.load(img)
	x  = x.unsqueeze(0)
	x = x.to(device)
	return x, MASK

def receptive_field(layer, stride, window):
        rf = 1
        for count in range(layer):
                rf += np.power(2,count)*stride[count]*(window[count] - 1)
        return rf


def relevant_kernel_map(mask, layer, dim , stride , window , im_size):
        '''
        Function which finds which kernel of convolutional layer are detecting colored object within scene
        '''
        
        rf = window[0]
        rf = receptive_field(layer, stride, window)	
        relevance = cv2.blur(mask.astype('float32'), (rf, rf))[::np.power(2,layer-1),::np.power(2,layer-1)]
        ind = (relevance.shape[0] - dim[layer-1])//2
        relevance = relevance[ind:dim[layer-1]+ind,ind:dim[layer-1]+ind]
        return np.where(relevance > 0.33)



def retrieve_activations(net,img, val_im_empty_scenes, testing = 'normal', type = 'npy'):
	x, mask = transform(img,testing, type)
	outputs = net(x)
	# Update: values here are for Original net
	relevant_conv1 = relevant_kernel_map(mask, 1, np.array([126,61,28]),np.array([1,1,1]), [5,3,3], 128)*1
	relevant_conv2 = relevant_kernel_map(mask, 2, np.array([126,61,28]),np.array([1,1,1]), [5,3,3], 128)*1
	relevant_conv3 = relevant_kernel_map(mask, 3, np.array([126,61,28]),np.array([1,1,1]), [5,3,3], 128)*1
	conv1 = np.mean(outputs['conv1'].cpu().detach().numpy()[:,:,relevant_conv1[0],relevant_conv1[1]],axis = -1)
	conv2 = np.mean(outputs['conv2'].cpu().detach().numpy()[:,:,relevant_conv2[0],relevant_conv2[1]],axis = -1)
	conv3 = np.mean(outputs['conv3'].cpu().detach().numpy()[:,:,relevant_conv3[0],relevant_conv3[1]],axis = -1)
	fc1 = outputs['fc1'].cpu().detach().numpy()
	fc2 = outputs['fc2'].cpu().detach().numpy()
	_, p = torch.max(outputs['out'].data, 1)
	out = outputs['out'].cpu().detach().numpy()
	return conv1[0],conv2[0],conv3[0],fc1[0],fc2[0], out[0], (p.cpu().detach().numpy())[0]

def evaluation_Readouts(net,img,readout_net,layer = 'fc2',testing = 'normal',type='npy'):
	x, _ = transform(img,testing, type)
	outputs = net(x)
	RC = readout_net(outputs[layer])
	_, p = torch.max(RC['out'].data, 1)
	#import pdb; pdb.set_trace()
	return (RC['out'].cpu().detach().numpy())[0], (p.cpu().detach().numpy())[0]
	

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

