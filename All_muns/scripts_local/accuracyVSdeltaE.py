
# In[1]:



from __future__ import print_function, division

import torch
import matplotlib.pyplot as plt
import numpy as np
import pickle
#import MODELS as M
import FUNCTIONS as F
import matplotlib.patheffects as PathEffects
import scipy.io as sio


# Ignore warnings
import warnings
warnings.filterwarnings("ignore")



# In[2]:


# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
#cudnn.benchmark = True

# In[9]: Compute accuracy as a measure of LAB distance

def softmax(x):
    """Compute softmax values for each sets of scores in x. Assumes the last dim is the dim upon make sum"""
    x = x.T
    return (np.exp(x) / np.sum(np.exp(x), axis=0)).T

def PREDICTION_LAB(PREDICTION, MUNSELL_LAB, list_WCS_labels, data_training = 'all'):
    from skimage import color
    """Function that computes the error, in delta i distance, of the prediction given by the model.
    By default, considers the training to have been made on all 1600 munsells.
    
    Parameters: 
        PREDICTION: array of predictions. Shape = [models, munsells, exemplar]
        MUNSELL_LAB: Corrdinates in CIElab of the 1600 munsell chips
        list_WCS_labels: array of the indexes of the WCS munsells in the list of 1600
        
    Returns:
        PREDICTION_ERROR: error, in delta i distance, of the prediction given by the model. Same shape as PREDICTION"""
    PREDICTION_ERROR = np.zeros((PREDICTION.shape))
    for m in range(PREDICTION.shape[0]):
        for i in range(PREDICTION.shape[1]):
            for ill in range(4):
                for exp in range(PREDICTION.shape[-1]):
                    if data_training == 'all':
                        dist = np.linalg.norm(MUNSELL_LAB[list_WCS_labels[i]] - 
                                                      MUNSELL_LAB[PREDICTION[m,i,ill,exp].astype(int).tolist()])
                    else:
                        dist = np.linalg.norm(MUNSELL_LAB[list_WCS_labels[i]] - 
                                                      MUNSELL_LAB[list_WCS_labels[PREDICTION[m,i,ill,exp].astype(int).tolist()]])
                    PREDICTION_ERROR[m,i,ill,exp] = dist 

    return PREDICTION_ERROR

def WEIGHTED_PREDICTION_LAB(OUT_soft, MUNSELL_LAB, list_WCS_labels, data_training = 'all'):
    """Function that computes the error, in delta i distance, of the prediction given by the model.
    By default, considers the training to have been made on all 1600 munsells.
    
    Parameters: 
        PREDICTION: array of predictions. Shape = [models, munsells, exemplar]
        MUNSELL_LAB: Corrdinates in CIElab of the 1600 munsell chips
        list_WCS_labels: array of the indexes of the WCS munsells in the list of 1600
        
    Returns:
        PREDICTION_ERROR: error, in delta i distance, of the prediction given by the model. Same shape as PREDICTION"""
    PREDICTION_ERROR = np.zeros(OUT_soft.shape[:-1])
    for m in range(OUT_soft.shape[0]):
        for i in range(OUT_soft.shape[1]):
            for ill in range(OUT_soft.shape[2]):
                for exp in range(OUT_soft.shape[3]):
                    if data_training == 'all':
                        dist = np.linalg.norm(MUNSELL_LAB[list_WCS_labels[i]] - 
                                                      np.average(MUNSELL_LAB, axis = 0,weights = OUT_soft[m,i,ill,exp]))
                    else:
                        dist = np.linalg.norm(MUNSELL_LAB[list_WCS_labels[i]] - 
                                                      np.average(MUNSELL_LAB[[list_WCS_labels]], axis = 0,weights = OUT_soft[m,i,ill,exp]))
                    PREDICTION_ERROR[m,i,ill,exp] = dist 

    return PREDICTION_ERROR

def DIFF_LAB(PREDICTION, MUNSELL_LAB, list_WCS_labels, data_training = 'all'):
    """Function that computes the error, in delta i distance, of the prediction given by the model.
    By default, considers the training to have been made on all 1600 munsells.
    
    Parameters: 
        PREDICTION: array of predictions. Shape = [models, munsells, exemplar]
        MUNSELL_LAB: Corrdinates in CIElab of the 1600 munsell chips
        list_WCS_labels: array of the indexes of the WCS munsells in the list of 1600
        
    Returns:
        PREDICTION_ERROR: error, in delta E distance, of the prediction given by the model. Same shape as PREDICTION + 3"""
    PREDICTION_ERROR = np.zeros((PREDICTION.shape + tuple([3])))
    for m in range(PREDICTION.shape[0]):
        for i in range(PREDICTION.shape[1]):
            for ill in range(PREDICTION.shape[2]):
                for exp in range(PREDICTION.shape[-1]):
                    if data_training == 'all':
                        diff = (MUNSELL_LAB[list_WCS_labels[i]] -
                                                      MUNSELL_LAB[PREDICTION[m,i,ill,exp].astype(int).tolist()])
                    else:
                        diff = (MUNSELL_LAB[list_WCS_labels[i]] -
                                                      MUNSELL_LAB[list_WCS_labels[PREDICTION[m,i,ill,exp].astype(int).tolist()]])
                    PREDICTION_ERROR[m,i,ill,exp] = diff
    return PREDICTION_ERROR

def DIFF_LAB_4(PREDICTION, WCS_LAB_4, list_WCS_labels, data_training = 'all'):
    """Function that computes the error, in delta E distance, of the prediction given by the model.
    By default, considers the training to have been made on all 1600 munsells.
    
    Parameters: 
        PREDICTION: array of predictions. Shape = [models, munsells, exemplar]
        MUNSELL_LAB: Corrdinates in CIElab of the 1600 munsell chips
        list_WCS_labels: array of the indexes of the WCS munsells in the list of 1600
        
    Returns:
        PREDICTION_ERROR: error, in delta i distance, of the prediction given by the model. Same shape as PREDICTION"""
    PREDICTION_ERROR = np.zeros((PREDICTION.shape + tuple([3])))
    for m in range(PREDICTION.shape[0]):
        for i in range(PREDICTION.shape[1]):
            for ill in range(PREDICTION.shape[2]):
                for exp in range(PREDICTION.shape[-1]):
                    if data_training == 'all':
                        diff = (WCS_LAB_4[i,ill] - WCS_LAB_4[PREDICTION[m,i,ill,exp].astype(int).tolist(),ill])
                    else:
                        diff = (WCS_LAB_4[i,ill] - WCS_LAB_4[PREDICTION[m,i,ill,exp].astype(int).tolist(),ill])
                    PREDICTION_ERROR[m,i,ill,exp] = diff
    return PREDICTION_ERROR

def Weighted_DIFF_LAB_4(OUT_soft, WCS_LAB_4, list_WCS_labels):
    """Function that computes the error, in delta E distance, of the prediction given by the model.
    By default, considers the training to have been made on all 1600 munsells.
    
    Parameters: 
        PREDICTION: array of predictions. Shape = [models, munsells, exemplar]
        MUNSELL_LAB: Corrdinates in CIElab of the 1600 munsell chips
        list_WCS_labels: array of the indexes of the WCS munsells in the list of 1600
        
    Returns:
        PREDICTION_ERROR: error, in delta i distance, of the prediction given by the model. Same shape as PREDICTION"""
    PREDICTION_ERROR = np.zeros((OUT_soft.shape[:-1] + tuple([3])))
    for m in range(OUT_soft.shape[0]):
        for i in range(OUT_soft.shape[1]):
            for ill in range(OUT_soft.shape[2]):
                for exp in range(OUT_soft.shape[3]):
                    diff = (WCS_LAB_4[i,ill] - np.average(WCS_LAB_4[:,ill], axis = 0, weights = OUT_soft[m,i,ill,exp]))
                    PREDICTION_ERROR[m,i,ill,exp] = diff
    return PREDICTION_ERROR

def evaluation(predictions,label):
    s = np.sum(predictions == label)
    return 100*s/len(predictions)



# In[9]: load mnodels results

pf2 = np.load('/home/alban/mnt/awesome/alban/project_color_constancy/PYTORCH/WCS/train_centered/WCS/pf2_4illu.npy')
pf2_no_patch = np.load('/home/alban/mnt/awesome/alban/project_color_constancy/PYTORCH/WCS/train_centered/WCS/pf2_4illu_no_patch.npy')
pf2_wrong_illu = np.load('/home/alban/mnt/awesome/alban/project_color_constancy/PYTORCH/WCS/train_centered/WCS/pf2_4illu_wrong_illu.npy')
pf2_no_back = np.load('/home/alban/mnt/awesome/alban/project_color_constancy/PYTORCH/WCS/train_centered/WCS/pf2_4illu_no_back.npy')


out_f2 = softmax(np.load('/home/alban/mnt/awesome/alban/project_color_constancy/PYTORCH/WCS/train_centered/WCS/out_f2_4illu.npy'))
out_f2_no_patch = softmax(np.load('/home/alban/mnt/awesome/alban/project_color_constancy/PYTORCH/WCS/train_centered/WCS/out_f2_4illu_no_patch.npy'))
out_f2_wrong_illu = softmax(np.load('/home/alban/mnt/awesome/alban/project_color_constancy/PYTORCH/WCS/train_centered/WCS/out_f2_4illu_wrong_illu.npy'))
out_f2_no_back = softmax(np.load('/home/alban/mnt/awesome/alban/project_color_constancy/PYTORCH/WCS/train_centered/WCS/out_f2_4illu_no_back.npy'))


nb_mod = 10
nb_obj = 330

#
# In[9]: Load WCS coordinates
L = list()
with open("/home/alban/Documents/pytorch/project_color_constancy/WCS_Xp/WCS_indx.txt") as f:
    for line in f:
       L.append(line.split())
      
WCS_X = [ord(char[0][0].lower()) - 97 for char in L]
WCS_Y = [int(char[0][1:]) for char in L]

WCS_X1 = [9-x for x in WCS_X]



# In[9]: Accuracy per munsell


CHROMA = list()
with open("/home/alban/Documents/pytorch/project_color_constancy/WCS_Xp/WCS_chroma.txt") as f:
    for line in f:
       CHROMA.append(int(line.split()[0]))    



WCS_MAT_CHROMA = np.zeros((10,41))
count = 0
for i in range(nb_obj):
    WCS_MAT_CHROMA[WCS_X[i],WCS_Y[i]] = float(CHROMA[count])
    count +=1
F.display_munsells_inv(WCS_MAT_CHROMA,16)


mat_speakers = sio.loadmat('/home/alban/Documents/pytorch/project_color_constancy/WCS_Xp/matrix_WCS_speakers.mat')['WCS_speakers'] 


mat_consistency = sio.loadmat('/home/alban/Documents/pytorch/project_color_constancy/WCS_Xp/consistency_map4Alban.mat')['cmap']
F.display_munsells_inv(mat_consistency,1)

mat_consistency2 = np.zeros(mat_consistency.shape)
for i in range(len(mat_consistency2)):
    mat_consistency2[i] = mat_consistency[7-i]
F.display_munsells_inv(mat_consistency2,1)

general_mat_consistency = sio.loadmat('/home/alban/Documents/pytorch/project_color_constancy/WCS_Xp/general_consensusconsistency_map4Alban.mat')['cmap_general']
general_mat_consistency2 = np.zeros(general_mat_consistency.shape)
for i in range(len(general_mat_consistency2)):
    general_mat_consistency2[i] = general_mat_consistency[7-i]
F.display_munsells(general_mat_consistency2[0:-3],1)



WCS_LAB_4 = np.moveaxis(np.load('/home/alban/Documents/pytorch/project_color_constancy/WCS_Xp/Test_4_illu/LAB_WCS_ABCD.npy'),0,-1)

WCS_muns = list()
with open("/home/alban/Documents/pytorch/project_color_constancy/WCS_Xp/WCS_muns.txt") as f:
    for line in f:
       WCS_muns.append(line.split()[0])

   
All_muns = list()
with open("/home/alban/Documents/pytorch/project_color_constancy/WCS_Xp/munsell_labels.txt") as f:
    for line in f:
       All_muns.append(line.split()[0])

list_WCS_labels = np.asarray([All_muns.index(WCS_muns[i]) for i in range(len(WCS_muns))])

LAB_WCS = np.load('/home/alban/Documents/pytorch/project_color_constancy/WCS_Xp/WCS_LAB.npy')
MUNSELL_LAB = np.load('/home/alban/Documents/pytorch/project_color_constancy/WCS_Xp/MUNSELL_LAB.npy')

# In[9]: Accuracy 

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.linalg.norm(array - value,axis = -1)).argmin()
    return idx

WCS_LAB_all = np.moveaxis(WCS_LAB_4, 0,1)

Displacement_LAB = WCS_LAB_all - LAB_WCS.T

prediction_no_cc = np.zeros((WCS_LAB_all.shape[0],WCS_LAB_all.shape[1]))
good_prediction_no_cc = prediction_no_cc.copy()
for ill in range(WCS_LAB_all.shape[0]):
    for muns in range(WCS_LAB_all.shape[1]):
        prediction_no_cc[ill,muns] = find_nearest(LAB_WCS.T, WCS_LAB_all[ill,muns])
        if prediction_no_cc[ill,muns] == muns:
            good_prediction_no_cc[ill,muns] = 1
        
Acc_no_cc = np.sum(good_prediction_no_cc)/good_prediction_no_cc.size

prediction_acc_MAT = np.zeros((10,41))

for i in range(nb_obj):
    prediction_acc_MAT[WCS_X[i],WCS_Y[i]] = np.mean(prediction_no_cc[:,i])

F.display_munsells_inv(prediction_acc_MAT,np.amax(prediction_acc_MAT))


PREDICTION_Accuracy_F2_no_patch = np.zeros((nb_mod,nb_obj))
PREDICTION_Accuracy_F2 = np.zeros((nb_mod,nb_obj))
PREDICTION_Accuracy_F2_wrong_illu = np.zeros((nb_mod,nb_obj))
PREDICTION_Accuracy_F2_no_back = np.zeros((nb_mod,nb_obj))

for m in range(nb_mod):
    for i in range(nb_obj):
        PREDICTION_Accuracy_F2[m,i] = evaluation(pf2[m,i].flatten(),i)
        PREDICTION_Accuracy_F2_no_patch[m,i] = evaluation(pf2_no_patch[m,i].flatten(),i)
        PREDICTION_Accuracy_F2_wrong_illu[m,i] = evaluation(pf2_wrong_illu[m,i].flatten(),i)
        PREDICTION_Accuracy_F2_no_back[m,i] = evaluation(pf2_no_back[m,i].flatten(),i)


Acc_full_im = np.mean(PREDICTION_Accuracy_F2)
Acc_no_patch = np.mean(PREDICTION_Accuracy_F2_no_patch)
Acc_f_wrong_illu = np.mean(PREDICTION_Accuracy_F2_wrong_illu)
Acc_no_back = np.mean(PREDICTION_Accuracy_F2_no_back)


fig = plt.figure(figsize = (4,4))
ax1 = fig.add_subplot(111)
ax1.bar([1,2,3,4,5],[Acc_full_im, Acc_no_patch, Acc_f_wrong_illu, Acc_no_back, 2.1],color = ['k',[0.4,0.7,0.8],[0.4,0.8,0.4],[0.7,0.8,0.4],[0.8,0.4,0.4],'grey'],linewidth = 6)
#ax1.errorbar([1,2,3,4,5],CCI_all_no_patch,yerr = CCI_bar_no_patch,color = [0.4,0.7,0.8],linewidth = 6)
#ax1.set_xlabel('Readouts',fontsize = 15)
ax1.set_xticks([1,2,3,4,5])
ax1.set_xticklabels(['full image','no patch','wrong\nback','no\nback','D65 only'],rotation=-75,fontsize = 25)
plt.xticks(fontsize = 21)
#plt.yticks(fontsize = 14)
plt.yticks(np.arange(0,105,25),fontsize = 21)
ax1.set_ylabel('Accuracy',fontsize = 25)
fig.tight_layout()
fig.savefig('/home/alban/Documents/pytorch/project_color_constancy/WCS_Xp/training_centered/POSTER_VSS_19/Accuracy.png', dpi=800)
plt.show()


# In[9]: Delta E

prediction_DE = np.linalg.norm(Displacement_LAB,axis = -1)
 

PREDICTION_DE_F2 = WEIGHTED_PREDICTION_LAB(out_f2, MUNSELL_LAB, list_WCS_labels,data_training = 'WCS')
PREDICTION_DE_F2_no_patch = WEIGHTED_PREDICTION_LAB(out_f2_no_patch, MUNSELL_LAB, list_WCS_labels,data_training = 'WCS')
PREDICTION_DE_F2_wrong_illu = WEIGHTED_PREDICTION_LAB(out_f2_wrong_illu, MUNSELL_LAB, list_WCS_labels,data_training = 'WCS')
PREDICTION_DE_F2_no_back = WEIGHTED_PREDICTION_LAB(out_f2_no_back, MUNSELL_LAB, list_WCS_labels,data_training = 'WCS')


DeltaE_full_im = np.median(PREDICTION_DE_F2)
DeltaE_no_patch = np.median(PREDICTION_DE_F2_no_patch)
DeltaE_wrong_illu = np.median(PREDICTION_DE_F2_wrong_illu)
DeltaE_no_back = np.median(PREDICTION_DE_F2_no_back)

fig = plt.figure(figsize = (4,4))
ax1 = fig.add_subplot(111)
ax1.bar([1,2,3,4,5],[DeltaE_full_im, DeltaE_no_patch, DeltaE_wrong_illu, DeltaE_no_back, 35],color = ['k',[0.4,0.7,0.8],[0.4,0.8,0.4],[0.7,0.8,0.4],[0.8,0.4,0.4]],linewidth = 6)
#ax1.errorbar([1,2,3,4,5],CCI_all_no_patch,yerr = CCI_bar_no_patch,color = [0.4,0.7,0.8],linewidth = 6)
#ax1.set_xlabel('Readouts',fontsize = 15)
ax1.set_xticks([1,2,3,4,5])
ax1.set_xticklabels(['full image','no patch','wrong\nback','no\nback','D65 only'],rotation=-75,fontsize = 25)
plt.xticks(fontsize = 21)
#plt.yticks(fontsize = 14)
plt.yticks(fontsize = 21)
ax1.set_ylabel('$\Delta$ E',fontsize = 25)
fig.tight_layout()
fig.savefig('/home/alban/Documents/pytorch/project_color_constancy/WCS_Xp/training_centered/POSTER_VSS_19/DE.png', dpi=800)
plt.show()


# In[9]: CCI

fig = plt.figure(figsize = (4,4))
ax1 = fig.add_subplot(111)
ax1.bar([1,2,3,4,5],[0.91, 0.66, 0, 0, 0],color = ['k',[0.4,0.7,0.8],[0.4,0.8,0.4],[0.7,0.8,0.4],[0.8,0.4,0.4]],linewidth = 6)
#ax1.errorbar([1,2,3,4,5],CCI_all_no_patch,yerr = CCI_bar_no_patch,color = [0.4,0.7,0.8],linewidth = 6)
#ax1.set_xlabel('Readouts',fontsize = 15)
ax1.set_xticks([1,2,3,4,5,6])
ax1.set_xticklabels(['full image','no patch','wrong\nback','no\nback','D65 only'],rotation=-75,fontsize = 25)
plt.xticks(fontsize = 21)
#plt.yticks(fontsize = 14)
plt.yticks(fontsize = 21)
ax1.set_ylabel('CCI',fontsize = 25)
fig.tight_layout()
fig.savefig('/home/alban/Documents/pytorch/project_color_constancy/WCS_Xp/training_centered/POSTER_VSS_19/CCI.png', dpi=800)
plt.show()