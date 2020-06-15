import numpy as np
import sys
sys.path.append('../../')

from utils_scripts import algos
from color_scripts import color_transforms as CT

txt_dir_path = '../../txt_files/'
npy_dir_path = '../../npy_files/'

'''
LIST OF FUNCTIONS USED for computing complex errors, Delta E, Color Contancy Index etc..
'''




def softmax(x):
    '''Compute softmax values for each sets of scores in x. Assumes the last dim is the dim upon make sum'''
    #import pdb; pdb.set_trace()
    x = x.T
    return (np.exp(x) / np.sum(np.exp(x), axis=0)).T

def PREDICTION_LAB(PREDICTION, data_training = 'all', test_WCS = True, space = 'CIELab', order = 'num'):
    '''Function that computes the error, in delta E distance, of the prediction given by the model.
    By default, considers the training to have been made on all 1600 munsells.

    Parameters:
        PREDICTION: array of predictions. Shape = [models, munsells, exemplar]
        MUNSELL_LAB: Corrdinates in CIElab of the 1600 munsell chips
        list_WCS_labels: array of the indexes of the WCS munsells in the list of 1600

    Returns:
        PREDICTION_ERROR: error, in delta i distance, of the prediction given by the model. Same shape as PREDICTION'''

    list_WCS_labels = algos.compute_WCS_Munsells_categories() # import idxes of WCS munsells among the 1600
    if order == 'alpha':
        muns_alpha2num = sorted(range(1600), key = str)
    if space == 'CIELab':
        MUNSELL_LAB = np.load(npy_dir_path +'MUNSELL_LAB.npy')
    elif space == 'Munsell':
        MUNSELL_LAB = np.load(npy_dir_path +'MUNS_COORDINATES_VHC.npy')
    PREDICTION_ERROR = np.zeros((PREDICTION.shape))
    for m in range(PREDICTION.shape[0]):
        for i in range(PREDICTION.shape[1]):
            for ill in range(PREDICTION.shape[2]):
                for exp in range(PREDICTION.shape[3]):
                    if data_training == 'all':
                        if order == 'alpha':
                            dist = np.linalg.norm(MUNSELL_LAB[list_WCS_labels[i]] - MUNSELL_LAB[muns_alpha2num[PREDICTION[m,i,ill,exp]]])
                        else:
                            dist = np.linalg.norm(MUNSELL_LAB[list_WCS_labels[i]] - MUNSELL_LAB[PREDICTION[m,i,ill,exp].astype(int).tolist()])
                    else:
                        dist = np.linalg.norm(MUNSELL_LAB[list_WCS_labels[i]] - MUNSELL_LAB[list_WCS_labels[PREDICTION[m,i,ill,exp].astype(int).tolist()]])
                    PREDICTION_ERROR[m,i,ill,exp] = dist

    return PREDICTION_ERROR

def PREDICTION_3D(PREDICTION, test_WCS = True, data_training = 'all', space = 'CIELab', order = 'num'):
    '''Function that computes the error, in delta E distance, of the prediction given by the model.
    By default, considers the training to have been made on all 1600 munsells.

    Parameters:
        PREDICTION: array of predictions. Shape = [models, munsells, exemplar]
        MUNSELL_LAB: Corrdinates in CIElab of the 1600 munsell chips
        list_WCS_labels: array of the indexes of the WCS munsells in the list of 1600

    Returns:
        PREDICTION_ERROR: error, in delta i distance, of the prediction given by the model. Same shape as PREDICTION'''

    list_WCS_labels = algos.compute_WCS_Munsells_categories() # import idxes of WCS munsells among the 1600
    if order == 'alpha':
        muns_alpha2num = sorted(range(1600), key = str)
    if space == 'CIELab':
        MUNSELL_LAB = np.load(npy_dir_path +'MUNSELL_LAB.npy')
    elif space == 'Munsell':
        MUNSELL_LAB = np.load(npy_dir_path +'MUNS_COORDINATES_VHC.npy')
    PREDICTION_ERROR = np.zeros((PREDICTION.shape) + tuple([3]))
    for m in range(PREDICTION.shape[0]):
        for i in range(PREDICTION.shape[1]):
            for ill in range(4):
                for exp in range(PREDICTION.shape[-1]):
                    if data_training == 'all':
                        if order == 'alpha':
                            dist = MUNSELL_LAB[list_WCS_labels[i]] - MUNSELL_LAB[muns_alpha2num[PREDICTION[m,i,ill,exp]].astype(int).tolist()]
                        else:
                            dist = MUNSELL_LAB[list_WCS_labels[i]] - MUNSELL_LAB[PREDICTION[m,i,ill,exp].astype(int).tolist()]
                    else:
                        dist = MUNSELL_LAB[list_WCS_labels[i]] - MUNSELL_LAB[list_WCS_labels[PREDICTION[m,i,ill,exp].astype(int).tolist()]]
                    PREDICTION_ERROR[m,i,ill,exp] = dist

    return PREDICTION_ERROR

def WEIGHTED_PREDICTION_LAB(OUT_soft, test_WCS = True, data_training = 'all', space = 'CIELab'):
    """Function that computes the error, in delta E distance, of the prediction given by the model.
    By default, considers the training to have been made on all 1600 munsells.

    Parameters:
        OUT_soft: array of predictions. Shape = [models, munsells, illuminants,exemplar, nb_outputs]

    Returns:
        PREDICTION_ERROR: error, in delta i distance, of the prediction given by the model. Same shape as PREDICTION"""

    list_WCS_labels = algos.compute_WCS_Munsells_categories() # import idxes of WCS munsells among the 1600
    if space == 'CIELab':
        MUNSELL_LAB = np.load(npy_dir_path +'MUNSELL_LAB.npy')
    elif space == 'Munsell':
        MUNSELL_LAB = np.load(npy_dir_path +'MUNS_COORDINATES_VXY.npy')
    PREDICTION_ERROR = np.zeros((OUT_soft.shape[:-1]))
    # Loop over models
    for m in range(OUT_soft.shape[0]):
        # Loop over munsells
        for i in range(OUT_soft.shape[1]):
            # Loop over illuminants
            for ill in range(OUT_soft.shape[2]):
                # Loop over examplars
                for exp in range(OUT_soft.shape[3]):
                    if test_WCS == True:# if we are only looping on the WCS munsells
                        if data_training == 'all':
                            dist = np.linalg.norm(MUNSELL_LAB[list_WCS_labels[i]] - np.average(MUNSELL_LAB, axis = 0,weights = OUT_soft[m,i,ill,exp]))
                        else:
                            dist = np.linalg.norm(MUNSELL_LAB[list_WCS_labels[i]] - np.average(MUNSELL_LAB[[list_WCS_labels]], axis = 0,weights = OUT_soft[m,i,ill,exp]))
                    else:# if we are looping on the 1600 munsells instead
                        dist = np.linalg.norm(MUNSELL_LAB[i] - np.average(MUNSELL_LAB, axis = 0,weights = OUT_soft[m,i,ill,exp]))
                    PREDICTION_ERROR[m,i,ill,exp] = dist
    return PREDICTION_ERROR

def WEIGHTED_PREDICTION_LAB_3D(OUT_soft, test_WCS = True, data_training = 'all', space = 'CIELab'):
    """Function that computes the error, in delta E distance, of the prediction given by the model.
    By default, considers the training to have been made on all 1600 munsells.

    Parameters:
        OUT_soft: array of predictions. Shape = [models, munsells, illuminants,exemplar, nb_outputs]

    Returns:
        PREDICTION_ERROR: error, in delta i distance, of the prediction given by the model. Same shape as PREDICTION"""

    list_WCS_labels = algos.compute_WCS_Munsells_categories() # import idxes of WCS munsells among the 1600
    if space == 'CIELab':
        MUNSELL_LAB = np.load(npy_dir_path +'MUNSELL_LAB.npy')
    elif space == 'Munsell':
        MUNSELL_LAB = np.load(npy_dir_path +'MUNS_COORDINATES_VHC.npy')
    PREDICTION_ERROR = np.zeros((OUT_soft.shape[:-1]) + tuple([3]))
    # Loop over models
    for m in range(OUT_soft.shape[0]):
        # Loop over munsells
        for i in range(OUT_soft.shape[1]):
            # Loop over illuminants
            for ill in range(OUT_soft.shape[2]):
                # Loop over examplars
                for exp in range(OUT_soft.shape[3]):
                    if test_WCS == True:# if we are only looping on the WCS munsells
                        if data_training == 'all':
                            if space == 'Munsell':
                                dist = MUNSELL_LAB[list_WCS_labels[i]] - CT.VXY2VHC(np.average(MUNSELL_LAB, axis = 0, weights = OUT_soft[m,i,ill,exp]))
                            else:
                                dist = MUNSELL_LAB[list_WCS_labels[i]] - np.average(MUNSELL_LAB, axis = 0, weights = OUT_soft[m,i,ill,exp])
                        else:
                            dist = MUNSELL_LAB[list_WCS_labels[i]] - np.average(MUNSELL_LAB[[list_WCS_labels]], axis = 0,weights = OUT_soft[m,i,ill,exp])
                    else:# if we are looping on the 1600 munsells instead
                        dist = MUNSELL_LAB[i] - np.average(MUNSELL_LAB, axis = 0,weights = OUT_soft[m,i,ill,exp])
                    PREDICTION_ERROR[m,i,ill,exp] = dist
    return PREDICTION_ERROR

def WEIGHTED_PREDICTION_LAB_4illu(OUT_soft, MUNSELL_LAB, list_WCS_labels, data_training = 'all'):
    """Function that computes the error, in delta E distance, of the prediction given by the model.
    By default, considers the training to have been made on all 1600 munsells.

    Parameters:
        OUT_soft: array of predictions. Shape = [models, munsells, illuminant, exemplar, nb_outputs]
        MUNSELL_LAB: Corrdinates in CIElab of the 1600 munsell chips
        list_WCS_labels: array of the indexes of the WCS munsells in the list of 1600

    Returns:
        PREDICTION_ERROR: error, in delta i distance, of the prediction given by the model. Same shape as PREDICTION"""

    PREDICTION_ERROR = np.zeros((OUT_soft.shape[:-1]))
    # Loop over models
    for m in range(OUT_soft.shape[0]):
        # Loop over munsells
        for i in range(OUT_soft.shape[1]):
            # Loop over illuminants
            for ill in range(OUT_soft.shape[2]):
                # Loop over examplars
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

def Weighted_DIFF_LAB_4(OUT_soft, WCS_LAB_4, list_WCS_labels, data_training = 'all'):
    """Function that computes the error, in delta E distance, of the weighted predictions given by the model.
    By default, considers the training to have been made on all 1600 munsells.

    Parameters:
        OUT_soft: array of predictions. Shape = [models, munsells, 4 illu, exemplars, classification cate]
        WCS_LAB_4: Coordinates in CIElab of the 330 munsell chips used in WCS
        list_WCS_labels: array of the indexes of the WCS munsells in the list of 1600

    Returns:
        PREDICTION_ERROR: error, in delta E distance, of the prediction given by the model. """
    PREDICTION_ERROR = np.zeros((OUT_soft.shape[:-1] + tuple([3])))
    for m in range(OUT_soft.shape[0]):
        for i in range(OUT_soft.shape[1]):
            for ill in range(OUT_soft.shape[2]):
                for exp in range(OUT_soft.shape[3]):
                    if data_training == 'all':
                        diff = (WCS_LAB_4[list_WCS_labels[i],ill] - np.average(WCS_LAB_4[:,ill], axis = 0, weights = OUT_soft[m,i,ill,exp]))
                    else:
                        diff = (WCS_LAB_4[i,ill] - np.average(WCS_LAB_4[:,ill], axis = 0, weights = OUT_soft[m,i,ill,exp]))
                    PREDICTION_ERROR[m,i,ill,exp] = diff
    return PREDICTION_ERROR


def evaluation(predictions, label):
    s = np.sum(predictions == label)
    return 100*s/len(predictions)

def evaluation5(predictions, label):
    s = 0
    for i in range(len(predictions)):
        s += (label in predictions[i])*1
    return (100*s)/len(predictions)

def error_muns(DE_VHC):
    '''
    Function that computes the error, in Munsell coordinates
    '''
    Hue_arr = np.arange(0,2*np.pi,2*np.pi/80)
    shape = DE_VHC.shape
    DE_VHC = DE_VHC.reshape(-1,3)
    DE_VHC[:,1] = DE_VHC[:,1]%80
    Hue_diff = Hue_arr[DE_VHC[:,1].astype(int)]
    error_hue = np.arccos(np.cos(Hue_diff))*np.sign(np.sin(Hue_diff))
    DE_VHC[:,1] = error_hue*180/np.pi/9
    DE_VHC = DE_VHC.reshape(shape)
    return DE_VHC


def evaluation_munscube(predictions):
    s = 0
    for i in range(len(predictions)):
        s += (True in (predictions[i]**2>1.5))*1
    return 100-(100*s)/len(predictions)

'''
def evaluation_munscube(predictions):
    s = 0
    for i in range(len(predictions)):
        s += (np.linalg.norm(predictions[i]) <= 1)*1
    return 100*(s/len(predictions))'''