import numpy as np
import sys
sys.path.append('../../')

from utils_scripts import algos

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

def PREDICTION_LAB(PREDICTION, MUNSELL_LAB, list_WCS_labels, data_training = 'all'):
    '''Function that computes the error, in delta E distance, of the prediction given by the model.
    By default, considers the training to have been made on all 1600 munsells.

    Parameters:
        PREDICTION: array of predictions. Shape = [models, munsells, exemplar]
        MUNSELL_LAB: Corrdinates in CIElab of the 1600 munsell chips
        list_WCS_labels: array of the indexes of the WCS munsells in the list of 1600

    Returns:
        PREDICTION_ERROR: error, in delta i distance, of the prediction given by the model. Same shape as PREDICTION'''

    list_WCS_labels = algos.compute_WCS_Munsells_categories() # import idxes of WCS munsells among the 1600
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

def WEIGHTED_PREDICTION_LAB(OUT_soft, test_WCS = True, data_training = 'all'):
    """Function that computes the error, in delta E distance, of the prediction given by the model.
    By default, considers the training to have been made on all 1600 munsells.

    Parameters:
        OUT_soft: array of predictions. Shape = [models, munsells, illuminants,exemplar, nb_outputs]

    Returns:
        PREDICTION_ERROR: error, in delta i distance, of the prediction given by the model. Same shape as PREDICTION"""

    list_WCS_labels = algos.compute_WCS_Munsells_categories() # import idxes of WCS munsells among the 1600
    MUNSELL_LAB = np.load(npy_dir_path +'MUNSELL_LAB.npy')
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
                            dist = np.linalg.norm(MUNSELL_LAB[list_WCS_labels[i]] -
                                                          np.average(MUNSELL_LAB, axis = 0,weights = OUT_soft[m,i,ill,exp]))
                        else:
                            dist = np.linalg.norm(MUNSELL_LAB[list_WCS_labels[i]] -
                                                          np.average(MUNSELL_LAB[[list_WCS_labels]], axis = 0,weights = OUT_soft[m,i,ill,exp]))
                    else:# if we are looping on the 1600 munsells instead
                        dist = np.linalg.norm(MUNSELL_LAB[i] -
                                                      np.average(MUNSELL_LAB, axis = 0,weights = OUT_soft[m,i,ill,exp]))
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


def evaluation(predictions,label):
    s = np.sum(predictions == label)
    return 100*s/len(predictions)
