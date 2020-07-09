import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

'''
LIST OF FUNTIONS USED FOR COMPLEX FIGURES.
'''

def DEFINE_PLT_RC(type = 1):
    '''
    Function taht sets the rc parameters of matplot lib.e.g in an article
    e.g. in an article, whether it will be an image with columnwidth, or half of it, full page etc..
    INPUT:
        type: 1 = full page size; 1/2 = half of page size; 1/4 = quarter of page size; 1/3 = third of page size
    '''

    plt.rc('figure', figsize = (7,5))     # fig size bigger
    if type == 0:
        #plt.rc('font', weight='bold')    # bold fonts are easier to see
        plt.rc('font', size=8.5) # thicker black lines
        plt.rc('xtick', labelsize=7.5)
        plt.rc('ytick', labelsize=7.5)# tick labels bigger
        plt.rc('axes', labelsize=8.5)     # tick labels bigger
        plt.rc('lines', lw=2.5) # thicker black lines
    elif type == 1:
        #plt.rc('font', weight='bold')    # bold fonts are easier to see
        plt.rc('font', size=17) # thicker black lines
        plt.rc('xtick', labelsize=15)     # tick labels bigger
        plt.rc('ytick', labelsize=15)     # tick labels bigger
        plt.rc('axes', labelsize=17)     # tick labels bigger
        plt.rc('lines', lw=5) # thicker black lines
        #plt.rc('text', fontsize=17) # font of text
    elif type == 0.5:
        #plt.rc('font', weight='bold')    # bold fonts are easier to see
        plt.rc('font', size=35) # thicker black lines
        plt.rc('xtick', labelsize=30)     # tick labels bigger
        plt.rc('ytick', labelsize=30)     # tick labels bigger
        plt.rc('axes', labelsize=35)     # tick labels bigger
        plt.rc('lines', lw=10) # thicker black lines
        #plt.rc('text', fontsize=35) # thicker black lines
    elif type == 0.33:
        #plt.rc('font', weight='bold')    # bold fonts are easier to see
        plt.rc('font', size=28) # thicker black lines
        plt.rc('xtick', labelsize=24)     # tick labels bigger
        plt.rc('ytick', labelsize=24)     # tick labels bigger
        plt.rc('axes', labelsize=28)     # tick labels bigger
        plt.rc('lines', lw=7.5) # thicker black lines
        #plt.rc('text', fontsize=28) # thicker black lines

def scatter_LAB(LAB, RGB):
	'''
	Function to plot CIE lab coordinates and display point with some RGB colors
	'''
	fig = plt.figure(figsize = (7,6))
	ax = fig.add_subplot(111)
	ax.scatter(LAB.T[:,1], LAB.T[:,2],marker = 'o',color=RGB,s = 40)
	#ax.set_title('CIELab values under %s'%ill,fontsize = 18)
	ax.set_xlim(-100,100)
	ax.set_ylim(-100,100)
	ax.set_xlabel('a*',fontsize = 25)
	ax.set_ylabel('b*',fontsize = 25)
	plt.xticks(range(-100,110, 50),fontsize = 20)
	plt.yticks(range(-100,110, 50),fontsize = 20)
	fig.tight_layout()
	plt.show()
	#fig.savefig('/home/alban/Documents/pytorch/project_color_constancy/WCS_Xp/training_centered/POSTER_VSS_19/CIELab_MUNS_ab.png', dpi=800)
	plt.close()

	fig = plt.figure(figsize = (7,6))
	ax = fig.add_subplot(111)
	ax.scatter(LAB.T[:,1], LAB.T[:,0],marker = 'o',color=RGB,s = 40)
	#ax.set_title('CIELab values under %s'%ill,fontsize = 18)
	ax.set_xlim(-100,100)
	ax.set_ylim(0,100)
	ax.set_xlabel('a*',fontsize = 25)
	ax.set_ylabel('L*',fontsize = 25)
	plt.xticks(range(-100,110, 50),fontsize = 20)
	plt.yticks(range(0,110, 50),fontsize = 20)
	fig.tight_layout()
	plt.show()
	#fig.savefig('/home/alban/Documents/pytorch/project_color_constancy/WCS_Xp/training_centered/POSTER_VSS_19/CIELab_MUNS_aL.png', dpi=800)
	plt.close()


def scatter_MDS(RESULT,title,path1,path2,RGB_muns, LABELS = ['DIM 1','DIM 2','DIM 3', 'DIM4'], display = True):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(RESULT[:,0], RESULT[:,1], RESULT[:,2], marker='o',c = RGB_muns)
    ax.set_xlabel(LABELS[0],fontsize = 15)
    ax.set_ylabel(LABELS[1],fontsize = 15)
    ax.set_zlabel(LABELS[2],fontsize = 15)
    fig.text(0.5,0.95,title,ha='center',fontsize = 18)
    fig.tight_layout()
    if display:
        plt.show()
    #fig.savefig(path1,format='png', dpi=1200)
    fig.savefig(path1,format='png')
    plt.close()

    fig = plt.figure(figsize = (9,3))
    ax1 = fig.add_subplot(131)
    ax1.scatter(RESULT[:,0], RESULT[:,1], marker='o',c = RGB_muns)
    ax1.set_xlabel(LABELS[0],fontsize = 15)
    ax1.set_ylabel(LABELS[1],fontsize = 15)
    ax2 = fig.add_subplot(132)
    ax2.scatter(RESULT[:,1], RESULT[:,2], marker='o',c = RGB_muns)
    ax2.set_xlabel(LABELS[1],fontsize = 15)
    ax2.set_ylabel(LABELS[2],fontsize = 15)
    ax3 = fig.add_subplot(133)
    ax3.scatter(RESULT[:,2], RESULT[:,3], marker='o',c = RGB_muns)
    ax3.set_xlabel(LABELS[2],fontsize = 15)
    ax3.set_ylabel(LABELS[3],fontsize = 15)
    fig.text(0.5,0.94,title,ha='center',fontsize = 18)
    fig.tight_layout()
    if display:
        plt.show()
    #fig.savefig(path2,format='png', dpi=1200)
    fig.savefig(path2,format='png')
    plt.close()

def scatter_MDS_vert(RESULT,title,path1,path2,RGB_muns, LABELS = ['DIM 1','DIM 2','DIM 3', 'DIM4'], display = True):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(RESULT[:,0], RESULT[:,1], RESULT[:,2], marker='o',c = RGB_muns)
    ax.set_xlabel(LABELS[0],fontsize = 15)
    ax.set_ylabel(LABELS[1],fontsize = 15)
    ax.set_zlabel(LABELS[2],fontsize = 15)
    fig.text(0.5,0.95,title,ha='center',fontsize = 18)
    fig.tight_layout()
    if display:
        plt.show()
    #fig.savefig(path1,format='png', dpi=1200)
    fig.savefig(path1,format='png')
    plt.close()

    fig = plt.figure(figsize = (3.5,6))
    ax1 = fig.add_subplot(211)
    ax1.scatter(RESULT[:,0], RESULT[:,1], marker='o',c = RGB_muns)
    ax1.set_xlabel(LABELS[0],fontsize = 15)
    ax1.set_ylabel(LABELS[1],fontsize = 15)
    #ax1.set_xticks(np.arange(-0.5,0.6,0.5))
    #ax1.set_yticks(np.arange(-0.5,0.6,0.5))
    ax2 = fig.add_subplot(212)
    ax2.scatter(RESULT[:,1], RESULT[:,2], marker='o',c = RGB_muns)
    ax2.set_xlabel(LABELS[1],fontsize = 15)
    #ax2.set_ylabel(LABELS[2],fontsize = 15)
    ax2.set_xticks(np.arange(-0.5,0.6,0.5))
    ax2.set_yticks(np.arange(-0.5,0.6,0.5))
    #fig.text(0.5,0.94,title,ha='center',fontsize = 18)
    fig.tight_layout()
    if display:
        plt.show()
    #fig.savefig(path2,format='png', dpi=1200)
    fig.savefig(path2,format='png')
    plt.close()

def scatter_MDS2(RESULT,title,path1,path2,RGB_muns, LABELS = ['DIM 1','DIM 2','DIM 3'], display = True):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(RESULT[:,0], RESULT[:,1], RESULT[:,2], marker='o',c = RGB_muns)
    ax.set_xlabel(LABELS[0],fontsize = 15)
    ax.set_ylabel(LABELS[1],fontsize = 15)
    ax.set_zlabel(LABELS[2],fontsize = 15)
    fig.text(0.5,0.95,title,ha='center',fontsize = 18)
    fig.tight_layout()
    if display:
        plt.show()
    #fig.savefig(path1,format='png', dpi=1200)
    fig.savefig(path1,format='png')
    plt.close()

    fig = plt.figure(figsize = (9,3))
    ax1 = fig.add_subplot(131)
    ax1.scatter(RESULT[:,0], RESULT[:,1], marker='o',c = RGB_muns)
    ax1.set_xlabel(LABELS[0],fontsize = 15)
    ax1.set_ylabel(LABELS[1],fontsize = 15)
    ax2 = fig.add_subplot(132)
    ax2.scatter(RESULT[:,0], RESULT[:,2], marker='o',c = RGB_muns)
    ax2.set_xlabel(LABELS[0],fontsize = 15)
    ax2.set_ylabel(LABELS[2],fontsize = 15)
    ax3 = fig.add_subplot(133)
    ax3.scatter(RESULT[:,1], RESULT[:,2], marker='o',c = RGB_muns)
    ax3.set_xlabel(LABELS[1],fontsize = 15)
    ax3.set_ylabel(LABELS[2],fontsize = 15)
    fig.text(0.5,0.94,title,ha='center',fontsize = 18)
    fig.tight_layout()
    if display:
        plt.show()
    #fig.savefig(path2,format='png', dpi=1200)
    fig.savefig(path2,format='png')
    plt.close()



def display_munsells(WCS_MAT, norm):
    '''
    Function that displays the results per munsell and displaying themn according to the WCS coordinates
    8*40 colored munsells + 10 achromatic munsells.
    INPUTS:
        WCS_MAT: matrix of measures, usually of shape = [10, 41] (1st column is for achromatic,
                 after 1st and last rows are empty)
        norm: normalization factor to force values form 0 to 1, to be displayed.
    '''
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

        axchro.set_ylabel('Value',fontsize = 15)
        axchro.set_xlabel('Hue',fontsize = 15)
        #plt.setp(axchro.set_ylabel('Tuning |elevation| (deg)',fontsize = 25))
        plt.setp(axchro.get_xticklabels(), fontsize=12)
        plt.setp(axchro.get_yticklabels(), fontsize=12)

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

def display_munsells_inv(WCS_MAT, norm, title,save = True, add = 'a'):
    '''
    Function that displays the results per munsell and displaying themn according to the WCS coordinates
    8*40 colored munsells + 10 achromatic munsells.
    INPUTS:
        WCS_MAT: matrix of measures, usually of shape = [10, 41] (1st column is for achromatic,
                 after 1st and last rows are empty)
        norm: normalization factor to force values form 0 to 1, to be displayed.
    '''
    import matplotlib.ticker as ticker
    WCS_MAT2 = (WCS_MAT/norm)
    WCS_MAT = np.zeros(WCS_MAT2.shape)
    if WCS_MAT.shape == (8,40):
        for i in range(len(WCS_MAT2)): # rearange values (hence inv)
            WCS_MAT[i] = WCS_MAT2[7-i]
        fig = plt.figure(figsize = (8,3))
        plt.imshow(np.stack((WCS_MAT,WCS_MAT,WCS_MAT),axis = 2))
        plt.xlabel('Hue',fontsize = 15)
        plt.ylabel('Value',fontsize = 15)
        fig.tight_layout
        plt.show()

    else:
        for i in range(len(WCS_MAT2)): # rearange values (hence inv)
            WCS_MAT[i] = WCS_MAT2[9-i]
        # definitions for the axes
        left, width = 0.08, 0.82
        bottom, height = 0.08, 0.85
        left_h = left + width + 0.05
        bottom_h = 0

        rect_chro = [left, bottom, width, height]
        rect_achro = [left_h, bottom_h, 0.8/40, 1]
        
        left_s, bottom_s, width_s, height_s = 0.65,0.03, 0.2, 0.1
        scale = [left_s,  bottom_s, width_s,height_s]

        fig = plt.figure(1, figsize=(8, 3))

        axchro = plt.axes(rect_chro)
        axachro = plt.axes(rect_achro)
        axscale = plt.axes(scale)

        # image plot:
        WCS_MAT_chro = WCS_MAT[1:-1,1:]
        if WCS_MAT.shape == (10,41):
            WCS_MAT_achro = WCS_MAT[:,0].reshape(10,1)
            im_chro = np.stack((WCS_MAT_chro,WCS_MAT_chro,WCS_MAT_chro),axis = 2)
            im_achro = np.stack((WCS_MAT_achro,WCS_MAT_achro,WCS_MAT_achro),axis = 2)
        elif WCS_MAT.shape == (10,41,3):
            WCS_MAT_achro = WCS_MAT[:,0].reshape(10,1,3)
            im_chro = WCS_MAT_chro
            im_achro = WCS_MAT_achro
        axchro.imshow(im_chro)
        axchro.set_title(title, fontsize = 15)
        axchro.xaxis.set_major_locator(ticker.MultipleLocator(4))
        majors = ["","R", "YR", 'Y', 'GY', 'G', 'BG', 'B', 'PB', 'P', 'RP']
        axchro.xaxis.set_major_formatter(ticker.FixedFormatter(majors))
        axachro.imshow(im_achro)
        axachro.set_xticks([])
        axachro.yaxis.set_major_locator(ticker.MultipleLocator(2))
        majors = ["","1","3","5","7","9"]
        axachro.yaxis.set_major_formatter(ticker.FixedFormatter(majors))
        axchro.yaxis.set_major_locator(ticker.MultipleLocator(2))
        majors = ["","2","4","6","8"]
        axchro.yaxis.set_major_formatter(ticker.FixedFormatter(majors))
        #axchro.set_xticks(np.arange(0,40,4), ('R', 'YR', 'Y', 'GY', 'G', 'BG', 'B', 'PB', 'P', 'RP'))

        axchro.set_ylabel('Value',fontsize = 15)
        axchro.set_xlabel('Hue',fontsize = 15)

        plt.setp(axchro.get_xticklabels(), fontsize=14)
        plt.setp(axchro.get_yticklabels(), fontsize=14)
        plt.setp(axachro.get_yticklabels(), fontsize=14)
        
        a = np.arange(0,1.05,0.1).reshape(1,11)
        im_scale = np.stack((a,a,a), axis = 2)
        axscale.imshow(im_scale)
        axscale.set_xticks([])
        axscale.set_yticks([])
        fig.text(left_s - 0.015, bottom_s + height_s/2, str(0), horizontalalignment='center', verticalalignment='center', fontsize = 14)
        if norm >10 :
                max_im = int(norm)
        else:
                max_im = np.round(norm,1)
        fig.text(left_s + width_s + 0.025, bottom_s+ height_s/2, str(max_im), horizontalalignment='center',verticalalignment='center', fontsize = 14)
        fig.tight_layout
        plt.show()
        if save:
            fig.savefig(add,dpi = 300)


        #import pdb; pdb.set_trace()

 

def vis_square(data, name_fig):
    #Take an array of shape (n, height, width) or (n, height, width, 3)
       #and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)"""

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



