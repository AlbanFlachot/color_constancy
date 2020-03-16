#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 15:51:54 2020

@author: alban
"""


PREPRE = np.moveaxis(PREDICTION,-1,1)

PREDICTION_CONFUSION = np.zeros((1600,1600))
for t in range(len(ARG)):
    arg = ARG[t]
    #print('Class %i is confused with class %i under illuminant %i' %(int(arg[2]), int(PREPRE[arg[0],arg[1],arg[2],arg[3]]), arg[3] ))
    PREDICTION_CONFUSION[int(arg[2]),int(PREPRE[arg[0],arg[1],arg[2],arg[3]])] += 1

fig = plt.figure()
plt.imshow(np.stack((PREDICTION_CONFUSION,PREDICTION_CONFUSION,PREDICTION_CONFUSION),axis = 2))
plt.xlabel('Given Munsell',fontsize = 15)
plt.ylabel('Correct Munsell',fontsize = 15)
plt.xticks(fontsize = 13)
plt.yticks(fontsize = 13)
plt.show()
fig.savefig('confusion_matrix.png', dpi=800)
plt.imshow(np.stack((PREDICTION_CONFUSION,PREDICTION_CONFUSION,PREDICTION_CONFUSION),axis = 2)/np.amax(PREDICTION_CONFUSION))
plt.show()



# In[9]: 3D CCI

PREDICTION_3D_error = EM.DIFF_LAB(PREDICTION, MUNSELL_LAB, list_WCS_labels,data_training = 'all')
PREDICTION_3D_error = np.moveaxis(PREDICTION_3D_error,-2,1)

CCI_L = 1 - np.absolute(PREDICTION_3D_error[:,:,:,:,0]/Displacement_LAB[:,:,0].T)
CCI_a = 1 - np.absolute(PREDICTION_3D_error[:,:,:,2:,1]/Displacement_LAB[2:,:,1].T)
CCI_b = 1 - np.absolute(PREDICTION_3D_error[:,:,:,:2,2]/Displacement_LAB[:2,:,2].T)


CCI[(1-CCI)<0] = 1

hY = np.histogram(1-CCI[:,:,:,0].flatten(),bins = np.arange(-1.05,1.1,0.05))
hB = np.histogram(1-CCI[:,:,:,1].flatten(),bins = np.arange(-1.05,1.1,0.05))
hG = np.histogram(1-CCI[:,:,:,2].flatten(),bins = np.arange(-1.05,1.1,0.05))
hR = np.histogram(1-CCI[:,:,:,3].flatten(),bins = np.arange(-1.05,1.1,0.05))


fig = plt.figure(figsize = (9,4))
ax1 = fig.add_subplot(131)
p1 = CCI_L
perc1 = np.nanpercentile(p1,np.arange(0,100,0.1))
ax1.plot(np.arange(0,10,0.1),perc1[:100], color= 'black')
ax1.hlines(np.nanmean(p1),xmin = 0, xmax = 10,color = 'r')

ax2 = fig.add_subplot(132,sharex=ax1)
p2 = CCI_a
perc2 = np.nanpercentile(p2,np.arange(0,100,0.1))
ax2.plot(np.arange(0,10,0.1),perc2[:100], color= 'black')
ax2.hlines(np.nanmean(p2),xmin = 0, xmax = 10,color = 'r')

ax3 = fig.add_subplot(133,sharex=ax1)
p3 = CCI_b
perc3 = np.nanpercentile(p3,np.arange(0,100,0.1))
ax3.plot(np.arange(0,10,0.1),perc3[:100], color= 'black')
ax3.hlines(np.nanmean(p3),xmin = 0, xmax = 10,color = 'r')

ax1.set_title('L*', fontsize = 15)
ax2.set_title('a*', fontsize = 15)
ax3.set_title('b*', fontsize = 15)

ax1.set_ylabel('CCI', fontsize = 15)
ax2.set_xlabel('Percentile', fontsize = 15)

fig.tight_layout
plt.show()


### Figure histogram CCIs.

fig = plt.figure(figsize = (9,4))
ax1 = fig.add_subplot(131)
p1 = CCI_L
h = ax1.hist((p1).flatten(), bins = np.arange(-1, 1.11, 0.1), color= 'black')
ax1.vlines(np.nanmean(p1),ymin = 0, ymax = 80000,color = 'r')
#ax1.vlines(np.nanpercentile(1-p1,25),ymin = 0, ymax = 80000,color = 'orange')
ax1.vlines(np.nanpercentile(p1,10),ymin = 0, ymax = 80000,color = 'g')
ax2 = fig.add_subplot(132,sharex=ax1)
p2 =  CCI_a
ax2.hist((p2).flatten(), bins = np.arange(-1, 1.11, 0.1), color= 'k')
ax2.vlines(np.nanmean((p2)),ymin = 0, ymax = 80000,color = 'r')
#ax2.vlines(np.nanpercentile((1-p2),25),ymin = 0, ymax = 80000,color = 'orange')
ax2.vlines(np.nanpercentile((p2),10),ymin = 0, ymax = 80000,color = 'g')
ax3 = fig.add_subplot(133,sharex=ax1)
p3 = CCI_b
ax3.hist((p3).flatten(), bins = np.arange(-1, 1.11, 0.1), color= 'k')
ax3.vlines(np.nanmean((p3)),ymin = 0, ymax = 80000,color = 'r')
#ax3.vlines(np.nanpercentile((1-p3),25),ymin = 0, ymax = 80000,color = 'orange')
ax3.vlines(np.nanpercentile((p3),10),ymin = 0, ymax = 80000,color = 'g')
ax2.set_yticks([])
ax3.set_yticks([])
ax1.set_title('L*', fontsize = 15)
ax2.set_title('a*', fontsize = 15)
ax3.set_title('b*', fontsize = 15)
ax1.set_xlim(-1,1.1)
ax2.set_xlim(-1,1.1)
ax3.set_xlim(-1,1.1)

ax1.set_ylabel('Count', fontsize = 15)
ax2.set_xlabel('CCI', fontsize = 15)

fig.tight_layout
plt.show()

### Figure histogram CCIs.

fig = plt.figure(figsize = (9,4))
ax2 = fig.add_subplot(121)
p2 =  CCI_a
ax2.hist((p2).flatten(), bins = np.arange(-1, 1.11, 0.1), color= 'k')
ax2.vlines(np.nanmedian((p2)),ymin = 0, ymax = 43000,color = 'r')
#ax2.vlines(np.nanpercentile((1-p2),25),ymin = 0, ymax = 80000,color = 'orange')
#ax2.vlines(np.nanpercentile((p2),10),ymin = 0, ymax = 53000,color = 'g')
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
ax3 = fig.add_subplot(122,sharex=ax2)
p3 = CCI_b
ax3.hist((p3).flatten(), bins = np.arange(-1, 1.11, 0.1), color= 'k')
ax3.vlines(np.nanmedian((p3)),ymin = 0, ymax = 43000,color = 'r')
#ax3.vlines(np.nanpercentile((1-p3),25),ymin = 0, ymax = 80000,color = 'orange')
#ax3.vlines(np.nanpercentile((p3),10),ymin = 0, ymax = 53000,color = 'g')
ax3.set_yticks([])
ax2.set_title('a*', fontsize = 15)
ax3.set_title('b*', fontsize = 15)
ax2.set_xlim(-1,1.1)
ax3.set_xlim(-1,1.1)
ax2.set_ylabel('Count', fontsize = 15)
ax2.set_xlabel('CCI', fontsize = 15)
ax3.set_xlabel('CCI', fontsize = 15)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
fig.savefig('/home/alban/Documents/pytorch/project_color_constancy/WCS_Xp/training_centered/POSTER_VSS_19/ab_distrib_CCI.png', dpi=800)
fig.tight_layout
plt.show()


### Figure displaying the percentile values of the distributions

fig = plt.figure(figsize = (9,4))
ax1 = fig.add_subplot(131)
p1 = 1 - np.absolute(PREDICTION_3D_error[:,:,:,:,0]/Displacement_LAB[:,:,0].T)
perc1 = np.nanpercentile(p1,np.arange(0.1,100,0.1))
ax1.plot(np.arange(0.1,100,0.1),perc1, color= 'black')
ax1.hlines(np.nanmean(p1),xmin = 0, xmax = 100,color = 'r')

ax2 = fig.add_subplot(132,sharex=ax1)
p2 = 1 - np.absolute(PREDICTION_3D_error[:,:,:,:,1]/Displacement_LAB[:,:,1].T)
perc2 = np.nanpercentile(p2,np.arange(0.1,100,0.1))
ax2.plot(np.arange(0.1,100,0.1),perc2, color= 'black')
ax2.hlines(np.nanmean(p2),xmin = 0, xmax = 100,color = 'r')

ax3 = fig.add_subplot(133,sharex=ax1)
p3 = 1 - np.absolute(PREDICTION_3D_error[:,:,:,:,2]/Displacement_LAB[:,:,2].T)
perc3 = np.nanpercentile(p3,np.arange(0.1,100,0.1))
ax3.plot(np.arange(0.1,100,0.1),perc3, color= 'black')
ax3.hlines(np.nanmean(p3),xmin = 0, xmax = 100,color = 'r')


ax1.set_title('L*', fontsize = 15)
ax2.set_title('a*', fontsize = 15)
ax3.set_title('b*', fontsize = 15)


ax1.set_ylabel('CCI', fontsize = 15)
ax2.set_xlabel('Percentile', fontsize = 15)

fig.tight_layout
plt.show()

fig = plt.figure(figsize = (9,4))
ax1 = fig.add_subplot(131)
p1 = 1 - np.absolute(PREDICTION_3D_error[:,:,:,:,0]/Displacement_LAB[:,:,0].T)
perc1 = np.nanpercentile(p1,np.arange(0.1,100,0.1))
ax1.plot(np.arange(0.1,10,0.1),perc1[:99], color= 'black')
ax1.hlines(np.nanmean(p1),xmin = 0, xmax = 10,color = 'r')

ax2 = fig.add_subplot(132,sharex=ax1)
p2 = 1 - np.absolute(PREDICTION_3D_error[:,:,:,:,1]/Displacement_LAB[:,:,1].T)
perc2 = np.nanpercentile(p2,np.arange(0.1,100,0.1))
ax2.plot(np.arange(0.1,10,0.1),perc2[:99], color= 'black')
ax2.hlines(np.nanmean(p2),xmin = 0, xmax = 10,color = 'r')

ax3 = fig.add_subplot(133,sharex=ax1)
p3 = 1 - np.absolute(PREDICTION_3D_error[:,:,:,:,2]/Displacement_LAB[:,:,2].T)
perc3 = np.nanpercentile(p3,np.arange(0.1,100,0.1))
ax3.plot(np.arange(0.1,10,0.1),perc3[:99], color= 'black')
ax3.hlines(np.nanmean(p3),xmin = 0, xmax = 10,color = 'r')

#ax2.set_yticks([])
#ax3.set_yticks([])
ax1.set_title('L*', fontsize = 15)
ax2.set_title('a*', fontsize = 15)
ax3.set_title('b*', fontsize = 15)
#ax1.set_xlim(-2,1)
#ax2.set_xlim(-2,1)
#ax3.set_xlim(-2,1)

ax1.set_ylabel('CCI', fontsize = 15)
ax2.set_xlabel('Percentile', fontsize = 15)

fig.tight_layout
plt.show()

