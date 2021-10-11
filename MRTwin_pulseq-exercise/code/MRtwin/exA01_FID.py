"""
Created on Tue Jan 29 14:38:26 2019
@author: mzaiss

"""
experiment_id = 'ex01_FID'
sequence_class = "super"
experiment_description = """
FID or 1 D imaging / spectroscopy
"""
excercise = """
A01.1. alter flip angle rf_event[3,0,0], find flip angle for max signal, guess the function signal(flip_angle) ~= ...
A01.2  phantom[:,:,3] += 1000 # Tweak dB0  do this to see lab frame movement, then 0 again.
A01.4. set flip to 90 and alter number of spins: How many spins are at least needed to get good approximation of NSpins=Inf.
A01.5. alter rf phase and adc rot
A01.6. alter event_time
A01.7. uncomment FITTING BLOCK, fit signal, alter R2star, where does the deviation come from?
"""
print(excercise)
#%%
#matplotlib.pyplot.close(fig=None)
#%%
import os, sys
import numpy as np
import scipy
import scipy.io
from  scipy import ndimage
from  scipy import optimize
import torch
import cv2
import matplotlib.pyplot as plt
from torch import optim
import core.spins
import core.scanner
import core.nnreco
import core.target_seq_holder
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)
import time; today_datestr = time.strftime('%y%m%d')
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"

from importlib import reload
reload(core.scanner)

double_precision = False
do_scanner_query = False

use_gpu = 0
gpu_dev = 0

print(experiment_id)    
print('use_gpu = ' +str(use_gpu)) 

# NRMSE error function
def e(gt,x):
    return 100*np.linalg.norm((gt-x).ravel())/np.linalg.norm(gt.ravel())
    
# torch to numpy
def tonumpy(x):
    return x.detach().cpu().numpy()

# get magnitude image
def magimg(x):
  return np.sqrt(np.sum(np.abs(x)**2,2))

def magimg_torch(x):
  return torch.sqrt(torch.sum(torch.abs(x)**2,1))

def tomag_torch(x):
    return torch.sqrt(torch.sum(torch.abs(x)**2,-1))

# device setter
def setdevice(x):
    if double_precision:
        x = x.double()
    else:
        x = x.float()
    if use_gpu:
        x = x.cuda(gpu_dev)    
    return x 

#############################################################################
## S0: define image and simulation settings::: #####################################
sz = np.array([4,4])                      # image size
NVox = sz[0]*sz[1]                        # number if voxels
extraMeas = 1                             # number of measurmenets/ separate scans
NRep = extraMeas*sz[1]                    # number of total repetitions
NRep = 1                                  # number of total repetitions  # phase encoding
szread=128                                # number of ADC or readout events  # frequency encoding
NEvnt = szread + 5 + 2                    # number of events F/R/P   5:pulse加载地方  2:delay如fully relax  128:信号收集
NSpins = 24**2                            # number of spin sims in each voxel
NCoils = 1                                # number of receive coil elements
noise_std = 0*1e-3                        # additive Gaussian noise std
kill_transverse = False                   # set transverse magentization to zero after each repetition, emulates full spoiling

#############################################################################
## S1: Init spin system and phantom::: #####################################
# initialize scanned object
spins = core.spins.SpinSystem(sz,NVox,NSpins,use_gpu+gpu_dev,double_precision=double_precision)

# either (i) load phantom (third dimension: PD, T1 T2 dB0 rB1)
phantom = spins.get_phantom(sz[0],sz[1],type='object1')  # type='object1' or 'brain1'

# or (ii) set phantom  manually
phantom = np.zeros((sz[0],sz[1],5), dtype=np.float32); 
phantom[1,1,:]=np.array([1, 1, 0.1, 0, 1]) # third dimension: PD, T1 T2 dB0 rB1

# adjust phantom
phantom[:,:,1] *= 1 # Tweak T1
phantom[:,:,2] *= 1 # Tweak T2
phantom[:,:,3] += 0 # Tweak dB0  磁场不均匀
phantom[:,:,4] *= 1 # Tweak rB1

if 1: # switch on for plot
    plt.figure("""phantom"""); plt.clf();  param=['PD','T1 [s]','T2 [s]','dB0 [Hz]','rB1 [rel.]']
    for i in range(5):
        plt.subplot(151+i), plt.title(param[i])
        ax=plt.imshow(phantom[:,:,i], interpolation='none')
        fig = plt.gcf(); fig.colorbar(ax) 
    fig.set_size_inches(18, 3); plt.show()

spins.set_system(phantom,R2dash=250.0)  # set phantom variables with overall constant R2' = 1/T2'  (R2*=R2+R2')

## end of S1: Init spin system and phantom ::: #####################################

#############################################################################
## S2: Init scanner system ::: #####################################
scanner = core.scanner.Scanner(sz,NVox,NSpins,NRep,NEvnt,NCoils,noise_std,use_gpu+gpu_dev,double_precision=double_precision)
#scanner.set_B1plus(phantom[:,:,4])  # use as defined in phantom
scanner.set_B1plus(1)               # overwriet with homogeneous excitation

#############################################################################
## S3: MR sequence definition ::: #####################################
# begin sequence definition
# allow for extra events (pulses, relaxation and spoiling) in the first five and last two events (after last readout event)
adc_mask = torch.from_numpy(np.zeros((NEvnt,1))).float()
adc_mask[5:-2]  = 1  # acqire data from event 5 to -2
scanner.set_adc_mask(adc_mask)

# RF events: rf_event and phases
rf_event = torch.zeros((NEvnt,NRep,2), dtype=torch.float32)
rf_event[3,0,0] = 90*np.pi/180  # 0:flip angle; 1:phase angle; 2:flag
# rf_event[3,0,1] = 180*np.pi/180  # 0:flip angle; 1:phase angle; 2:flag #143+144就改变了rf phase
rf_event = setdevice(rf_event)  

scanner.set_flip_tensor_withB1plus(rf_event)
# rotate ADC according to excitation phase
# we want that the ADC phase follows the rf phase
rfsign = ((rf_event[3,:,0]) < 0).float() # translate neg flips to pos flips with 180 deg phase shift.
scanner.set_ADC_rot_tensor(-rf_event[3,:,1] + np.pi/2 + np.pi*rfsign) #GRE/FID specific

# event timing vector  改变180°发生的时间;增加delay_time
event_time = torch.from_numpy(0.08*1e-3*np.ones((NEvnt,NRep))).float()
event_time[-1,0] = 1
event_time = setdevice(event_time)

# gradient-driver precession
# Cartesian encoding    gradm_event: define position in kspace gradient moment g = 积分G(t)·r·dt
gradm_event = torch.zeros((NEvnt,NRep,2), dtype=torch.float32)
gradm_event = setdevice(gradm_event)

scanner.init_gradient_tensor_holder()
scanner.set_gradient_precession_tensor(gradm_event,sequence_class)  # refocusing=False for GRE/FID, adjust for higher echoes
## end S3: MR sequence definition ::: #####################################


#############################################################################
## S4: MR simulation forward process ::: #####################################
scanner.init_signal()
scanner.forward(spins, event_time)

# sequence and signal plotting
targetSeq = core.target_seq_holder.TargetSequenceHolder(rf_event,event_time,gradm_event,scanner,spins,scanner.signal)
#targetSeq.print_seq_pic(True,plotsize=[12,9])
targetSeq.print_seq(plotsize=[12,9], time_axis=1)  #time_axis=0 is event_idx axis; 1 is real time axis

# do it yourself: sequence and signal plotting  
fig=plt.figure("""signals""")
ax1=plt.subplot(131)
ax=plt.plot(tonumpy(scanner.signal[0,:,:,0,0]).transpose().ravel(),label='real')
plt.plot(tonumpy(scanner.signal[0,:,:,1,0]).transpose().ravel(),label='imag')
plt.title('signal')
plt.legend()
plt.ion()

plt.show()

# do it yourself: sequence and signal plotting 
#fig=plt.figure("""seq and signal"""); fig.set_size_inches(64, 7)
#plt.subplot(311); plt.title('seq: RF, time, ADC')
#plt.plot(np.tile(tonumpy(adc_mask),NRep).flatten('F'),'.',label='ADC')
#plt.plot(tonumpy(event_time).flatten('F'),'.',label='time')
#plt.plot(tonumpy(rf_event[:,:,0]).flatten('F'),label='RF')
#plt.legend()
#plt.subplot(312); plt.title('seq: gradients')
#plt.plot(tonumpy(gradm_event[:,:,0]).flatten('F'),label='gx')
#plt.plot(tonumpy(gradm_event[:,:,1]).flatten('F'),label='gy')
#plt.legend()
#plt.subplot(313); plt.title('signal')
#plt.plot(tonumpy(scanner.signal[0,:,:,0,0]).flatten('F'),label='real')
#plt.plot(tonumpy(scanner.signal[0,:,:,1,0]).flatten('F'),label='imag')
#plt.legend()
#plt.show()

                        
#%%  FITTING BLOCK
#t=np.cumsum(tonumpy(event_time).transpose().ravel())
#y=tonumpy(scanner.signal[0,:,:,0,0]).transpose().ravel()
#t=t[5:-2]; y=y[5:-2]
#def fit_func(t, a, R,c):
#    return a*np.exp(-R*t) + c   
#
#p=scipy.optimize.curve_fit(fit_func,t,y,p0=(np.mean(y), 1,np.min(y)))
#print(p[0][1])
#
#fig=plt.figure("""fit""")
#ax1=plt.subplot(131)
#ax=plt.plot(t,y,label='data')
#plt.plot(t,fit_func(t,p[0][0],p[0][1],p[0][2]),label="f={:.2}*exp(-{:.2}*t)+{:.2}".format(p[0][0], p[0][1],p[0][2]))
#plt.title('fit')
#plt.legend()
#plt.ion()
#
#fig.set_size_inches(64, 7)
#plt.show()
            