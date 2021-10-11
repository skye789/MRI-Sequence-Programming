"""
Created on Tue Jan 29 14:38:26 2019
@author: mzaiss

"""
experiment_id = 'exD02_bSSFP'
sequence_class = "super"
experiment_description = """
2 D imaging
"""
excercise = """
GRE会有artifact，因为有old transversal magnitisation,
balance的意思是不用spoil gradient而是用相反的phase encoding加在每个Repetition最后来抵消old transversal magnitisation
10°产生FID,设置TE很短，再施加20°产生FID和spinecho,等到spinecho到最大值时再重复施加20°.Refocusing发生在10°
This starts from A09 which was the fully relaxed GRE sequence. 
flip angle was set to 5 degree
D02.1. As before let us decrease the recovery time. This time make it very short event_time[-1,:] =  0.002
		You should observe an image with artifacts when going from event_time[-1,:] =  5 to 0.002. Last time we tried to get rid of higher echoes. This time we want to understand them better.
				
D02.2. As shown in exD01 the echoes are at the same time point. But they have a different encoding as their transverse magnetization saw different gradients.
        This can actually be observed in the k-space plot: do you see the additional intensity at the egde of the k-space?
        
D02.3.  In the second repetition, the FID or gre signal starts at k=0, then the revinder and readout is applied.
        At which k-space location does the spin echo start?
        How can you realize that also the spin echo starts at k=0 at the beginning of the second repetition?

D02.5 You might still see some artifacts, especially in the phase. This is because the spin echo and the FID will have a different phase. 
        To correct this you must alter the rf phase in every cycle. This code can be helpful
        alternate= torch.tensor([0,1])
        alternate.repeat(NRep//2)

D02.6. Now you have a balance ssfp sequence!  
        If you switch of the gradients again, you will see that it osclillates in the beginning:
            This can be solved using  a preppulse, a so called alpha/2 pulse
            rf_event[2,0,0] = 2.5*np.pi/180  # 90deg excitation now for every rep
            rf_event[2,0,1] = 180*np.pi/180  # 90deg excitation now for every rep
            
            with the correct timing

            event_time[2,0] =  torch.sum(event_time[2:,0])*0.5 
        
D02.4  Is there any stimulated echo?

D02.5  If all timing and prep is correct, you can also try centric reordering.

D02.6  Try to reduce number of spins. Why can you decrease this now?
"""
#%%
#matplotlib.pyplot.close(fig=None)
#%%
import os, sys
import numpy as np
import scipy
import scipy.io
from  scipy import ndimage
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

from importlib import reload
reload(core.scanner)

double_precision = False
do_scanner_query = False

use_gpu = 1
gpu_dev = 0

if sys.platform != 'linux':
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

def phaseimg(x):
    return np.angle(1j*x[:,:,1]+x[:,:,0])

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
sz = np.array([32,32])                      # image size
extraMeas = 1                               # number of measurmenets/ separate scans
NRep = extraMeas*sz[1]                   # number of total repetitions
szread=sz[1]
NEvnt = szread + 5 + 2                               # number of events F/R/P
NSpins = 24**2                               # number of spin sims in each voxel
NCoils = 1                                  # number of receive coil elements
noise_std = 0*1e-3                          # additive Gaussian noise std
kill_transverse = True                     # kills transverse when above 1.5 k.-spaces
import time; today_datestr = time.strftime('%y%m%d')
NVox = sz[0]*szread

#############################################################################
## S1: Init spin system and phantom::: #####################################
# initialize scanned object
spins = core.spins.SpinSystem(sz,NVox,NSpins,use_gpu+gpu_dev,double_precision=double_precision)

# either (i) load phantom (third dimension: PD, T1 T2 dB0 rB1)
phantom = spins.get_phantom(sz[0],sz[1],type='object1')  # type='object1' or 'brain1'

# or (ii) set phantom  manually to single pixel phantom
#phantom = np.zeros((sz[0],sz[1],5), dtype=np.float32); 
#phantom[1,1,:]=np.array([1, 1, 0.1, 0, 1]) # third dimension: PD, T1 T2 dB0 rB1

# adjust phantom
phantom[:,:,1] *= 1 # Tweak T1
phantom[:,:,2] *= 1 # Tweak T2
phantom[:,:,3] += 0 # Tweak dB0
phantom[:,:,4] *= 1 # Tweak rB1

if 1: # switch on for plot
    plt.figure("""phantom"""); plt.clf();  param=['PD','T1 [s]','T2 [s]','dB0 [Hz]','rB1 [rel.]']
    for i in range(5):
        plt.subplot(151+i), plt.title(param[i])
        ax=plt.imshow(phantom[:,:,i], interpolation='none')
        fig = plt.gcf(); fig.colorbar(ax) 
    fig.set_size_inches(18, 3); plt.show()

spins.set_system(phantom,R2dash=30.0)  # set phantom variables with overall constant R2' = 1/T2'  (R2*=R2+R2')

## end of S1: Init spin system and phantom ::: #####################################


#############################################################################
## S2: Init scanner system ::: #####################################
scanner = core.scanner.Scanner(sz,NVox,NSpins,NRep,NEvnt,NCoils,noise_std,use_gpu+gpu_dev,double_precision=double_precision)
#scanner.set_B1plus(phantom[:,:,4])  # use as defined in phantom
scanner.set_B1plus(1)               # overwrite with homogeneous excitation

#############################################################################
## S3: MR sequence definition ::: #####################################
# begin sequence definition
# allow for extra events (pulses, relaxation and spoiling) in the first five and last two events (after last readout event)
adc_mask = torch.from_numpy(np.ones((NEvnt,1))).float()
adc_mask[:5]  = 0
adc_mask[-2:] = 0
scanner.set_adc_mask(adc_mask=setdevice(adc_mask))

# RF events: rf_event and phases
rf_event = torch.zeros((NEvnt,NRep,4), dtype=torch.float32)
#flip angle变小，能减少banding artifact which is caused by B0 inhomogeinity
rf_event[2,0,0] = 10*np.pi/180  # 90deg excitation now for every rep
rf_event[2,0,1] = 180*np.pi/180
rf_event[3,:,0] = 20*np.pi/180  # 90deg excitation now for every rep
rf_event[3,:,3] = 1

alternate= torch.tensor([0,1])
rf_event[3,:,1]=np.pi*alternate.repeat(NRep//2)

rf_event = setdevice(rf_event)
scanner.init_flip_tensor_holder()    
scanner.set_flip_tensor_withB1plus(rf_event)
# rotate ADC according to excitation phase
rfsign = ((rf_event[3,:,0]) < 0).float()
scanner.set_ADC_rot_tensor(-rf_event[3,:,1] + np.pi/2 + np.pi*rfsign) #sequence specific

# event timing vector 
event_time = torch.from_numpy(0.08*1e-3*np.ones((NEvnt,NRep))).float()
# event_time[-1,:] =  5
event_time[-1,:] =  0.002
event_time = setdevice(event_time)
TA = tonumpy(torch.sum(event_time))
# gradient-driver precession
# Cartesian encoding
gradm_event = torch.zeros((NEvnt,NRep,2), dtype=torch.float32)
gradm_event[4,:,1] = -0.5*szread

gradm_event[5:-2,:,1] = 1
gradm_event[4,:,0] = torch.arange(0,NRep,1)-NRep/2

##要让每次Repitition后，在kspace上的位置回到原点(0,0) balance
gradm_event[-2,:,1] = -0.5*szread  # readback
gradm_event[-2,:,0] = -gradm_event[4,:,0]  #phasebackblip

if 0: # centric 
    permvec= np.zeros((NRep,),dtype=int) 
    permvec[0] = 0
    for i in range(1,int(NRep/2)+1):
        permvec[i*2-1] = (-i)
        if i < NRep/2:
            permvec[i*2] = i
    permvec=permvec+NRep//2     # centric out reordering
    gradm_event[4,:,0]=gradm_event[4,permvec,0]
    gradm_event[-2,:,0] = -gradm_event[4,:,0]  # phase backblip
else:
    permvec=np.arange(0,NRep,1)  # this eliminates the permutation again
gradm_event = setdevice(gradm_event)

scanner.init_gradient_tensor_holder()
scanner.set_gradient_precession_tensor_super(gradm_event,rf_event)  # refocusing=False for GRE/FID, adjust for higher echoes
## end S3: MR sequence definition ::: #####################################



#############################################################################
## S4: MR simulation forward process ::: #####################################
scanner.init_signal()
scanner.forward_fast(spins, event_time)

targetSeq = core.target_seq_holder.TargetSequenceHolder(rf_event,event_time,gradm_event,scanner,spins,scanner.signal)
targetSeq.print_seq_pic(True,plotsize=[12,9])
targetSeq.print_seq(plotsize=[12,9])
  
#%% ############################################################################
## S5: MR reconstruction of signal ::: #####################################

spectrum = tonumpy(scanner.signal[0,adc_mask.flatten()!=0,:,:2,0].clone()) 
spectrum = spectrum[:,:,0]+spectrum[:,:,1]*1j # get all ADC signals as complex numpy array
spectrum_adc= spectrum
inverse_perm = np.arange(len(permvec))[np.argsort(permvec)]
spectrum=spectrum[:,inverse_perm]
kspace=spectrum
space = np.zeros_like(spectrum)
spectrum = np.roll(spectrum,szread//2,axis=0)
spectrum = np.roll(spectrum,NRep//2,axis=1)
space = np.fft.ifft2(spectrum)
space = np.roll(space,szread//2-1,axis=0)
space = np.roll(space,NRep//2-1,axis=1)
space = np.flip(space,(0,1))
       
plt.subplot(4,6,19)
plt.imshow(phantom[:,:,0].transpose(), interpolation='none'); plt.xlabel('PD')
plt.subplot(4,6,20)
plt.imshow(phantom[:,:,3].transpose(), interpolation='none'); plt.xlabel('dB0')

plt.subplot(4,6,21)
plt.imshow(np.abs(spectrum_adc).transpose(), interpolation='none'); plt.xlabel('spectrum/signal')
plt.subplot(4,6,22)
plt.imshow(np.abs(kspace).transpose(), interpolation='none'); plt.xlabel('kspace')
plt.subplot(4,6,23)
plt.imshow(np.abs(space).transpose(), interpolation='none',aspect = sz[0]/szread); plt.xlabel('mag_img')
plt.subplot(4,6,24)
mask=(np.abs(space)>0.2*np.max(np.abs(space))).transpose()
plt.imshow(np.angle(space).transpose()*mask, interpolation='none',aspect = sz[0]/szread); plt.xlabel('phase_img')
plt.show()                     
