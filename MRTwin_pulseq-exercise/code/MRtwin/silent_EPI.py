# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 22:10:52 2021

@author: Nan Lan
"""

excercise = """
This is  Sinusoidal ZIGZAG EPI
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
import scipy.interpolate
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"

from importlib import reload
reload(core.scanner)

double_precision = False
do_scanner_query = False

use_gpu = 1
gpu_dev = 0

if sys.platform != 'linux':
    use_gpu = 0
    gpu_dev = 0
  
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
sz = np.array([24,24])                      # image size
extraMeas = 1                               # number of measurmenets/ separate scans
NRep = extraMeas*sz[1]                      # number of total repetitions
szread=sz[1]
NEvnt = szread + 5 + 2                               # number of events F/R/P
NSpins = 16**2                               # number of spin sims in each voxel
NCoils = 1                                  # number of receive coil elements
noise_std = 0*100*1e-3                        # additive Gaussian noise std
kill_transverse = False                     #
import time; today_datestr = time.strftime('%y%m%d')
NVox = sz[0]*szread

#############################################################################
## S1: Init spin system and phantom::: #####################################
# initialize scanned object
spins = core.spins.SpinSystem(sz,NVox,NSpins,use_gpu+gpu_dev,double_precision=double_precision)

# either (i) load phantom (third dimension: PD, T1 T2 dB0 rB1)
phantom = spins.get_phantom(sz[0],sz[1],type='object1')  # type='object1' or 'brain1'


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
rf_event[3,0,0] = 90*np.pi/180  # 90deg excitation now for every rep
rf_event[3,0,3]=1
rf_event = setdevice(rf_event)
scanner.init_flip_tensor_holder()    
scanner.set_flip_tensor_withB1plus(rf_event)
# rotate ADC according to excitation phase
rfsign = ((rf_event[3,:,0]) < 0).float()
scanner.set_ADC_rot_tensor(-rf_event[3,:,1]+ np.pi/2 + np.pi*rfsign) #GRE/FID specific

# event timing vector 
event_time = torch.from_numpy(0.008*1e-3*np.ones((NEvnt,NRep))).float()
event_time = setdevice(event_time)

# gradient-driver precession
gradm_event = torch.zeros((NEvnt,NRep,2), dtype=torch.float32)

case = 1 #1 is Sinusoidal EPI; 0 is normal EPI
PE_type = "constant" #blip / constant / ZAP:Zigzag_Aligned_Projections

if case==1:#Sinusoidal EPI
    # Midpoint correction
    gradm_event[4,0,0] = -0.5*szread-1/2    # first rewinder in read  1 comes from mid_step_PE
    gradm_event[4,0,1] =  -0.5*NRep-1.5730/2     # first rewinder in phase  1.5730 comes from max_step_ro
 
    # for scaling the magnitude of sin function
    sum = 0   
    for i in range(szread):
        step = torch.tensor(np.sin(i/szread*np.pi))       
        sum+=step
    
    #readout direction
    max_step_ro = 0
    for i in range(1,szread*NRep):
        idx_Evnt = i%szread
        idx_Rpep = i//szread
        radio_ro = szread/sum  #scaling factor
        step_ro = torch.tensor(radio_ro*np.sin(i/szread*np.pi))
        gradm_event[5+idx_Evnt:-2,idx_Rpep,1] = step_ro      #adc for readout     
        # print(step_ro)
        
        #midpoint correction in readoout direction
        if(max_step_ro < step_ro): 
            max_step_ro = step_ro
            
    print("max_step_ro is",max_step_ro )       
   
    # phase direction
    if PE_type is "blip":  
        gradm_event[4,1:,0] = 1  #phase blip
        
    elif PE_type is "constant": 
        gradm_event[5:-2,1:,0] = 1/szread  #constant PE
        
    elif PE_type is "ZAP":  
        mid_step_PE = 0
        for i in range(szread*NRep):
            idx_Evnt = i%szread
            idx_Rpep = i//szread
            radio_PE = szread/sum/NRep #scaling factor
            step_PE = torch.tensor(np.abs(radio_PE*np.sin(i/szread*np.pi)))
            gradm_event[5+idx_Evnt:-2,idx_Rpep,0] = step_PE      #adc for readout
            # print(step_PE)
            
            #midpoint correction in phase direction
            if (i>(NRep//2-1)*szread and i<(NRep//2)*szread): 
                mid_step_PE += step_PE
            
        print("mid_step_PE is",mid_step_PE )  


elif case==0: # constant     
    gradm_event[4,0,0] = -0.5*szread    # first rewinder in read  
    gradm_event[4,0,1] =  -0.5*NRep     # first rewinder in phase  1.5730 comes from max_step_ro
    gradm_event[4,1:,0] = 1             #phase blibs
    gradm_event[5:-2,::2,1] = 1.0       #adc  for even lines
    gradm_event[5:-2,1::2,1] = -1.0     #adc for odd lines
    
    gradm_event[4,1::2,1] = -1.0        #adjust adc start before odd lines
    gradm_event[4,2::2,1] = +1.0        #adjust adc start before even lines

gradm_event = setdevice(gradm_event)

scanner.init_gradient_tensor_holder()
scanner.set_gradient_precession_tensor_super(gradm_event,rf_event)  # refocusing=False for GRE/FID, adjust for higher echoes
## end S3: MR sequence definition ::: #####################################


#############################################################################
## S4: MR simulation forward process ::: #####################################
scanner.init_signal()
scanner.forward_fast(spins, event_time)

# sequence and signal plotting
targetSeq = core.target_seq_holder.TargetSequenceHolder(rf_event,event_time,gradm_event,scanner,spins,scanner.signal)
targetSeq.print_seq_pic(True,plotsize=[12,9])

targetSeq.print_seq(plotsize=[12,9],time_axis=1)   
  
#%% ############################################################################
## S5: MR reconstruction of signal ::: #####################################
spectrum = tonumpy(scanner.signal[0,adc_mask.flatten()!=0,:,:2,0].clone()) 
spectrum = spectrum[:,:,0]+spectrum[:,:,1]*1j # get all ADC signals as complex numpy array
spectrum_adc= spectrum
kspace= spectrum
space = np.zeros_like(spectrum)

plt.subplot(413); plt.ylabel('signal')
plt.plot(np.real(spectrum).flatten('F'),label='real')
plt.plot(spectrum.imag.flatten('F'),label='imag')

# NUFFT
adc_idx = np.where(scanner.adc_mask.cpu().numpy())[0]        
grid = scanner.kspace_loc[adc_idx,:,:]
NCol=adc_idx.size

X, Y = np.meshgrid(np.linspace(0,NCol-1,NCol) - NCol / 2, np.linspace(0,NRep-1,NRep) - NRep/2)
grid = np.double(grid.detach().cpu().numpy())
grid[np.abs(grid) < 1e-5] = 0

plt.subplot(336); plt.plot(grid[:,:,0].ravel('F'),grid[:,:,1].ravel('F'),'rx',markersize=3);  plt.plot(X,Y,'k.',markersize=2);
plt.show()

spectrum_resampled_x = scipy.interpolate.griddata((grid[:,:,0].ravel(), grid[:,:,1].ravel()), np.real(kspace[:,:]).ravel(), (X, Y), method='cubic')
spectrum_resampled_y = scipy.interpolate.griddata((grid[:,:,0].ravel(), grid[:,:,1].ravel()), np.imag(kspace[:,:]).ravel(), (X, Y), method='cubic')

kspace=spectrum_resampled_x+1j*spectrum_resampled_y
kspace[np.isnan(kspace)] = 0

kspace = np.roll(kspace,NCol//2,axis=0)
kspace = np.roll(kspace,NRep//2,axis=1)
        
space = np.fft.ifft2(kspace)  # for kspace,  

space = np.roll(space,szread//2-1,axis=0)
space = np.roll(space,NRep//2-1,axis=1)
space = np.flip(space,(0,1))


targetSeq.print_seq(plotsize=[12,9])
      
plt.subplot(4,6,19)
plt.imshow(phantom[:,:,0].transpose(), interpolation='none'); plt.xlabel('PD')
plt.subplot(4,6,20)
plt.imshow(phantom[:,:,3].transpose(), interpolation='none'); plt.xlabel('dB0')
plt.subplot(4,6,21)
plt.imshow(np.abs(spectrum_adc).transpose(), interpolation='none'); plt.xlabel('spectrum')
plt.subplot(4,6,22)
plt.imshow(np.abs(kspace).transpose(), interpolation='none'); plt.xlabel('kspace')
plt.subplot(4,6,23)
plt.imshow(np.abs(space).transpose(), interpolation='none'); plt.xlabel('mag_img')
plt.subplot(4,6,24)
plt.imshow(np.angle(space).transpose(), interpolation='none'); plt.xlabel('phase_img')
plt.show()                      
