"""
Created on Tue Jan 29 14:38:26 2019
@author: mzaiss

"""
experiment_id = 'exZ01_'
sequence_class = "FLASH_T1 Mapping"
experiment_description = """
"""
excercise = """
Z01.1. make a spin echo EPI
Z01.2. make two readouts from one measurement: eg. FID/STE, or STE and SE separatley as two seoparate contrasts
Z01.3. diffusion
Z01.4. b0 mapping  (phase maps)
Z01.5. b1 mapping  (cosine fit)
Z01.6. t1 mapping  (invrec) : scatter plot plt.plot(img[:],img2[:]) plot of the prediction,.... quantitative data
Z01.7. t2 weighting/mapping (TE or t2prep)

Z02.1. Display all Spins as a vector or many vectors
Z02.2. Display all Spins as a vector or many vectors, and animate this during a sequence


"""
"""
Created on Tue Jan 29 14:38:26 2019
@author: mzaiss

"""

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
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"
from importlib import reload
reload(core.scanner)

double_precision = False
do_scanner_query = False

use_gpu = 0
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
#%%
#############################################################################
## S0: define image and simulation settings::: #####################################
sz = np.array([64,64])                      # image size
extraMeas = 1                               # number of measurmenets/ separate scans
NRep = extraMeas*sz[1]                      # number of total repetitions
szread=sz[1]
NEvnt = szread + 5 + 2                               # number of events F/R/P
NSpins = 8**2                               # number of spin sims in each voxel
NCoils = 1                                  # number of receive coil elements
noise_std = 0*100*1e-3                        # additive Gaussian noise std
kill_transverse = True                     #
import time; today_datestr = time.strftime('%y%m%d')
NVox = sz[0]*szread

#############################################################################
## S1: Init spin system and phantom::: #####################################
# initialize scanned object
spins = core.spins.SpinSystem(sz,NVox,NSpins,use_gpu+gpu_dev,double_precision=double_precision)

# either (i) load phantom (third dimension: PD, T1 T2 dB0 rB1)
phantom = spins.get_phantom(sz[0],sz[1],type='brain1')  # type='object1' or 'brain1'

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
#%%& store 8 images 
T1images = np.zeros([sz[0],sz[1],16])
nTI =0
diff_TI =np.linspace(0.1,5,16)
for TI in diff_TI:
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
    rf_event[0,0,0] = 180*np.pi/180  # inversion recovery pulse for every rep
    rf_event[3,:,0] = 15*np.pi/180  # 90deg excitation now for every rep
    rf_event[3,:,3] =1
    rf_event[3,:,1]=torch.arange(0,50*NRep,50)*np.pi/180 
    
    def get_phase_cycler(n, dphi,flag=0):
        out = np.cumsum(np.arange(n) * dphi)  #  from Alex (standard)
        if flag:
            for j in range(0,n,1):               # from Zur et al (1991)
                out[j] = dphi/2*(j**2 +j+2)
        out = torch.from_numpy(np.mod(out, 360).astype(np.float32))
        return out    
    
    rf_event[3,:,1]=get_phase_cycler(NRep,117)*np.pi/180 
    
    
    rf_event = setdevice(rf_event)
    scanner.init_flip_tensor_holder()    
    scanner.set_flip_tensor_withB1plus(rf_event)
    # rotate ADC according to excitation phase
    rfsign = ((rf_event[3,:,0]) < 0).float()
    scanner.set_ADC_rot_tensor(-rf_event[3,:,1] + np.pi/2 + np.pi*rfsign) #GRE/FID specific

# # store 8 images 
# T1images = np.zeros([32,32,8])
# nTI =0
# event timing vector 
    event_time = torch.from_numpy(0.08*1e-3*np.ones((NEvnt,NRep))).float()


    event_time [1,0] = TI #&
    #event_time [3,:] = 0.003
    event_time = setdevice(event_time)

# gradient-driver precession
# Cartesian encoding
    gradm_event = torch.zeros((NEvnt,NRep,2), dtype=torch.float32)
    gradm_event[4,:,1] = -0.5*szread
    gradm_event[5:-2,:,1] = 1.0
    gradm_event[4,:,0] = torch.arange(0,NRep,1)-NRep/2 #phase blib
    gradm_event[-2,:,0] = -gradm_event[4,:,0]  # phase backblip
    gradm_event[-2,:,1] = 1.5*szread         # spoiler (even numbers sometimes give stripes, best is ~ 1.5 kspaces, for some reason 0.2 works well,too  )
    gradm_event = setdevice(gradm_event)

# centric 
    permvec= np.zeros((NRep,),dtype=int) 
    permvec[0] = 0
    for i in range(1,int(NRep/2)+1):
        permvec[i*2-1] = (-i)
        if i < NRep/2:
            permvec[i*2] = i
    permvec=permvec+NRep//2     # centric out reordering
    
    #permvec=np.arange(0,NRep,1)  # this eleiminates the permutation again
    #permvec=np.arange(NRep-1,-1,-1)  # inverse linear reordering
    #permvec=np.random.permutation(NRep) # inverse linear reordering
    
    gradm_event[4,:,0]=gradm_event[4,permvec,0]
    gradm_event[-2,:,0] = -gradm_event[4,:,0]  # phase backblip
    
    scanner.init_gradient_tensor_holder()
    scanner.set_gradient_precession_tensor_super(gradm_event,rf_event)  # refocusing=False for GRE/FID, adjust for higher echoes
    ## end S3: MR sequence definition ::: #####################################



#%%############################################################################
## S4: MR simulation forward process ::: #####################################
    scanner.init_signal()
    scanner.forward_fast(spins, event_time)
    
    # sequence and signal plotting
    # targetSeq = core.target_seq_holder.TargetSequenceHolder(rf_event,event_time,gradm_event,scanner,spins,scanner.signal)
    # targetSeq.print_seq_pic(True,plotsize=[12,9])
    # targetSeq.print_seq(plotsize=[12,9], time_axis=0)
      

    print(TI)
#%% ############################################################################
## S5: MR reconstruction of signal ::: #####################################

    spectrum = tonumpy(scanner.signal[0,adc_mask.flatten()!=0,:,:2,0].clone()) 
    spectrum = spectrum[:,:,0]+spectrum[:,:,1]*1j # get all ADC signals as complex numpy array
    inverse_perm = np.arange(len(permvec))[np.argsort(permvec)]
    spectrum=spectrum[:,inverse_perm]
    #spectrum[:,permvec]=spectrum
    kspace=spectrum
    spectrum = np.roll(spectrum,szread//2,axis=0)
    spectrum = np.roll(spectrum,NRep//2,axis=1)
    space = np.zeros_like(spectrum)
    space = np.fft.ifft2(spectrum)
    space = np.roll(space,szread//2-1,axis=0)
    space = np.roll(space,NRep//2-1,axis=1)
    space = np.flip(space,(0,1))
    T1images[:,:,nTI] = np.abs(space) #& store magnitude of images 
    nTI += 1
    #%%
# fig2 =plt.figure(3)
# plt.subplot(4,6,19)
# plt.imshow(phantom[:,:,0].transpose(), interpolation='none'); plt.xlabel('PD')
# plt.subplot(4,6,20)
# plt.imshow(phantom[:,:,3].transpose(), interpolation='none'); plt.xlabel('dB0')
    
# plt.subplot(4,6,22)
# plt.imshow(np.abs(kspace).transpose(), interpolation='none'); plt.xlabel('kspace')
# plt.subplot(4,6,23)
# plt.imshow(np.abs(space).transpose(), interpolation='none',aspect = sz[0]/szread); plt.xlabel('mag_img')
# plt.subplot(4,6,24)
# mask=(np.abs(space)>0.2*np.max(np.abs(space))).transpose()
# plt.imshow(np.angle(space).transpose()*mask, interpolation='none',aspect = sz[0]/szread); plt.xlabel('phase_img')
# plt.show()
   

#%%
fig2 =plt.figure(2)
plt.subplot(1,8,1)
plt.imshow(np.abs(T1images[:,:,0]).transpose(), interpolation='none',aspect = sz[0]/szread,clim=(0,0.7)); plt.xlabel('mag_img'); 
plt.subplot(1,8,2)
plt.imshow(np.abs(T1images[:,:,1]).transpose(), interpolation='none',aspect = sz[0]/szread,clim=(0,0.7)); plt.xlabel('mag_img');
plt.subplot(1,8,3)
plt.imshow(np.abs(T1images[:,:,2]).transpose(), interpolation='none',aspect = sz[0]/szread,clim=(0,0.7)); plt.xlabel('mag_img')
plt.subplot(1,8,4)
plt.imshow(np.abs(T1images[:,:,3]).transpose(), interpolation='none',aspect = sz[0]/szread,clim=(0,0.7)); plt.xlabel('mag_img')
plt.subplot(1,8,5)
plt.imshow(np.abs(T1images[:,:,4]).transpose(), interpolation='none',aspect = sz[0]/szread,clim=(0,0.7)); plt.xlabel('mag_img')
plt.subplot(1,8,6)
plt.imshow(np.abs(T1images[:,:,5]).transpose(), interpolation='none',aspect = sz[0]/szread,clim=(0,0.7)); plt.xlabel('mag_img')
plt.subplot(1,8,7)
plt.imshow(np.abs(T1images[:,:,6]).transpose(), interpolation='none',aspect = sz[0]/szread,clim=(0,0.7)); plt.xlabel('mag_img')
plt.subplot(1,8,8)
plt.imshow(np.abs(T1images[:,:,7]).transpose(), interpolation='none',aspect = sz[0]/szread,clim=(0,0.7)); plt.xlabel('mag_img')
plt.show()  
#%%
## Fitting Block run in every pixel ,image->loop->pixel
def fit_func(diff_TI, a, t1):
    return np.abs(a*(1-2*np.exp(-diff_TI/t1))) ##
T1map = np.zeros([sz[0],sz[1]])
for x in range(T1images.shape[0]):         
    for y in range(T1images.shape[1]):
        if T1images[x,y,0]>0.03:
            p =scipy.optimize.curve_fit(fit_func,diff_TI,T1images[x,y,:])
            T1map[x][y] = p[0][1]
       
fig3= plt.figure(3)  
plt.subplot(1,3,1)
plt.title('T1 Mapping')
plt.imshow(T1map, interpolation='none',aspect = sz[0]/szread,clim=(0,2.5)); plt.xlabel('mag_img'); plt.colorbar()
plt.subplot(1,3,2)
plt.title('Original T1')
plt.imshow(phantom[:,:,1],interpolation='none',aspect = sz[0]/szread,clim=(0,2.5));plt.colorbar()
plt.subplot(1,3,3)
plt.title('Difference')
plt.imshow(abs(phantom[:,:,1]-T1map),interpolation='none',aspect = sz[0]/szread,clim=(0,2.5));plt.colorbar()

    # plt.figure("""phantom"""); plt.clf();  param=['PD','T1 [s]','T2 [s]','dB0 [Hz]','rB1 [rel.]']
    # for i in range(5):
    #     plt.subplot(151+i), plt.title(param[i])
    #     ax=plt.imshow(phantom[:,:,i], interpolation='none')
    #     fig = plt.gcf(); fig.colorbar(ax) 
    # fig.set_size_inches(18, 3); plt.show()

#%%
## Fitting Block run in every pixel ,image->loop->pixel

            