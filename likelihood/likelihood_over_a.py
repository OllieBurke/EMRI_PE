import cupy as cp
import numpy as np
import os 
import sys
# sys.path.append("../")
from EMRI_settings import (M, mu, a, p0, e0, Y0, 
                      dist, Phi_phi0, Phi_theta0, Phi_r0, qS, phiS, qK, phiK, 
                      mich, T, inspiral_kwargs, sum_kwargs, amplitude_kwargs,xp, use_gpu, delta_t
                      ) 



from scipy.signal import tukey       # I'm always pro windowing.  

from lisatools.sensitivity import noisepsd_AE,noisepsd_T # Power spectral densities
from fastlisaresponse import ResponseWrapper             # Response

# Import relevant EMRI packages
from few.waveform import GenerateEMRIWaveform
from few.trajectory.inspiral import EMRIInspiral
from few.utils.utility import get_separatrix, get_p_at_t

# Import features from eryn
from eryn.ensemble import EnsembleSampler
from eryn.moves import StretchMove
from eryn.prior import ProbDistContainer, uniform_dist
from eryn.backends import HDFBackend

YRSID_SI = 31558149.763545603

np.random.seed(1234)

# Set up response parameters
t0 = 20000.0   # How many samples to remove from start and end of simulations.
order = 25

#TODO: Need to figure out how to make this NOT hard coded.
orbit_file_esa = "/home/ad/burkeol/work/Github_repositories/lisa-on-gpu/orbit_files/equalarmlength-trailing-fit.h5"
orbit_kwargs_esa = dict(orbit_file=orbit_file_esa)

# 1st or 2nd or custom (see docs for custom)
tdi_gen = "1st generation"

index_lambda = 8
index_beta = 7

tdi_kwargs_esa = dict(
    orbit_kwargs=orbit_kwargs_esa, order=order, tdi=tdi_gen, tdi_chan="AET",
    )

TDI_channels = ['TDIA','TDIE']
N_channels = len(TDI_channels)

def zero_pad(data):
    """
    Inputs: data stream of length N
    Returns: zero_padded data stream of new length 2^{J} for J \in \mathbb{N}
    """
    N = len(data)
    pow_2 = xp.ceil(np.log2(N))
    return xp.pad(data,(0,int((2**pow_2)-N)),'constant')

def inner_prod(sig1_f,sig2_f,N_t,delta_t,PSD):
    """
    Compute stationary noise-weighted inner product
    Inputs: sig1_f and sig2_f are signals in frequency domain 
            N_t length of padded signal in time domain
            delta_t sampling interval
            PSD Power spectral density

    Returns: Noise weighted inner product 
    """
    prefac = 4*delta_t / N_t
    sig2_f_conj = xp.conjugate(sig2_f)
    return prefac * xp.real(xp.sum((sig1_f * sig2_f_conj)/PSD))

##======================Likelihood and Posterior (change this)=====================

def llike(params):
    """
    Inputs: Parameters to sample over
    Outputs: log-whittle likelihood
    """
    # Intrinsic Parameters
    M_val = float(params[0])
    mu_val =  float(params[1])
    a_val =  float(params[2])            
    p0_val = float(params[3])
    e0_val = float(params[4])
    Y0_val = float(params[5])
    
    # Luminosity distance 
    D_val = float(params[6])

    # Angular Parameters
    qS_val = float(params[7])
    phiS_val = float(params[8])
    qK_val = float(params[9])
    phiK_val = float(params[10])

    # Angular parameters
    Phi_phi0_val = float(params[11])
    Phi_theta0_val = float(params[12])
    Phi_r0_val = float(params[13])

    # Propose new waveform model
    waveform_prop = EMRI_TDI(M_val, mu_val, a_val, p0_val, e0_val, 
                                  Y0_val, D_val, qS_val, phiS_val, qK_val, phiK_val,
                                    Phi_phi0=Phi_phi0_val, Phi_theta0=Phi_theta0_val, Phi_r0=Phi_r0_val, 
                                    mich=mich, dt=delta_t, T=T)  # EMRI waveform across A, E and T.


    # Taper and then zero pad. 
    EMRI_AET_w_pad_prop = [zero_pad(window*waveform_prop[i]) for i in range(N_channels)]

    # Compute in frequency domain
    EMRI_AET_fft_prop = [xp.fft.rfft(item) for item in EMRI_AET_w_pad_prop]

    # Compute (d - h| d- h)
    diff_f_AET = [data_f_AET[k] - EMRI_AET_fft_prop[k] for k in range(N_channels)]
    inn_prod = xp.asarray([inner_prod(diff_f_AET[k],diff_f_AET[k],N_t,delta_t,PSD_AET[k]) for k in range(N_channels)])
    
    # Return log-likelihood value as numpy val. 
    llike_val_np = xp.asnumpy(-0.5 * (xp.sum(inn_prod))) 
    return (llike_val_np)

## ===================== CHECK TRAJECTORY ====================
# 
# traj = EMRIInspiral(func="pn5")  # Set up trajectory module, pn5 AAK
traj = EMRIInspiral(func="KerrEccentricEquatorial")  # Set up trajectory module, pn5 AAK

t_traj, p_traj, e_traj, Y_traj, Phi_phi_traj, Phi_r_traj, Phi_theta_traj = traj(M, mu, a, p0, e0, Y0,
                                             Phi_phi0=Phi_phi0, Phi_theta0=Phi_theta0, Phi_r0=Phi_r0, T=T)


traj_args = [M, mu, a, e_traj[0], Y_traj[0]]
index_of_p = 3

# Check to see what value of semi-latus rectum is required to build inspiral lasting T years. 
p_new = get_p_at_t(
    traj,
    T,
    traj_args,
    index_of_p=3,
    index_of_a=2,
    index_of_e=4,
    index_of_x=5,
    xtol=2e-12,
    rtol=8.881784197001252e-16,
    bounds=[None, 12],
)


print("We require initial semi-latus rectum of ",p_new, "for inspiral lasting", T, "years")
print("Your chosen semi-latus rectum is", p0)
if p0 < p_new:
    print("Careful, the smaller body is plunging. Expect instabilities.")
else:
    print("Body is not plunging.") 
print("Final point in semilatus rectum achieved is", p_traj[-1])
print("Separatrix : ", get_separatrix(a, e_traj[-1], Y_traj[-1]))


model_choice = "Pn5AAKWaveform"
model_choice = "FastSchwarzschildEccentricFlux"
model_choice = "KerrEccentricEquatorialFlux"

import time
print("Now going to load in class")
start = time.time()
Waveform_model = GenerateEMRIWaveform(model_choice, inspiral_kwargs=inspiral_kwargs, sum_kwargs=sum_kwargs, amplitude_kwargs=amplitude_kwargs, use_gpu=use_gpu)
# Waveform_model = GenerateEMRIWaveform(model_choice, inspiral_kwargs=inspiral_kwargs, sum_kwargs=sum_kwargs, use_gpu=use_gpu)
end = time.time() - start

print("It took",end," seconds to load in the waveform")
####=======================True Responsed waveform==========================
# Build the response wrapper
EMRI_TDI = ResponseWrapper(Waveform_model,T,delta_t,
                          index_lambda,index_beta,t0=t0,
                          flip_hx = True, use_gpu = use_gpu, is_ecliptic_latitude=False,
                          remove_garbage = "zero", **tdi_kwargs_esa)


# Set true params
params = [M,mu,a,p0,e0,Y0,dist,qS, phiS, qK, phiK] 
waveform = EMRI_TDI(*params, Phi_phi0 = Phi_phi0, Phi_theta0 = Phi_theta0, Phi_r0 = Phi_r0)  # Generate h_plus and h_cross

# Window to reduce leakage. 
window = cp.asarray(tukey(len(waveform[0]),0.05))
# Taper and then zero_pad signal
EMRI_AET_w_pad = [zero_pad(window*waveform[i]) for i in range(N_channels)]
N_t = len(EMRI_AET_w_pad[0])

# Compute signal in frequency domain
EMRI_AET_fft = xp.asarray([xp.fft.rfft(waveform) for waveform in EMRI_AET_w_pad])
freq = xp.fft.rfftfreq(N_t,delta_t)
freq[0] = freq[1]   # To "retain" the zeroth frequency

# Define PSDs
freq_np = xp.asnumpy(freq)

PSD_AET = [noisepsd_AE(freq_np, includewd=T),noisepsd_AE(freq_np,includewd=T),noisepsd_T(freq_np,includewd=T)]
PSD_AET = [cp.asarray(item) for item in PSD_AET] # Convert to cupy array

# Compute optimal matched filtering SNR
SNR2_AET = xp.asarray([inner_prod(EMRI_AET_fft[i],EMRI_AET_fft[i],N_t,delta_t,PSD_AET[i]) for i in range(N_channels)])

for i in range(N_channels):
    print("SNR in channel {0} is {1}".format(TDI_channels[i],SNR2_AET[i]**(1/2)))

SNR = xp.asnumpy(xp.sum(SNR2_AET)**(1/2))
print("Final SNR = ",SNR)

if SNR < 15:
    print("Warning: The SNR of the source is low. Be careful during")
    print("your parameter estimation run. ")
##=====================Noise Setting: Currently 0=====================

# Compute Variance and build noise realisation
variance_noise_AET = [N_t * PSD_AET[k] / (4*delta_t) for k in range(N_channels)]
noise_f_AET_real = [xp.random.normal(0,np.sqrt(variance_noise_AET[k])) for k in range(N_channels)]
noise_f_AET_imag = [xp.random.normal(0,np.sqrt(variance_noise_AET[k])) for k in range(N_channels)]

# Compute noise in frequency domain
noise_f_AET = xp.asarray([noise_f_AET_real[k] + 1j * noise_f_AET_imag[k] for k in range(N_channels)])

# Dealing with positive transform, so first and last values are real. 
# todo: fix
#noise_f_AET[0] = np.sqrt(2)*np.real(noise_f_AET) 
#noise_f_AET[-1] = np.sqrt(2)*np.real(noise_f_AET)

data_f_AET = EMRI_AET_fft + 0*noise_f_AET   # define the data

##===========================Likelihood -- Plot stuff!============================

# Calculate FM on a
delta_a = 1e-8

params_p = [M,mu,a + delta_a,p0,e0,Y0,dist,qS, phiS, qK, phiK] 
waveform_p = EMRI_TDI(*params_p, Phi_phi0 = Phi_phi0, Phi_theta0 = Phi_theta0, Phi_r0 = Phi_r0)  # Generate h_plus and h_cross

params_m = [M,mu,-(a - delta_a),p0,e0,-1*Y0,dist,qS, phiS, qK, phiK] 
waveform_m = EMRI_TDI(*params_m, Phi_phi0 = Phi_phi0, Phi_theta0 = Phi_theta0, Phi_r0 = Phi_r0)  # Generate h_plus and h_cross

deriv_waveform = [(waveform_p[k] - waveform_m[k])/(2*delta_a) for k in range(N_channels)] 
deriv_waveform_pad = [zero_pad(window*deriv_waveform[i]) for i in range(N_channels)]

deriv_waveform_fft = xp.asarray([xp.fft.rfft(waveform) for waveform in deriv_waveform_pad])

gamma_aa = xp.sum(xp.asarray([inner_prod(deriv_waveform_fft[k],deriv_waveform_fft[k],N_t,delta_t,PSD_AET[k]) for k in range(N_channels)]))

delta_a = xp.asnumpy(gamma_aa**(-1/2)) # Precision measurement


print("Now creating range of a")
a_range = np.arange(a - 3*delta_a, a +3*delta_a, delta_a/30)
a_range[0] = a
a_range = np.sort(a_range, kind = 'quicksort')

llike_vec = []
from tqdm import tqdm as tqdm
for spin_val in tqdm(a_range):
    if spin_val < 0:
        params =[M,mu,-1*spin_val,p0,e0,-1*Y0,dist,qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0]
    else:
        params =[M,mu,spin_val,p0,e0,Y0,dist,qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0]
    # params =[M,mu,spin_val,p0,e0,Y0,dist,qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0]
    llike_val = llike(params)
    llike_vec.append(llike_val)

import matplotlib.pyplot as plt
llike_array = np.array(llike_vec)

plt.plot(a_range, np.exp(llike_vec), '*', label = "Likelihood")
plt.plot(a_range, np.exp(llike_vec), c = 'blue', alpha = 0.5)
plt.axvline(x = a + delta_a, c = 'black', ls = 'dashed', label = "Dirty FM approx")
plt.axvline(x = a - delta_a, c = 'black', ls = 'dashed') 
plt.legend(fontsize = 12)

plt.xlabel(r'Spin values', fontsize = 18)
plt.ylabel(r'Likelihood', fontsize = 18)
plt.title(r'Likelihood over spin -- a = 0.0', fontsize = 18)
plt.savefig("plots/likelihood_spin_a_neg_0p0.pdf",bbox_inches="tight")
plt.clf()






# Quick checks
