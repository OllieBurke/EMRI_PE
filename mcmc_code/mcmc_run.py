import cupy as cp
import numpy as np
import os 
import sys
sys.path.append("../")
from EMRI_settings import (M, mu, a, p0, e0, Y0, 
                      dist, Phi_phi0, Phi_theta0, Phi_r0, qS, phiS, qK, phiK, 
                      mich, T, inspiral_kwargs, sum_kwargs, xp, use_gpu, delta_t) 

from scipy.signal import tukey       # I'm always pro windowing.  

from lisatools.sensitivity import noisepsd_AE,noisepsd_T # Power spectral densities
from fastlisaresponse import ResponseWrapper             # Response

# Import relevant EMRI packages
from few.waveform import Pn5AAKWaveform
from few.trajectory.inspiral import EMRIInspiral
from few.utils.utility import get_separatrix, Y_to_xI, get_p_at_t

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
orbit_file_esa = "../Github_Repos/lisa-on-gpu/orbit_files/esa-trailing-orbits.h5"
orbit_kwargs_esa = dict(orbit_file=orbit_file_esa)

# 1st or 2nd or custom (see docs for custom)
tdi_gen = "1st generation"

index_lambda = 8
index_beta = 7

tdi_kwargs_esa = dict(
    orbit_kwargs=orbit_kwargs_esa, order=order, tdi=tdi_gen, tdi_chan="AET",
    )

TDI_channels = ['TDIA','TDIE','TDIT']
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
traj = EMRIInspiral(func="pn5")  # Set up trajectory module, pn5 AAK

# Compute trajectory 
t_traj, p_traj, e_traj, Y_traj, Phi_phi_traj, Phi_r_traj, Phi_theta_traj = traj(M, mu, a, p0, e0, Y0,
                                             Phi_phi0=Phi_phi0, Phi_theta0=Phi_theta0, Phi_r0=Phi_r0, T=T)

x_I_traj = Y_to_xI(a, p_traj, e_traj, Y_traj)

traj_args = [M, mu, a, e_traj[0], x_I_traj[0]]
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
    bounds=None,
)


print("We require initial semi-latus rectum of ",p_new, "for inspiral lasting", T, "years")
print("Your chosen semi-latus rectum is", p0)
if p0 < p_new:
    print("Careful, the smaller body is plunging. Expect instabilities.")
else:
    print("Body is not plunging.") 
print("Final point in semilatus rectum achieved is", p_traj[-1])
print("Separatrix : ", get_separatrix(a, e_traj[-1], x_I_traj[-1]))

# Construct the AAK model with 5PN trajectories
AAK_waveform_model = Pn5AAKWaveform(inspiral_kwargs=inspiral_kwargs, sum_kwargs=sum_kwargs, use_gpu=use_gpu)

####=======================True Responsed waveform==========================
# Build the response wrapper
EMRI_TDI = ResponseWrapper(AAK_waveform_model,T,delta_t,
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

##===========================MCMC Settings============================

iterations = 4000 #10000  # The number of steps to run of each walker
burnin = 0 # I always set burnin when I analyse my samples
nwalkers = 50  #50 #members of the ensemble, like number of chains

ntemps = 1             # Number of temperatures used for parallel tempering scheme.
                       # Each group of walkers (equal to nwalkers) is assigned a temperature from T = 1, ... , ntemps.

tempering_kwargs=dict(ntemps=ntemps)  # Sampler requires the number of temperatures as a dictionary

d = 1 # A parameter that can be used to dictate how close we want to start to the true parameters
# Useful check: If d = 0 and noise_f = 0, llike(*params)!!

# We start the sampler exceptionally close to the true parameters and let it run. This is reasonable 
# if and only if we are quantifying how well we can measure parameters. We are not performing a search. 

# Intrinsic Parameters
start_M = M*(1. + d * 1e-7 * np.random.randn(nwalkers,1))   
start_mu = mu*(1. + d * 1e-7 * np.random.randn(nwalkers,1))
start_a = a*(1. + d * 1e-7 * np.random.randn(nwalkers,1))
start_p0 = p0*(1. + d * 1e-8 * np.random.randn(nwalkers, 1))
start_e0 = e0*(1. + d * 1e-7 * np.random.randn(nwalkers, 1))
start_Y0 = Y0*(1. + d * 1e-7 * np.random.randn(nwalkers, 1))

# Luminosity distance
start_D = dist*(1 + d * 1e-6 * np.random.randn(nwalkers,1))

# Angular parameters
start_qS = qS*(1. + d * 1e-6 * np.random.randn(nwalkers,1))
start_phiS = phiS*(1. + d * 1e-6 * np.random.randn(nwalkers,1))
start_qK = qK*(1. + d * 1e-6 * np.random.randn(nwalkers,1))
start_phiK = phiK*(1. + d * 1e-6 * np.random.randn(nwalkers,1))

# Initial phases 
start_Phi_Phi0 = Phi_phi0*(1. + d * 1e-6 * np.random.randn(nwalkers, 1))
start_Phi_theta0 = Phi_theta0*(1. + d * 1e-6 * np.random.randn(nwalkers, 1))
start_Phi_r0 = Phi_r0*(1. + d * 1e-6 * np.random.randn(nwalkers, 1))

# Set up starting coordinates
start = np.hstack((start_M,start_mu, start_a, start_p0, start_e0, start_Y0, start_D, 
start_qS, start_phiS, start_qK, start_phiK,start_Phi_Phi0, start_Phi_theta0, start_Phi_r0))

if ntemps > 1:
    # If we decide to use parallel tempering, we fall into this if statement. We assign each *group* of walkers
    # an associated temperature. We take the original starting values and "stack" them on top of each other. 
    start = np.tile(start,(ntemps,1,1))

if np.size(start.shape) == 1:
    start = start.reshape(start.shape[-1], 1)
    ndim = 1
else:
    ndim = start.shape[-1]
# ================= SET UP PRIORS ========================

n = 25 # size of prior

Delta_theta_intrinsic = [100, 1e-3, 1e-4, 1e-4, 1e-4, 1e-4]  # M, mu, a, p0, e0 Y0
Delta_theta_D = dist/np.sqrt(np.sum(SNR))
priors_in = {
    # Intrinsic parameters
    0: uniform_dist(M - n*Delta_theta_intrinsic[0], M + n*Delta_theta_intrinsic[0]), # Primary Mass M
    1: uniform_dist(mu - n*Delta_theta_intrinsic[1], mu + n*Delta_theta_intrinsic[1]), # Secondary Mass mu
    2: uniform_dist(a - n*Delta_theta_intrinsic[2], a + n*Delta_theta_intrinsic[2]), # Spin parameter a
    3: uniform_dist(p0 - n*Delta_theta_intrinsic[3], p0 + n*Delta_theta_intrinsic[3]), # semi-latus rectum p0
    4: uniform_dist(e0 - n*Delta_theta_intrinsic[4], e0 + n*Delta_theta_intrinsic[4]), # eccentricity e0
    5: uniform_dist(Y0 - n*Delta_theta_intrinsic[5], Y0 + n*Delta_theta_intrinsic[5]), # Cosine of inclination (Y0 = cos(iota0))
    6: uniform_dist(dist - n*Delta_theta_D, dist + n* Delta_theta_D), # distance D
    # Extrinsic parameters -- Angular parameters
    7: uniform_dist(0, np.pi), # Polar angle (sky position)
    8: uniform_dist(0, 2*np.pi), # Azimuthal angle (sky position)
    9: uniform_dist(0, np.pi),  # Polar angle (spin vec)
    10: uniform_dist(0, 2*np.pi), # Azimuthal angle (spin vec)
    # Initial phases
    11: uniform_dist(0, 2*np.pi), # Phi_phi0
    12: uniform_dist(0, 2*np.pi), # Phi_theta0
    13: uniform_dist(0, 2*np.pi) # Phi_r00
}  

priors = ProbDistContainer(priors_in, use_cupy = False)   # Set up priors so they can be used with the sampler.

# =================== SET UP PROPOSAL ==================

moves_stretch = StretchMove(a=2.0, use_gpu=True)

# Quick checks
if ntemps > 1:
    print("Value of starting log-likelihood points", llike(start[0][0])) 
    if np.isinf(sum(priors.logpdf(xp.asarray(start[0])))):
        print("You are outside the prior range, you fucked up")
        quit()
else:
    print("Value of starting log-likelihood points", llike(start[0])) 

fp = "../data_files/test_few.h5"

backend = HDFBackend(fp)

ensemble = EnsembleSampler(
                            nwalkers,          
                            ndim,
                            llike,
                            priors,
                            backend = backend,                 # Store samples to a .h5 file
                            tempering_kwargs=tempering_kwargs,  # Allow tempering!
                            moves = moves_stretch
                            )
Reset_Backend = True # NOTE: CAREFUL HERE. ONLY TO USE IF WE RESTART RUNS!!!!
if Reset_Backend:
    os.remove(fp) # Manually get rid of backend
    backend = HDFBackend(fp) # Set up new backend
    ensemble = EnsembleSampler(
                            nwalkers,          
                            ndim,
                            llike,
                            priors,
                            backend = backend,                 # Store samples to a .h5 file
                            tempering_kwargs=tempering_kwargs,  # Allow tempering!
                            moves = moves_stretch
                            )
else:
    start = backend.get_last_sample() # Start from last sample
out = ensemble.run_mcmc(start, iterations, progress=True)  # Run the sampler

##===========================MCMC Settings (change this)============================

