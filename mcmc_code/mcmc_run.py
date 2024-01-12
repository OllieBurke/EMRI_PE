import numpy as np
import cupy as cp
import os 
import matplotlib.pyplot as plt
import sys
sys.path.append("../")
from EMRI_settings import (M, mu, a, p0, e0, Y0, 
                      dist, Phi_phi0, Phi_theta0, Phi_r0, qS, phiS, qK, phiK, 
                      mich, T, inspiral_kwargs, sum_kwargs, xp, use_gpu) 

# from scipy.signals.window import tukey
from scipy.signal import tukey
from lisatools.sensitivity import noisepsd_AE,noisepsd_T
from lisatools.diagnostic import *
from few.trajectory.inspiral import EMRIInspiral
from few.utils.utility import get_separatrix, Y_to_xI, get_p_at_t
from fastlisaresponse import ResponseWrapper

from few.waveform import Pn5AAKWaveform

# Need tempering to cope with eccentricity...
from eryn.ensemble import EnsembleSampler
from eryn.moves import StretchMove
from eryn.prior import ProbDistContainer, uniform_dist
from eryn.backends import HDFBackend

YRSID_SI = 31558149.763545603

np.random.seed(1234)

# order of the langrangian interpolation
t0 = 20000.0   # How many samples to remove from start and end of simulations
order = 25

orbit_file_esa = "/home/ad/burkeol/work/Github_repositories/lisa-on-gpu/orbit_files/esa-trailing-orbits.h5"

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
    N = len(data)
    pow_2 = xp.ceil(np.log2(N))
    return xp.pad(data,(0,int((2**pow_2)-N)),'constant')

def inner_prod(sig1_f,sig2_f,N_t,delta_t,PSD):
    prefac = 4*delta_t / N_t
    sig2_f_conj = xp.conjugate(sig2_f)
    return prefac * xp.real(xp.sum((sig1_f * sig2_f_conj)/PSD))


##======================Likelihood and Posterior (change this)=====================

def llike(params):

    M_val = float(params[0])
    mu_val =  float(params[1])
    a_val =  float(params[2])            # This works fine! 
    p0_val = float(params[3])
    e0_val = float(params[4])
    Y0_val = float(params[5])
    D_val = float(params[6])
    qS_val = float(params[7])
    phiS_val = float(params[8])
    qK_val = float(params[9])
    phiK_val = float(params[10])

    Phi_phi0_val = float(params[11])
    Phi_theta0_val = float(params[12])
    Phi_r0_val = float(params[13])

    waveform_prop = EMRI_TDI(M_val, mu_val, a_val, p0_val, e0_val, 
                                  Y0_val, D_val, qS_val, phiS_val, qK_val, phiK_val,
                                    Phi_phi0=Phi_phi0_val, Phi_theta0=Phi_theta0_val, Phi_r0=Phi_r0_val, 
                                    mich=mich, dt=delta_t, T=T)  # Generate h_plus and h_cross

    EMRI_AET_w_pad_prop = [zero_pad(window*waveform_prop[i]) for i in range(N_channels)]

    EMRI_AET_fft_prop = [xp.fft.rfft(item) for item in EMRI_AET_w_pad_prop]

    diff_f_AET = [data_f_AET[k] - EMRI_AET_fft_prop[k] for k in range(N_channels)]
    
    inn_prod = xp.asarray([inner_prod(diff_f_AET[k],diff_f_AET[k],N_t,delta_t,PSD_AET[k]) for k in range(N_channels)])
    
    llike_val_np = xp.asnumpy(-0.5 * (xp.sum(inn_prod))) # Eryn does not like things being spat out as cupy arrays.
    return (llike_val_np)

def lpost(params):
    '''
    Compute log posterior
    '''
    if cp.isinf(lprior(params)):
        print("Prior returns -\infty")
        return -np.inf
    else:
        return llike(params)

##==========================Waveform Settings========================
delta_t = 10 
sampling_frequency =  1/delta_t
## ===================== CHECK TRAJECTORY CLASS ====================

traj = EMRIInspiral(func="pn5")
t_traj, p_traj, e_traj, Y_traj, Phi_phi_traj, Phi_r_traj, Phi_theta_traj = traj(M, mu, a, p0, e0, Y0,
                                             Phi_phi0=Phi_phi0, Phi_theta0=Phi_theta0, Phi_r0=Phi_r0, T=T)

x_I_traj = Y_to_xI(a, p_traj, e_traj, Y_traj)

traj_args = [M, mu, a, e_traj[0], x_I_traj[0]]
index_of_p = 3

t_out = T
# run trajectory
p_new = get_p_at_t(
    traj,
    t_out,
    traj_args,
    index_of_p=3,
    index_of_a=2,
    index_of_e=4,
    index_of_x=5,
    xtol=2e-12,
    rtol=8.881784197001252e-16,
    bounds=None,
)

print("We require initial separatrix of ",p_new, "for inspiral lasting", t_out, "years")
print("Final point in semilatus rectum achieved is", p_traj[-1])
print("Separatrix : ", get_separatrix(a, e_traj[-1], x_I_traj[-1]))

# Construct the AAK model with 5PN trajectories

AAK_waveform_model = Pn5AAKWaveform(inspiral_kwargs=inspiral_kwargs, sum_kwargs=sum_kwargs, use_gpu=use_gpu)

####=======================True waveform==========================
params = [M,mu,a,p0,e0,Y0,dist,qS, phiS, qK, phiK] 

EMRI_TDI = ResponseWrapper(AAK_waveform_model,T,delta_t,
                          index_lambda,index_beta,t0=t0,
                          flip_hx = True, use_gpu = use_gpu, is_ecliptic_latitude=False,
                          remove_garbage = "zero", **tdi_kwargs_esa)

waveform = EMRI_TDI(*params, Phi_phi0 = Phi_phi0, Phi_theta0 = Phi_theta0, Phi_r0 = Phi_r0)  # Generate h_plus and h_cross

window = cp.asarray(tukey(len(waveform[0]),0.001))
EMRI_AET_w_pad = [zero_pad(window*waveform[i]) for i in range(N_channels)]
N_t = len(EMRI_AET_w_pad[0])

EMRI_AET_fft = xp.asarray([xp.fft.rfft(item) for item in EMRI_AET_w_pad])
freq = xp.fft.rfftfreq(N_t,delta_t)
freq[0] = freq[1]   # To "retain" the zeroth frequency

# Define PSDs
freq_np = xp.asnumpy(freq)

PSD_AET = [noisepsd_AE(freq_np, includewd=T),noisepsd_AE(freq_np,includewd=T),noisepsd_T(freq_np,includewd=T)]

PSD_AET = [cp.asarray(item) for item in PSD_AET] # Convert to cupy array

SNR2_AET = xp.asarray([inner_prod(EMRI_AET_fft[i],EMRI_AET_fft[i],N_t,delta_t,PSD_AET[i]) for i in range(N_channels)])

for i in range(N_channels):
    print("SNR in channel {0} is {1}".format(TDI_channels[i],SNR2_AET[i]**(1/2)))

SNR2 = xp.asnumpy(xp.sum(SNR2_AET)**(1/2))
print("Final SNR = ",SNR2)
##=====================Noise Setting: Currently 0=====================

variance_noise_AET = [N_t * PSD_AET[k] / (4*delta_t) for k in range(N_channels)]
noise_f_AET_real = [xp.random.normal(0,np.sqrt(variance_noise_AET[k])) for k in range(N_channels)]
noise_f_AET_imag = [xp.random.normal(0,np.sqrt(variance_noise_AET[k])) for k in range(N_channels)]
noise_f_AET = xp.asarray([noise_f_AET_real[k] + 1j * noise_f_AET_imag[k] for k in range(N_channels)])


data_f_AET = EMRI_AET_fft + 0*noise_f_AET   # define the data

##===========================MCMC Settings (change this)============================
iterations = 4000 #10000  # The number of steps to run of each walker
burnin = 0
nwalkers = 50  #50 #members of the ensemble, like number of chains

ntemps = 1            # Number of temperatures used for parallel tempering scheme.
                       # Each group of walkers (equal to nwalkers) is assigned a temperature from T = 1, ... , ntemps.

tempering_kwargs=dict(ntemps=ntemps)  # Sampler requires the number of temperatures as a dictionary

d = 0

#here we should be shifting by the *relative* error! 

start_M = M*(1. + d * 1e-7 * np.random.randn(nwalkers,1))   # changed to 1e-6 careful of starting points! Before I started on secondaries... haha.
start_mu = mu*(1. + d * 1e-7 * np.random.randn(nwalkers,1))
start_a = a*(1. + d * 1e-7 * np.random.randn(nwalkers,1))
start_p0 = p0*(1. + d * 1e-8 * np.random.randn(nwalkers, 1))
start_e0 = e0*(1. + d * 1e-7 * np.random.randn(nwalkers, 1))
start_Y0 = Y0*(1. + d * 1e-7 * np.random.randn(nwalkers, 1))

start_D = dist*(1 + d * 1e-6 * np.random.randn(nwalkers,1))


start_qS = qS*(1. + d * 1e-6 * np.random.randn(nwalkers,1))
start_phiS = phiS*(1. + d * 1e-6 * np.random.randn(nwalkers,1))
start_qK = qK*(1. + d * 1e-6 * np.random.randn(nwalkers,1))
start_phiK = phiK*(1. + d * 1e-6 * np.random.randn(nwalkers,1))

start_Phi_Phi0 = Phi_phi0*(1. + d * 1e-6 * np.random.randn(nwalkers, 1))
start_Phi_theta0 = Phi_theta0*(1. + d * 1e-6 * np.random.randn(nwalkers, 1))
start_Phi_r0 = Phi_r0*(1. + d * 1e-6 * np.random.randn(nwalkers, 1))

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
Delta_theta_D = dist/np.sqrt(np.sum(SNR2))
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

# os.chdir('/home/ad/burkeol/work/1PA_systematics/Data_Analysis/Parameter_Estimation/mcmc_code_ecc/with_response/cluster_sub')
fp = "../data_files/test_few.h5"


backend = HDFBackend(fp)

# ensemble = EnsembleSampler(
#                             nwalkers,          
#                             ndim,
#                             llike,
#                             priors,
#                             backend = backend,                 # Store samples to a .h5 file, sets up backend
#                             tempering_kwargs=tempering_kwargs,  # Allow tempering!
#                             moves = moves_stretch
#                             ) 
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

