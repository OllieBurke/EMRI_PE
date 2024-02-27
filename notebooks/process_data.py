import numpy as np
from eryn.backends import HDFBackend as eryn_HDF_Backend
import matplotlib.pyplot as plt
import corner
import os
import warnings
import matplotlib.lines as mlines
from matplotlib.font_manager import FontProperties
import sys
sys.path.append("../mcmc_code")

from EMRI_settings import (M, mu, a, p0, e0, Y0, 
                      dist, Phi_phi0, Phi_theta0, Phi_r0, qS, phiS, qK, phiK, 
                      mich, T, inspiral_kwargs, sum_kwargs, xp, use_gpu, delta_t) 

# Now analyse the results using 9PN for circular orbits

reader = eryn_HDF_Backend('../data_files/kerr_eq_ecc_a0p9_M1e6_mu10_e0_0p2_p0_8p58_SNR_43.h5',read_only = True)

N_iterations = reader.get_chain()['model_0'].shape[0]
N_temps = reader.get_chain()['model_0'].shape[1]
N_walkers = reader.get_chain()['model_0'].shape[2]
N_params = reader.get_chain()['model_0'].shape[-1]

samples_after_burnin = [reader.get_chain(discard = 800)['model_0'][:,i].reshape(-1,N_params) 
                    for i in range(N_temps)]  # Take true chain]

log_like = reader.get_log_like(discard = 0)



# Check against FM approx
print("Finished reading in the data")

true_vals = [M, mu, a, p0, e0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_r0]
params_corner =[r"$M/M_{\odot}$", r"$\mu/M_{\odot}$", r"$a$", 
                     r"$p_{0}$", r"$e_{0}$", r"$D_{s}/Gpc$", 
                     r"$\theta_{S}$", r"$\phi_{S}$", r"$\theta_{K}$", r"$\phi_{K}$", 
                     r"$\Phi_{\phi_{0}}$", r"$\Phi_{r_{0}}$"] 


corner_kwargs = dict(plot_datapoints=False,smooth1d=True,
                       labels=params_corner, levels=(1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9 / 2.)), 
                       label_kwargs=dict(fontsize=40), max_n_ticks=4,
                       show_titles=False, smooth = True, labelpad = 0.4)


samples_corner = np.column_stack(samples_after_burnin)

print("Now building the corner plot")
figure = corner.corner(samples_corner,bins = 30, color = 'blue', **corner_kwargs)

axes = np.array(figure.axes).reshape((N_params, N_params))

for i in range(N_params):
    ax = axes[i, i]
    ax.axvline(true_vals[i], color="k")
    
for yi in range(N_params):
    for xi in range(yi):
        ax = axes[yi, xi]
        ax.axhline(true_vals[yi], color="k")
        ax.axvline(true_vals[xi],color= "k")
        ax.plot(true_vals[xi], true_vals[yi], "sk")
        
for ax in figure.get_axes():
    ax.tick_params(axis='both', labelsize=18)
  
blue_line = mlines.Line2D([], [], color='blue', label=r'Posterior Distributions')
red_line = mlines.Line2D([], [], color='red', label=r'Fisher Matrix Estimate')
black_line = mlines.Line2D([], [], color='black', label='True Value')

params_labels_vals = "M = {}, mu = {}, a = {}, p0 = {}, e0 = {}, \ndist = {}, qs = {}, phis = {}, thetak = {}, phik = {}, \nPhi_phi0 = {},  Phi_r0 = {}, SNR = 43\n".format(true_vals[0],true_vals[1],true_vals[2],true_vals[3],true_vals[4], true_vals[5], true_vals[6], true_vals[7], true_vals[8], true_vals[9], true_vals[10], true_vals[11])

plt.legend(handles=[blue_line, black_line], fontsize = 65, frameon = True, bbox_to_anchor=(0.25, N_params+1.5), loc="upper right", title = params_labels_vals, title_fontproperties = FontProperties(size = 40, weight = 'bold'))
plt.subplots_adjust(left=-0.1, bottom=-0.1, right=None, top=None, wspace=0.15, hspace=0.15)
print("Now saving")
plt.savefig("plots/kerr_eq_ecc_a0p9_M1e6_mu10_e0_0p2_p0_8p58_SNR_43.pdf",bbox_inches="tight")
