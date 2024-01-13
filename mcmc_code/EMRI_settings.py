###
# This is the main file that should be edited. 
###
try:
    import cupy as cp
    import numpy as np
    xp = cp
    use_gpu = True
except ImportError:
    import numpy as np
    xp = np
    use_gpu = False

# Intrinsic parameters   
M = 1e6;    # Primary mass (units of solar mass)
mu = 10.0;  # Secondary mass (units of solar mass)
a = 0.9;    # Primary spin parameter (a \in [0,1])
p0 = 9.2;   # Initial semi-latus rectum (dimensionless)
e0 = 0.2;   # Initial eccentricity (dimensionless)
iota0 = 0.8;  # Initial inclination angle (with respect to the equatorial plane, (radians))
Y0 = np.cos(iota0);  

dist = 2.0;   # Distance (units of giga-parsecs)

# Angular variables
qS = 1.5; 
phiS = 0.7; 
qK = 1.2
phiK = 0.6;  

# Initial angular phases -- positional elements along the orbit. 
Phi_phi0 = 2.0   # Azimuthal phase
Phi_theta0 = 3.0;   # Polar phase
Phi_r0 = 4.0;    # Radial phase

# Waveform params
delta_t = 10.0;  # Sampling interval [seconds]
T = 2.0     # Evolution time [years]

mich = False #mich = True implies output in hI, hII long wavelength approximation

# keyword arguments for inspiral generator (RunKerrGenericPn5Inspiral)

inspiral_kwargs = {
    "DENSE_STEPPING": 0,  # we want a sparsely sampled trajectory -- much faster
    "max_init_len": int(1e4),  # all of the trajectories will be well under len = 1000
}

# keyword arguments for summation generator (AAKSummation)
sum_kwargs = {
    "use_gpu": use_gpu,  # GPU is available for this type of summation
    "pad_output": True,
}




