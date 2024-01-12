
try:
    import cupy as cp
    xp = cp
    use_gpu = True
except ImportError:
    import numpy as np
    xp = np
    use_gpu = False

import numpy as np
import cupy as cp
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70.5, Om0=0.274, Ob0 = 0.046)
 
M = 1e6; mu = 10; a = 0.9; e0 = 0.2; iota0 = 0.8; Y0 = np.cos(iota0); Phi_phi0 = 2
Phi_theta0 = 3; Phi_r0 = 4; p0 = 9.2; qS = 1.5; phiS = 0.7; qK = 1.2
phiK = 0.6; dist = 2.0; mich = False #mich = True implies output in hI, hII long wavelength approximation

dt = 10.0; T = 2.0

# keyword arguments for inspiral generator (RunKerrGenericPn5Inspiral)
inspiral_kwargs = {
    "DENSE_STEPPING": 0,  # we want a sparsely sampled trajectory
    "max_init_len": int(1e4),  # all of the trajectories will be well under len = 1000
}

# keyword arguments for summation generator (AAKSummation)
sum_kwargs = {
    "use_gpu": use_gpu,  # GPU is availabel for this type of summation
    "pad_output": True,
}




