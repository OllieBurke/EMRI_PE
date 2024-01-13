# EMRI_PE
# EMRI Parameter Estimation

Generic code used to estimate EMRI parameters. This is specific to the CNES cluster here in France. 

Viva la France! 

## Set up environments

Run 
```
source setup_file 
```

profit.

## Install dependencies:

The code relies on codes to generate EMRIs (FEW), the lisa response (GPU accelerated lisa-on-gpu) and a code that generates the latest power spectral densities of the noise processes. The repositories below should be installed. 

[lisaAnalysistools](https://github.com/mikekatz04/LISAanalysistools.git)
[lisa-on-gpu](https://github.com/mikekatz04/lisa-on-gpu.git)
[FastEMRIWaveforms](https://github.com/BlackHolePerturbationToolkit/FastEMRIWaveforms.git)

After activating your environment `conda activate vanilla_few` each of the above repositories should be installed within that environment. Simply run `python setup.py install` inside each repository.

## Code Structure

The code has been set up such that only one file needs to be edited. This is `EMRI_settings.py`. 


python mcmc_run.py

