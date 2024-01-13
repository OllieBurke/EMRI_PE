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

## What does this do? 

The script `setup_file` has been built for the lazy user in mind. Running `source setup_file` will

1. Build an conda environment vanilla_few with necessary python modules 
2. Install a specific version of cupy for the GPUs here at CNES
3. Install eryn, the sampler that I use for MCMC with EMRIs. 
4. Create directory `Github_repos` to then clone (and install!) these three repositories 
    a) [lisaAnalysistools](https://github.com/mikekatz04/LISAanalysistools.git)
    b) [lisa-on-gpu](https://github.com/mikekatz04/lisa-on-gpu.git)
    c) [FastEMRIWaveforms](https://github.com/BlackHolePerturbationToolkit/FastEMRIWaveforms.git)

A user can (understandably) build the various conda environments or, if they already have FEW and lisa-on-gpu already on the cluster, they can just `python setup.py` install them separately.

## Code Structure

The code has been set up such that only one file needs to be edited. This is `EMRI_settings.py`. In this file there is a list of EMRI parameters that dictate the true EMRI signal to be inferred. 

In `Cluster_sub/submit_job.sh`, there is a simple submit file one can use to submit jobs to the CNES clusters using the a100 GPUs. 


python mcmc_run.py

