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

## Setting yourself up to run the code

### Interactive Mode

Run the following command:

`
salloc ...
`

Assuming that everything has run smoothly, resources will be allocated to you. It may look like trexgpuX is available. You want to run the command `ssh trexgpuX` in order to gain access to that specific GPU. 

Once you have entered the job, you want to run the following commands:

`
module load conda
conda activate vanilla_few
module load gcc
module load cuda
`
Or, if you wish to be lazy, just type `source cluster_modules` and everything will be done for you.

### Send the job to the cluster

In `cluster_sub/` there is a file called `submit_job.sh`. Running the command
`
sbatch submit_job.sh
`
will send the MCMC simulation off to the GPU and the samples (an .h5 file) will be send to `data_files`. These samples can then be analysed using the jupyter notebook located in `notebooks/analyse_samples.ipynb`.  
## Code Structure

The code has been set up such that only one file needs to be edited. This is `EMRI_settings.py`. In this file there is a list of EMRI parameters that dictate the true EMRI signal to be inferred. 

In `Cluster_sub/submit_job.sh`, there is a simple submit file one can use to submit jobs to the CNES clusters using the a100 GPUs. 


python mcmc_run.py

### Author

Ollie Burke
Laboratoire des deux infinis
Toulouse
France
ollie.burke@l2it.in2p3.fr
