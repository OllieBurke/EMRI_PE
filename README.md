# EMRI_PE
# EMRI Parameter Estimation

Generic code used to estimate EMRI parameters. This is specific to the CNES cluster here in France. 

Vive la France! 

## Set up environments

Run 
```
cd setup_files
source setup_file.sh
```

profit.

## What does this do? 

The script `setup_file.sh` has been built for the lazy user in mind. Running `source setup_file` will

1. Build an conda environment `EMRI_PE_env` with necessary python modules 
2. Install a specific version of cupy for the GPUs here at CNES
3. Install eryn, the sampler that I use for MCMC with EMRIs. 
4. Create directory `Github_repos` to then clone (and install!) these three repositoriese
-    [lisaAnalysistools](https://github.com/mikekatz04/LISAanalysistools.git) **[credit: Michael Katz]**
-    [lisa-on-gpu](https://github.com/mikekatz04/lisa-on-gpu.git) **[credit: Michael Katz, Alvin Chua, Jean Baptiste-Bayle, Michele Valisneri]**
-    [FastEMRIWaveforms](https://github.com/BlackHolePerturbationToolkit/FastEMRIWaveforms.git) **[credit: Michael Katz, Alvin Chua, Niels Warburton, Lorenzo Speri, Scott Hughes]**

A user can (understandably) build the various conda environments or, if they already have FEW and lisa-on-gpu already on the cluster, they can just `python setup.py` install them separately. The code `source setup_file.sh` is used as a simple and quick way to get the EMRI code running. If the user would prefer a more manual installation, then go for it.

## Setting yourself up to run the code

### Interactive Mode

Run the following command:

```
salloc -A lisa -N 1 -n 1 -c 20 --gres=gpu:1 --mem=20G -p gpu_a100 --qos=gpu_all -t 01:00:00
```

Example output is 

```
salloc: Pending job allocation 4303336
salloc: job 4303336 queued and waiting for resources
salloc: job 4303336 has been allocated resources
salloc: Granted job allocation 4303336
salloc: Waiting for resource configuration
salloc: Nodes trexgpu03 are ready for job
```

this is assuming that everything has run smoothly. In this case, trexgpu03 is available. You want to run the command `ssh trexgpu03` in order to gain access to that specific GPU. 

Once you have entered the job, you want to run the following commands:

```
module load conda
conda activate EMRI_PE_env
module load gcc
module load cuda
```

Or, if you wish to be lazy, just type `source cluster_modules` in the directory `setup_files` and everything will be done for you.

If you then locate the `mcmc_code` directory, you can simply run 
```
python mcmc_run.py
```

If a progress bar appears at the bottom, then the sampler has been initiated!! 

Feel free to change `EMRI_parameters.py` to infer different parameter sets for the EMRI. 
### Send the job to the cluster

In `cluster_sub/` there is a file called `submit_job.sh`. Running the command
```
sbatch submit_job.sh
```

will send the MCMC simulation off to the GPU and the samples (an .h5 file) will be send to `data_files`. These samples can then be analysed using the jupyter notebook located in `notebooks/analyse_samples.ipynb`.  

You can check on your job by writing

```
squeue -u $USER
```

## Code Structure

The code has been set up such that only one file needs to be edited. This is `EMRI_settings.py`. In this file there is a list of EMRI parameters that dictate the true EMRI signal to be inferred. 

In `Cluster_sub/submit_job.sh`, there is a simple submit file one can use to submit jobs to the CNES clusters using the a100 GPUs. 


### Author

Ollie Burke

Laboratoire des deux infinis

Toulouse

France

ollie.burke@l2it.in2p3.fr
