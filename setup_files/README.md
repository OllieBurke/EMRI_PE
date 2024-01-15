# Setup files

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

Finally, it will write to the end of the file `cluster_sub/submit_job.sh` with the path of the executible `mcmc_run.py` specific to your path on your system. Just to make life that little bit easier for you. =). 

## Interactive mode

The lazy way to run this code is 

```
source cluster_modules.sh
```

If in interactive mode and you have ssh'd into a GPU, then it is important that you load in specific modules and activate the `vanilla_few` environment. These are 

```
module load conda
conda activate vanilla_few
module load gcc
module load cuda
```

After this, you can simply run the mcmc_run.py script. 

### Author

Ollie Burke
Laboratoire des deux infinis
Toulouse
France
ollie.burke@l2it.in2p3.fr
