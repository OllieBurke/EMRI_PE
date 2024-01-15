#!/bin/bash

#echo Now going to build environments and set up dependencies. 
#
cd .. # enter root directory

echo building directory for data files
mkdir data_files  # Build data directory to store .h5 samples

echo Loading conda module
module load conda  # CNES cluster, need to load conda prior to using it

echo Now creating environment      # Set up conda environment -- vanilla_few
conda create -n EMRI_PE_env -c conda-forge gcc_linux-64 gxx_linux-64 wget gsl lapack=3.6.1 hdf5 numpy Cython scipy tqdm jupyter ipython h5py requests matplotlib corner python=3.9 
conda activate EMRI_PE_env         

echo Installing cupy-cuda11x toolkit
pip install cupy-cuda11x           # Warning: this is SPECIFIC to the CNES cluster for the GPUs available

module load gcc

# Important to load cuda when installing the repos below! 
module load cuda

echo Installing eryn, sampler built off emcee. 
pip install eryn                   # Install Eryn 

# The code below git clones various repositories and installs them in one sitting
echo Now going to clone dependencies 
mkdir Github_Repos; cd Github_Repos

# Clone the repositories
git clone https://github.com/mikekatz04/LISAanalysistools.git
git clone https://github.com/mikekatz04/lisa-on-gpu.git
git clone https://github.com/BlackHolePerturbationToolkit/FastEMRIWaveforms.git

# install each one using python
echo Now installing LISAanalysistools 
cd LISAanalysistools
python setup.py install

cd ../lisa-on-gpu
echo Now installing lisa-on-gpu 
python setup.py install

echo Now installing FastEMRIWaveforms 
cd ../FastEMRIWaveforms
python setup.py install

cd ../../ # Get back to root directory

# Now to set up submit file.

insert_text() {
    # Insert text in a variable script at a variable line number with some text
    local target_file=$1
    local line_number=$2
    local text=$3

    # Use awk to insert the text at the specified line in the target file
    awk -v line="$line_number" -v text="$text" 'NR==line {$0=text} {print}' "$target_file" > temp && mv temp "$target_file"
}

echo $pwd

cd mcmc_code
exec_path=$(pwd) # Extract path of python code to be executed

cd ../cluster_sub

# Replace submit_job.sh at line 22 with the executible path
insert_text "submit_job.sh" 22 "python $exec_path/mcmc_run.py"

echo Your installation is complete!




