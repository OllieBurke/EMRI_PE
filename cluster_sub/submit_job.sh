#!/bin/sh

#SBATCH --job-name=test_EMRI
#SBATCH --output=output_logs/output_%j.out
#SBATCH --error=error_logs/error_%j.err
#SBATCH --account=lisa
#SBATCH --partition=gpu_a100
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu_all
#SBATCH --mem=100G
#SBATCH --cpus-per-gpu=20
#SBATCH --time=06:00:00

module load conda
module unload conda
module load conda
conda activate EMRI_PE_env 
module load gcc/10.2.0
module load cuda/11.7.0



