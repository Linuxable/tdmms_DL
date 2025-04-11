#!/bin/sh

#SBATCH --job-name="tdmms_dl-training"
#SBATCH -e stderrs/%j-tdmms_dl-training.err
#SBATCH -o stdouts/%j-tdmms_dl-training.out
#SBATCH --partition=gpu-a100-small
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=32G
#SBATCH --account=education-as-bsc-tn

module load miniconda3

unset CONDA_SHLVL
source "$(conda info --base)/etc/profile.d/conda.sh"

conda create -n env_tf24 python=3.8.10 -c conda-forge
conda activate env_tf24

conda install cudatoolkit=11.0 -c conda-forge
conda install cudnn=8.0 -c conda-forge

cd /home/aldelange/ai/tdmms_DL

pip install -r requirements.txt

conda deactivate
