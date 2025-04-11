#!/bin/sh

#SBATCH --job-name="tdmms_dl-training"
#SBATCH -e stderrs/%j-tdmms_dl-training.err
#SBATCH -o stdouts/%j-tdmms_dl-training.out
#SBATCH --partition=gpu-a100
#SBATCH --time=09:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=8G
#SBATCH --account=education-as-bsc-tn

module load miniconda3

unset CONDA_SHLVL
source "$(conda info --base)/etc/profile.d/conda.sh"

conda activate env_tf24

previous=$(nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' | /usr/bin/tail -n '+2')
nvidia-smi

cd /home/aldelange/ai/tdmms_DL
srun python train.py --starting_material MoS2 --last_layers True

/usr/bin/nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' | /usr/bin/grep -v -F "$previous"

conda deactivate
