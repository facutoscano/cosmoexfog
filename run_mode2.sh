#!/bin/bash
#SBATCH --job-name=CosmoMode2
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --time=3-00:00
#SBATCH --output=logs/mode2_%j.out
#SBATCH --error=logs/mode2_%j.err

source /home/ftoscano/miniconda3/etc/profile.d/conda.sh
conda activate cobaya_env

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "=========================================================="
echo "Starting Job: Multipole Cuts"
echo "Node: $SLURMD_NODENAME"
echo "Working Folder: $(pwd)"
echo "=========================================================="

mkdir -p logs

python run_pipeline.py multipole_cuts --sampler iminuit

echo "=========================================================="
echo "Job Finished."
