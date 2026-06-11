#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# CosmoExFog — Mode 2 (Cobaya MCMC) — Clemente batch partition
#
# Cobaya runs 4 parallel chains via MPI.  Each chain is one MPI task on the
# same node, with 7 CPUs for OpenMP threading in CAMB.
# 4 tasks × 7 CPUs = 28 cores = one full node.
#
# Usage:
#   mkdir -p logs
#   sbatch run_mode2_cobaya.sh
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=CosmoMode2_cobaya
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=7
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=logs/mode2_cobaya_%j.out
#SBATCH --error=logs/mode2_cobaya_%j.err

# ── Environment ───────────────────────────────────────────────────────────────
source /home/ftoscano/miniconda3/etc/profile.d/conda.sh
conda activate cmb-fit

# Let CAMB use the 7 CPUs assigned to each MPI task for its own threading.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# ── Run ───────────────────────────────────────────────────────────────────────
echo "=========================================================="
echo "Job: ${SLURM_JOB_NAME}  ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "MPI tasks: ${SLURM_NTASKS}  CPUs/task: ${SLURM_CPUS_PER_TASK}"
echo "Start: $(date)"
echo "=========================================================="

mpirun -n ${SLURM_NTASKS} python run_pipeline.py --config config.yaml multipole_cuts --sampler cobaya

echo "=========================================================="
echo "End: $(date)"
echo "Job finished."
echo "=========================================================="
