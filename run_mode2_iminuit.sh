#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# CosmoExFog — Mode 2 (iMinuit) — Clemente batch partition
#
# Usage:
#   mkdir -p logs            # must exist BEFORE sbatch (SLURM creates the file
#   sbatch run_mode2_iminuit.sh   # at job start, not at mkdir time)
#
# Resources:
#   One node, all 28 physical cores.  iMinuit uses mp.ProcessPoolExecutor,
#   so MPI is not needed here — we just want as many cores as possible.
#   Each core runs independent CAMB evaluations with OMP=1.
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=CosmoMode2_iminuit
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=logs/mode2_iminuit_%j.out
#SBATCH --error=logs/mode2_iminuit_%j.err

# ── Environment ───────────────────────────────────────────────────────────────
source /home/ftoscano/miniconda3/etc/profile.d/conda.sh
conda activate cmb-fit

# Disable threading inside workers so ProcessPoolExecutor gets clean cores.
# Without this, numpy/CAMB may spawn extra threads and oversubscribe the node.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# ── Run ───────────────────────────────────────────────────────────────────────
echo "=========================================================="
echo "Job: ${SLURM_JOB_NAME}  ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}  CPUs: ${SLURM_CPUS_PER_TASK}"
echo "Working dir: $(pwd)"
echo "Start: $(date)"
echo "=========================================================="

python run_pipeline.py --config config.yaml multipole_cuts --sampler iminuit

echo "=========================================================="
echo "End: $(date)"
echo "Job finished."
echo "=========================================================="
