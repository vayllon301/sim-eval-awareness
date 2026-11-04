#!/bin/bash
# Exploratory (Hopper sessions; docs/SESSIONS.md). Not part of the testbed.
#
# A Slurm job that reads a run's exported episodes on one GPU: explore/interp/read_run.sh, as it is,
# with WORK on scratch. submit_read.sh, on the laptop, puts the code and the episodes there first and
# submits this. No server runs here and no episode runs: only the reader. Its exit code is the job's:
# 3 means episodes are left, and submitting again resumes.
#
# Required (sbatch --export): WORK. Optional: read_run.sh's settings (PARTS, PROJ_ENGINE, LAYERS,
# SAE_LAYERS, TRANSFER_LAYERS, CHUNK_MIN, NO_SERVER_IDS). --export splits on commas and can't carry spaces, so
# PARTS and the layer lists go with plus signs. The time limit is a parameter of sbatch, not set here:
# CHUNK_MIN, which proj and logit share, must end before it, since a job killed mid-episode loses that episode.
#SBATCH --job-name=sim-read
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
set -uo pipefail
: "${WORK:?set WORK to a scratch directory with about 110 GB free}"
export WORK
exec bash "${CODE:-$WORK/sim}/explore/interp/read_run.sh"
