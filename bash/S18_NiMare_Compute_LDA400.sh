#!/bin/bash

# =============================================================================================
# Author: Javier Gonzalez-Castillo
# Date: 04/24/2023
#
# Description:
# This script sets the correct environment for cpm_batch.py to be able to proceed. It passes
# all provided inputs as inputs parameters to the program.
#
# =============================================================================================
set -e

# Once you are on a tmux/no machine terminal in an spersist node with the suggested configuration, do the following:

# Load conda environment
# ----------------------
source /data/SFIMJGC_HCP7T/Apps/miniconda38/etc/profile.d/conda.sh && conda activate fc_introspection 

# Unset DISPLAY variable
# ----------------------
unset DISPLAY

# Enter the notebook directory
# ----------------------------
cd /data/SFIMJGC_Introspec/2023_fc_introspection/code/fc_introspection/notebooks/

# Set NUMEXPR_MAX_THREADS to use 24 cpus
# --------------------------------------
export NUMEXPR_MAX_THREADS=24


# Run the LDA model for the 50 topics
# -----------------------------------
python ./S18_NiMare_Compute_LDA400_Models.py

# Completion message
# ------------------
echo "++ Script completed correctly"
