#!/bin/bash

# =============================================================================================
# Author: Javier Gonzalez-Castillo
# Date: 03/10/2023
#
# Description:
# This script sets the correct environment for cpm_batch.py to be able to proceed. It passes
# all provided inputs as inputs parameters to the program.
#
# =============================================================================================
set -e

# Load conda environment
# ----------------------
#source /data/SFIMJGC_HCP7T/Apps/miniconda38/etc/profile.d/conda.sh && conda activate fc_introspection 
source /data/SFIMJGC_HCP7T/Apps/miniconda38/etc/profile.d/conda.sh && conda activate fc_introspection_2023_py310 

# Enter lemon dataset pre-processing pipelines
# --------------------------------------------
cd /data/SFIMJGC_Introspec/2023_fc_introspection/code/fc_introspection/python/

# Unset DISPLAY variable
# ----------------------
unset DISPLAY

ARG_LIST=`echo -V ${VOCAB} -t ${TOPIC}`

# Pass additional boolean flags if needed
# =======================================

if [[ "${VERBOSE}" == "True" ]]; then echo "++ Setting Verbose Flag"; ARG_LIST=`echo ${ARG_LIST} --verbose`; fi
if [[ "${SETUP}" == "True" ]]; then echo "++ Setting Verbose Flag"; ARG_LIST=`echo ${ARG_LIST} -s`; fi

# CALL THE CPM_BATCH PROGRAM 
# ==========================
echo "++ Calling: python ./S15_NiMare_Create_TopicMaps.py ${ARG_LIST}"
python ./S15_NiMare_Create_TopicMaps.py ${ARG_LIST}
