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
source /data/SFIMJGC_HCP7T/Apps/miniconda38/etc/profile.d/conda.sh && conda activate fc_introspection 

# Enter lemon dataset pre-processing pipelines
# --------------------------------------------
cd /data/SFIMJGC_Introspec/2023_fc_introspection/code/fc_introspection/notebooks/cpm/

# Unset DISPLAY variable
# ----------------------
unset DISPLAY

BASIC_INPUT_ARG_LIST=`echo -b ${BEHAV_PATH} -f ${FC_PATH} -o ${OUT_DIR} -t ${BEHAVIOR} -k ${NUM_FOLDS} -i ${NUM_ITER} -c ${CORR_TYPE} -s ${E_SUMMARY_METRIC} -m ${SPLIT_MODE}`

# Ensure we pass thresholding options
# ===================================
# NOTE: Density thresholding not implemented yet.
if [[ "${E_THR_P}" == "None" ]]; then 
   echo "++ P-thresholding = None"
else
   echo "++ Setting P-thresholding flag"
   ARG_LIST=`echo ${BASIC_INPUT_ARG_LIST} -p ${E_THR_P}` 
fi
if [[ "${E_THR_R}" == "None" ]]; then
   echo "++ R-thresholding = None"
else
   echo "++ Setting R-thresholding flag"; ARG_LIST=`echo ${BASIC_INPUT_ARG_LIST} -r ${E_THR_R}` 
fi

# Pass additional boolean flags if needed
# =======================================
if [[ "${RANDOMIZE_BEHAVIOR}" == "True" ]]; then echo "++ Setting Randomization Flag"; ARG_LIST=`echo ${ARG_LIST} --randomize_behavior`; fi
if [[ "${CONFOUNDS}" == "True" ]]; then echo "++ Setting Confound Residualization Flag"; ARG_LIST=`echo ${ARG_LIST} --residualize_motion -M ${CONFOUNDS_PATH}`; fi

if [[ "${VERBOSE}" == "True" ]]; then echo "++ Setting Verbose Flag"; ARG_LIST=`echo ${ARG_LIST} --verbose`; fi

# CALL THE CPM_BATCH PROGRAM 
# ==========================
echo "++ Calling: python ./cpm_batch.py ${ARG_LIST}"
python ../../python/cpm_batch_program.py ${ARG_LIST}
