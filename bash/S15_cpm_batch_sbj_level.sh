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

if [[ ! -z "${E_THR_R}" ]]; then echo "++ Received R-based Threshold"; ARG_LIST=`echo ${BASIC_INPUT_ARG_LIST} -r ${E_THR_R}`; fi
if [[ ! -z "${E_THR_P}" ]]; then echo "++ Received P-based Threshold"; ARG_LIST=`echo ${BASIC_INPUT_ARG_LIST} -p ${E_THR_P}`; fi

if [[ "${RANDOMIZE_BEHAVIOR}" == "True" ]]; then echo "++ Setting Randomization Flag"; ARG_LIST=`echo ${ARG_LIST} --randomize_behavior`; fi
if [[ "${CONFOUNDS}" == "True" ]]; then echo "++ Setting Confound Residualization Flag"; ARG_LIST=`echo ${ARG_LIST} --residualize_motion`; fi

if [[ "${VERBOSE}" == "True" ]]; then echo "++ Setting Verbose Flag"; ARG_LIST=`echo ${ARG_LIST} --verbose`; fi
# Run transformation to MNI pipeline
# ----------------------------------
echo "++ Calling: python ./cpm_batch_sbj_level.py ${ARG_LIST}"
python ./cpm_batch_sbj_level.py ${ARG_LIST}
