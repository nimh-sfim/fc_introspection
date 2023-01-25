# Author: Javier Gonzalez-Castillo
# Date: January 23th, 2023
#
# Description:
# This script takes two input parameters: subject and run name
#
# This script will extract representative timeseries for the default atlas used in this work

set -e
echo "++ Load AFNI"
module load afni

# Unset DISPLAY variable
# ----------------------
echo "++ Unset DISPLAY variable"
unset DISPLAY

# Enter this subject pre-processing (sink) folder
PRCS_DATA_DIR='/data/SFIMJGC_Introspec/pdn/PrcsData/'
SFC_DIR=`echo ${PRCS_DATA_DIR}/${SBJ}/preprocessed/func/pb06_staticFC`

ATLAS_DIR='/data/SFIMJGC_Introspec/2023_fc_introspection/atlases/Schaefer2018_200Parcels_7Networks'
ATLAS_PATH=`echo ${ATLAS_DIR}/Schaefer2018_200Parcels_7Networks_order_FSLMNI152_2mm.ranked.nii.gz`
ATLAS_NAME='Schaefer2018_200Parcels_7Networks'

INPUT_PATH=`echo ${PRCS_DATA_DIR}/${SBJ}/preprocessed/func/pb05_mni/_scan_id_ses-02_task-rest_acq-${RUN}_bold/rest2mni.b0.scale.denoise.nii.gz`
INPUT_FILE=`basename ${INPUT_PATH}`
INPUT_DIR=`dirname ${INPUT_PATH}`
SCAN_ID=`basename ${INPUT_DIR}`
MNI_DIR=`echo ${PRCS_DATA_DIR}/${SBJ}/preprocessed/func/pb05_mni/${SCAN_ID}`
MOCO_DIR=`echo ${PRCS_DATA_DIR}/${SBJ}/preprocessed/func/pb01_moco/${SCAN_ID}`
FB_MASK_PATH=`echo ${MNI_DIR}/mask.FB.nii.gz`

echo "++ INFO: Input Path     = ${INPUT_PATH}"
echo "         Input Filename = ${INPUT_FILE}"
echo "         Input Folder   = ${INPUT_DIR}"
echo "         Scan ID        = ${SCAN_ID}"
echo "         SFC Folder     = ${SFC_DIR}"
echo "         Atlas Path     = ${ATLAS_PATH}"
echo "         MOCO Path      = ${MOCO_DIR}"
echo "         FB Mask Path   = ${FB_MASK_PATH}"
echo "==================================================================================================================================="

# Extract list of good volumes
num_good_vols=`cat ${MOCO_DIR}/motion_good_volume_list.csv | tr -s ',' '\n' | wc -l`
echo "        Number of volumes for this run after censory = ${num_good_vols}"
echo "==================================================================================================================================="

# Check all files are on the same grid
3dinfo -n4 -same_all_grid -d3 -header_name -header_line ${ATLAS_PATH} ${INPUT_PATH} ${FB_MASK_PATH}
echo "==================================================================================================================================="

# Extract Timeseries and Correlation Matrix
cd ${SFC_DIR}
pwd
3dNetCorr -overwrite                   \
          -mask ${FB_MASK_PATH}        \
          -inset ${INPUT_PATH}         \
          -in_rois ${ATLAS_PATH}       \
          -prefix ${RUN}.${ATLAS_NAME} \
          -ts_out

rm ${RUN}.${ATLAS_NAME}_000.niml.dset
rm ${RUN}.${ATLAS_NAME}_000.roidat

echo "=================================================="  
echo "++ INFO: Finished computation of static FC matrix "
echo "=================================================="  