# Author: Javier Gonzalez-Castillo
# Date: January 20th, 2023
#
# Description:
# This script takes two input parameters: subject and run name
#
set -e
echo "++ Load FSL, AFNI and ANTs modules"
module load afni
module load ANTs/2.2.0

# Unset DISPLAY variable
# ----------------------
echo "++ Unset DISPLAY variable"
unset DISPLAY

# Enter this subject pre-processing (sink) folder
PRCS_DATA_DIR='/data/SFIMJGC_Introspec/pdn/PrcsData/'
cd ${PRCS_DATA_DIR}/${SBJ}/preprocessed/func/

# Create a series of variables needed to access all files of interest
INPUT_PATH=`echo ${PRCS_DATA_DIR}/${SBJ}/preprocessed/func/pb03_coregister/_scan_id_ses-02_task-rest_acq-${RUN}_bold/rest2anat.nii.gz`
INPUT_FILE=`basename ${INPUT_PATH}`
INPUT_DIR=`dirname ${INPUT_PATH}`
SCAN_ID=`basename ${INPUT_DIR}`

MNI_DIR=`echo ${PRCS_DATA_DIR}/${SBJ}/preprocessed/func/pb05_mni/${SCAN_ID}`

echo "++ INFO: Input Path     = ${INPUT_PATH}"
echo "         Input Filename = ${INPUT_FILE}"
echo "         Input Folder   = ${INPUT_DIR}"
echo "         Scan ID        = ${SCAN_ID}"
echo "         MNI Folder     = ${MNI_DIR}"
echo "==================================================================================================================================="

# Enter working directory
# =======================
cd ${MNI_DIR}

# Reduce filesize by getting rid of empty borders (1.2G)
# ======================================================
echo "++ INFO: Remove area of zeros around the brain (to keep file size as small as possible)"
echo "++ ------------------------------------------------------------------------------------"
3dZeropad -overwrite \
             -master ${PRCS_DATA_DIR}/ALL_SCANS/all_mean.box.nii.gz \
             -prefix ${MNI_DIR}/rest2mni.nii.gz \
                     ${MNI_DIR}/rest2mni.nii.gz 
                     
3dZeropad -overwrite \
             -master ${PRCS_DATA_DIR}/ALL_SCANS/all_mean.box.nii.gz \
             -prefix ${MNI_DIR}/rest_mean_2mni.nii.gz \
                     ${MNI_DIR}/rest_mean_2mni.nii.gz

# Generate a Full Brain mask for each scan
# ========================================
echo "++ INFO: Generate a Full Brain mask for this scan"
echo "++ ----------------------------------------------"
3dAutomask -overwrite -peels 3 -prefix ${MNI_DIR}/mask.FB.nii.gz ${MNI_DIR}/rest_mean_2mni.nii.gz

3dcalc -overwrite -a ${PRCS_DATA_DIR}/ALL_SCANS/all_mean.mask.boxed.nii.gz \
                  -b ${MNI_DIR}/mask.FB.nii.gz \
                  -expr 'a*b' \
                  -prefix ${MNI_DIR}/mask.FB.nii.gz
                  
# Print Success completion message
# ================================
echo "++ INFO: Script finished correctly"
