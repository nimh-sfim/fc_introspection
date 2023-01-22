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

# If MNI Folder does not exist, we create it
# ==========================================
if [ ! -d ${MNI_DIR} ]; then mkdir ${MNI_DIR}; fi

# Enter working directory
# =======================
cd ${PRCS_DATA_DIR}/${SBJ}/preprocessed/func/

# Create Mean Image of Motion Corrected Timeseries
# ================================================
echo " + INFO: Generate mean image of motion corrected timeseries in original space"
3dTstat -overwrite -mean -prefix ${INPUT_DIR}/rest2anat.mean.nii.gz ${INPUT_PATH}

# Bring Mean Image into MNI space (big grid)
# ==========================================
echo " + INFO: Bring mean image into MNI space"
antsApplyTransforms --default-value 0 --float 0 \
                    --input ${INPUT_DIR}/rest2anat.mean.nii.gz \
                    --input-image-type 3 --interpolation BSpline \
                    --output ./pb05_mni/${SCAN_ID}/rest_mean_2mni.nii.gz \
                    --reference-image /usr/local/apps/fsl/5.0.10/data/standard/MNI152_T1_2mm_brain.nii.gz \
                    --transform [ /data/SFIMJGC_Introspec/pdn/PrcsData/${SBJ}/preprocessed/anat/transforms2mni/transform1Warp.nii.gz, 0 ] \
                    --transform [ /data/SFIMJGC_Introspec/pdn/PrcsData/${SBJ}/preprocessed/anat/transforms2mni/transform0GenericAffine.mat, 0 ]
3drefit -space MNI ${MNI_DIR}/rest_mean_2mni.nii.gz
    
# Transform Motion Corrected Timeseries into MNI space (4.2G)
# ===========================================================
echo "++ INFO: Running ANTs (volreg -> MNI transformation in one step)"
echo "++ ------------------------------------------------------------"
antsApplyTransforms --default-value 0 --float 0 \
                    --input ${INPUT_DIR}/rest2anat.nii.gz \
                    --input-image-type 3 --interpolation BSpline \
                    --output ${MNI_DIR}/rest2mni.nii.gz \
                    --reference-image /usr/local/apps/fsl/5.0.10/data/standard/MNI152_T1_2mm_brain.nii.gz \
                    --transform [ ${PRCS_DATA_DIR}/${SBJ}/preprocessed/anat/transforms2mni/transform1Warp.nii.gz, 0 ]  \
                    --transform [ ${PRCS_DATA_DIR}/${SBJ}/preprocessed/anat/transforms2mni/transform0GenericAffine.mat, 0 ]
3drefit -space MNI ${MNI_DIR}/rest2mni.nii.gz

# Print Success completion message
# ================================
echo "++ INFO: Script finished correctly"
