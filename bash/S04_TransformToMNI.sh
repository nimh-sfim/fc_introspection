#set -e

# Load conda environment
# ----------------------
echo "++ Activate Conda environment lemon_preproc_py27_nipype"
if [ ${USER} == "javiergc" ]; then
   echo "++ User is javiergc"
   echo "   source /data/SFIMJGC_HCP7T/Apps/miniconda38/etc/profile.d/conda.sh"
   source /data/SFIMJGC_HCP7T/Apps/miniconda38/etc/profile.d/conda.sh
fi

if [ ${USER} == "spurneyma" ]; then
   echo "++ User is spurneyma"
   echo "   source /data/spurneyma/miniconda/etc/profile.d/conda.sh"
   source /data/spurneyma/miniconda/etc/profile.d/conda.sh
fi

conda activate lemon_preproc_py27_nipype
   
# Load software needed by the pipeline available as modules in biowulf
# --------------------------------------------------------------------
echo "++ Load FSL, AFNI and ANTs modules"
module load fsl/5.0.10
module load afni
module load ANTs/2.2.0

# Enter lemon dataset pre-processing pipelines
# --------------------------------------------
echo "++ Enter NC Pipelines folder"
cd /data/SFIMJGC_Introspec/2023_fc_introspection/code/lsd_pipelines_nih/src/lsd_lemon/

# Unset DISPLAY variable
# ----------------------
echo "++ Unset DISPLAY variable"
unset DISPLAY

# Run transformation to MNI pipeline
# ----------------------------------
echo "++ Run ToMNI pipeline"
python ./run_lsd_tomni_nih.py s ${SBJ}

# Extra steps to get the volreg into MNI space (for computing box)
# ================================================================
echo "++ ============================================================================="
echo "++ Nipype ended "
echo "++ ============================================================================="
cd /data/SFIMJGC_Introspec/pdn/PrcsData/${SBJ}/preprocessed/func/
pwd
INPUTS=(`ls ./pb03_coregister/_scan_id_ses-02_task-rest_acq-??_run-??_bold/rest2anat.nii.gz`)
echo "++ INPUT FILES:  ${INPUTS[@]}}"
for INPUT_PATH in ${INPUTS[@]}
do
    echo " + WORKING ON RUN [${INPUT_PATH}]"  
    INPUT_FILE=`basename ${INPUT_PATH}`
    INPUT_DIR=`dirname ${INPUT_PATH}`
    SCAN_ID=`basename ${INPUT_DIR}`
    
    # Transform mean image into MNI space
    # ===================================
    echo " +                ${INPUT_PATH} --> ${INPUT_FILE} --> ${INPUT_DIR} --> ${SCAN_ID}"
    3dTstat -overwrite -mean -prefix ${INPUT_DIR}/rest2anat.mean.nii.gz ${INPUT_PATH}
    antsApplyTransforms --default-value 0 --float 0 \
                        --input ${INPUT_DIR}/rest2anat.mean.nii.gz \
                        --input-image-type 3 --interpolation BSpline \
                        --output ./pb05_mni/${SCAN_ID}/rest_mean_2mni.nii.gz \
                        --reference-image /usr/local/apps/fsl/5.0.10/data/standard/MNI152_T1_2mm_brain.nii.gz \
                        --transform [ /data/SFIMJGC_Introspec/pdn/PrcsData/${SBJ}/preprocessed/anat/transforms2mni/transform1Warp.nii.gz, 0 ] \
                        --transform [ /data/SFIMJGC_Introspec/pdn/PrcsData/${SBJ}/preprocessed/anat/transforms2mni/transform0GenericAffine.mat, 0 ]
    3drefit -space MNI ./pb05_mni/${SCAN_ID}/rest_mean_2mni.nii.gz
done
echo "++ INFO: Script finished successfully...."
