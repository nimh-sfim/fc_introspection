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
MOCO_DIR=`echo ${PRCS_DATA_DIR}/${SBJ}/preprocessed/func/pb01_moco/${SCAN_ID}`

echo "++ INFO: Input Path     = ${INPUT_PATH}"
echo "         Input Filename = ${INPUT_FILE}"
echo "         Input Folder   = ${INPUT_DIR}"
echo "         Scan ID        = ${SCAN_ID}"
echo "         MOCO Folder    = ${MOCO_DIR}"
echo "         MNI Folder     = ${MNI_DIR}"
echo "==================================================================================================================================="

# Enter working directory
# =======================
cd ${MNI_DIR}

# Use 3dSeg output to generate a scan specific GM mask
# ====================================================
echo "\n"
echo "++ INFO: Create scan-specific GM mask"
echo "++ ----------------------------------"
3dresample -overwrite \
           -master ${MNI_DIR}/mask.FB.nii.gz \
           -rmode NN \
           -inset ${PRCS_DATA_DIR}/${SBJ}/preprocessed/anat/Segsy/Classes+tlrc \
           -prefix ${MNI_DIR}/mask.tissue_classes.nii.gz

# a) Get a file with the GM ribbon
# --------------------------------
echo "\n"
echo "++ INFO: Writting GM ribbon mask to disk"
echo "++ -------------------------------------"
3dcalc -overwrite -a ${MNI_DIR}/mask.tissue_classes.nii.gz -expr 'equals(a,2)' -prefix ${MNI_DIR}/mask.GM.nii.gz
3dcalc -overwrite -a ${MNI_DIR}/mask.GM.nii.gz -b ${MNI_DIR}/mask.FB.nii.gz -expr 'a*b' -prefix ${MNI_DIR}/mask.GM.nii.gz

# b) Get a ventricular mask
# -------------------------
echo "\n"
echo "++ INFO: Generating ventricular mask"
echo "++ ---------------------------------"
3dcalc -overwrite -a ${MNI_DIR}/mask.tissue_classes.nii.gz -expr 'equals(a,1)' -prefix ${MNI_DIR}/mask.CSF.nii.gz
3dcalc -overwrite -a ${PRCS_DATA_DIR}/ALL_SCANS/all_mean.mask.CSF_region.boxed.nii.gz  -b ${MNI_DIR}/mask.CSF.nii.gz -expr 'a*b' -prefix ${MNI_DIR}/mask.CSF.nii.gz
3dmask_tool -overwrite -input ${MNI_DIR}/mask.CSF.nii.gz -prefix ${MNI_DIR}/mask.CSF.nii.gz -fill_holes -fill_dirs xy

# c) Get an eroded WM mask
# ------------------------
echo "\n"
echo "++ INFO: Generating WM mask"
echo "++ ------------------------"
3dcalc -overwrite -a ${MNI_DIR}/mask.tissue_classes.nii.gz -expr 'equals(a,3)' -prefix ${MNI_DIR}/mask.WM.nii.gz
3dmask_tool -overwrite -inputs ${MNI_DIR}/mask.WM.nii.gz -dilate_inputs -1 -prefix ${MNI_DIR}/mask.WM.nii.gz
3dcalc -overwrite -a ${MNI_DIR}/mask.WM.nii.gz -b ${MNI_DIR}/mask.FB.nii.gz -expr 'a*b' -prefix ${MNI_DIR}/mask.WM.nii.gz

# d) Combine eWM and CSF into single file for CompCorr
# ----------------------------------------------------
echo "\n"
echo "++ INFO: Generating final ComCor mask (vents + WM)"
echo "++ -----------------------------------------------"
3dcalc -overwrite -a ${MNI_DIR}/mask.WM.nii.gz -b ${MNI_DIR}/mask.CSF.nii.gz -expr 'a+b' -prefix ${MNI_DIR}/mask.compcor.nii.gz

# Scaling Dataset
# ===============
echo "\n"
echo "++ INFO: Scale the data prior to any smoothing operation (needed for FC and ReHo analyses)"
echo "++ ---------------------------------------------------------------------------------------"
3dTstat -overwrite -prefix ${MNI_DIR}/rest2mni.b0.MEAN.nii.gz ${MNI_DIR}/rest2mni.nii.gz
3dcalc  -overwrite \
           -a ${MNI_DIR}/rest2mni.nii.gz \
           -b ${MNI_DIR}/rest2mni.b0.MEAN.nii.gz \
           -c ${MNI_DIR}/mask.FB.nii.gz \
           -expr 'c * min(200, a/b*100)*step(a)*step(b)' \
           -prefix ${MNI_DIR}/rest2mni.b0.scale.nii.gz


# Compute Nuisance Regressors
# ===========================
echo "\n"
echo "++ INFO: Generating Nuisance regressors"
echo "++ ===================================="

# a) Generate bandpass filtering regressors
# -----------------------------------------
echo "++ INFO: Computing bandpass filtering regressors"
echo "++ ---------------------------------------------"
nt=`3dinfo -nt ${MNI_DIR}/rest2mni.nii.gz`
tr=`3dinfo -tr ${MNI_DIR}/rest2mni.nii.gz`
echo "TR=$tr NT=$nt"
1dBport -nodata $nt $tr -band 0.01 0.15 -invert -nozero > ${MNI_DIR}/bandpass.1D

# b) Prepare Motion-related regressors
# ------------------------------------
echo "++ INFO: Creage Motion Regressors"
echo "++ ------------------------------"
1d_tool.py -overwrite -infile ${MOCO_DIR}/rest_realigned.par -set_nruns 1 -demean             -write ${MNI_DIR}/motion_demean.1D
1d_tool.py -overwrite -infile ${MOCO_DIR}/rest_realigned.par -set_nruns 1 -derivative -demean -write ${MNI_DIR}/motion_deriv.1D

# c) Prepare CompCorr based regressors
# ------------------------------------
echo "++ INFO: Compute Physio Noise Regressors as PCA of ventricular mask"
echo "++ ----------------------------------------------------------------"
cd ${MNI_DIR}
3dTproject -overwrite -mask ${MNI_DIR}/mask.FB.nii.gz -polort 3 -prefix rm.det_pcin.nii.gz -censor ${MOCO_DIR}/censor.1D -cenmode KILL -input rest2mni.b0.scale.nii.gz
 
3dpc -overwrite -mask ${MNI_DIR}/mask.compcor.nii.gz \
     -pcsave 5 \
     -prefix ${MNI_DIR}/rm.rest2mni.ROIPC.vents \
             ${MNI_DIR}/rm.det_pcin.nii.gz
 
1d_tool.py -overwrite -censor_fill_parent ${MOCO_DIR}/censor.1D -infile ${MNI_DIR}/rm.rest2mni.ROIPC.vents_vec.1D -write ${MNI_DIR}/rest2mni.ROIPC.vents.1D
rm ${MNI_DIR}/rm.det_pcin.nii.gz
rm ${MNI_DIR}/rm.rest2mni.ROIPC.vents* 


# Perform Nuissance Regression on a single step
# =============================================
echo "\n"
echo "++ INFO: Full denoisinig data"
echo "++ ==========================" 
 3dDeconvolve -overwrite                                               \
               -censor ${MOCO_DIR}/censor.1D                           \
               -input  ${MNI_DIR}/rest2mni.b0.scale.nii.gz             \
               -ortvec ${MNI_DIR}/bandpass.1D bandpass                 \
               -ortvec ${MNI_DIR}/motion_demean.1D mot_demean          \
               -ortvec ${MNI_DIR}/motion_deriv.1D  mot_deriv           \
               -ortvec ${MNI_DIR}/rest2mni.ROIPC.vents.1D compcor      \
               -polort 3                                               \
               -float                                                  \
               -num_stimts 0                                           \
               -jobs 24                                                \
               -fout -tout -x1D ${MNI_DIR}/X.b0.xmat.1D                \
               -xjpeg ${MNI_DIR}/X.b${b}.jpg                           \
               -x1D_uncensored ${MNI_DIR}/X.b0.nocensor.xmat.1D        \
               -x1D_stop

 3dTproject -overwrite                                               \
            -mask ${MNI_DIR}/mask.FB.nii.gz                          \
            -polort 0                                                \
            -input ${MNI_DIR}/rest2mni.b0.scale.nii.gz               \
            -censor ${MOCO_DIR}/censor.1D                            \
            -cenmode KILL                                            \
            -ort ${MNI_DIR}/X.b0.nocensor.xmat.1D                    \
            -prefix ${MNI_DIR}/rest2mni.b0.scale.denoise.nii.gz

# Print Success completion message
# ================================
echo "++ -------------------------------"
echo "++ INFO: Script finished correctly"
echo "++ -------------------------------"
