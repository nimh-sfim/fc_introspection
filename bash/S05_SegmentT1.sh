# Author: Javier Gonzalez-Castillo
# Date: June 16th, 2022
# 
# Description:
# This script will do most pre-processing that happens after all spatial transformation matrices
# have been computed using Nipype
#
set -e

# Load software needed by the pipeline available as modules in biowulf
# --------------------------------------------------------------------
echo "++ Load FSL, AFNI and ANTs modules"
module load afni

# Unset DISPLAY variable
# ----------------------
echo "++ Unset DISPLAY variable"
unset DISPLAY

# Print Configuration Parameters
# ==============================
PRCS_DATA_DIR='/data/SFIMJGC_Introspec/pdn/PrcsData/'

ANAT_DIR=`echo ${PRCS_DATA_DIR}/${SBJ}/preprocessed/anat`

echo " ++ INFO: ANAT Folder    = ${ANAT_DIR}"
echo "================================================================================================================"

# Extra steps on the anatomical
# =============================
echo "++ INFO: Correct space flag in MNI version of the T1"
echo "++ -------------------------------------------------"
cd ${ANAT_DIR}
3drefit -space MNI T1_brain2mni.nii.gz
if [ -d Segsy ]; then
   rm -rf Segsy
fi
echo "++ INFO: Run 3dSeg on T1"
echo "++ ---------------------"
3dSeg -anat T1_brain2mni.nii.gz -mask AUTO -classes 'CSF ; GM ; WM' -bias_classes 'GM ; WM' -bias_fwhm 25 -mixfrac UNI -main_N 5 -blur_meth BFT

echo "++ =============================== "
echo "++  SCRIPT FINISHED CORRECTLY      "
echo "++ =============================== "
