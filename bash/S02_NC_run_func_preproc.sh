set -e


# Activating miniconda
echo "++ Activating miniconda"
if [ ${USER} == "javiergc" ]; then
   echo " + User is javiergc";
   echo "   /data/SFIMJGC_HCP7T/Apps/miniconda38/etc/profile.d/conda.sh"
   source /data/SFIMJGC_HCP7T/Apps/miniconda38/etc/profile.d/conda.sh
fi

if [ ${USER} == "spurneyma" ]; then
   echo " + User is spurneyma"
   echo "   source /data/spurneyma/miniconda/etc/profile.d/conda.sh"
   source /data/spurneyma/miniconda/etc/profile.d/conda.sh
fi

# Activate the correct conda environment
echo "++ Activating dyneusr environment"
if [ ${USER} == "javiergc" ]; then
   echo " + User is javiergc";
   echo "   conda activate lemon_preproc_py27_nipype"
   conda activate lemon_preproc_py27_nipype
fi

if [ ${USER} == "spurneyma" ]; then
   echo " + User is spurneyma"
   echo "   conda activate lemon_preproc_py27_nipype"
   conda activate lemon_preproc_py27_nipype
fi

# Load software needed by the pipeline available as modules in biowulf
# --------------------------------------------------------------------
module load fsl/5.0.10
module load afni
module load freesurfer/5.3.0

# Freesurfer setup
# ----------------
source $FREESURFER_HOME/SetUpFreeSurfer.sh

# Enter lemon dataset pre-processing pipelines
# --------------------------------------------
cd /data/SFIMJGC_Introspec/2023_fc_introspection/code/lsd_pipelines_nih/src/lsd_lemon/

# Unset DISPLAY variable
# ----------------------
unset DISPLAY

# Run transformation to MNI pipeline
# ----------------------------------
python ./run_resting_lsd_nih.py s ${SBJ}
