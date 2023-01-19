set -e

# Load conda environment
# ----------------------
#source /data/spurneyma/miniconda/etc/profile.d/conda.sh && conda activate lemon_preproc_py27_nipype
source /data/SFIMJGC_HCP7T/Apps/miniconda38/etc/profile.d/conda.sh && conda activate lemon_preproc_py27_nipype

# Load software needed by the pipeline available as modules in biowulf
# --------------------------------------------------------------------
module load fsl/5.0.10
module load freesurfer/5.3.0
module load ANTs/2.2.0

# Freesurfer setup
# ----------------
source $FREESURFER_HOME/SetUpFreeSurfer.sh

# MIPAV Setup
# -----------
export JAVALIB=/data/SFIMJGC_Introspec/2023_fc_introspection/sw/mipav/jre/lib/ext/
export MIPAV=/data/SFIMJGC_Introspec/2023_fc_introspection/sw/mipav
export PLUGINS=/home/javiergc/mipav/plugins
export CLASSPATH=$JAVALIB*:$MIPAV:$MIPAV/lib/*:$PLUGINS

#export JAVALIB=/data/SFIMJGC_Introspec/2023_fc_introspection/sw/mipav/jre/lib/ext/
#export MIPAV=/data/SFIMJGC_Introspec/2023_fc_introspection/sw/mipav/
#export PLUGINS=/home/javiergc/mipav/plugins/
#export CLASSPATH=$JAVALIB*:$MIPAV:$MIPAVlib/*:$PLUGINS

echo "CLASSPATH = ${CLASSPATH}"

# Enter lemon dataset pre-processing pipelines
# --------------------------------------------
cd /data/SFIMJGC_Introspec/2023_fc_introspection/code/lsd_pipelines_nih/src/lsd_lemon

# Unset DISPLAY variable
# ----------------------
unset DISPLAY

# Run transformation to MNI pipeline
# ----------------------------------
python ./run_structural_nih.py s ${SBJ} ${ANAT_PATH} ${ANAT_PREFIX}
