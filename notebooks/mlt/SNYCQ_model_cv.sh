#!/bin/bash

set -e

# Enter the directory where the python file resides
echo "++ Entering Notebooks directory"
cd /data/SFIMJGC_Introspec/2023_fc_introspection/code/fc_introspection/notebooks/mlt/

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
   echo "   conda activate fc_introspection"
   conda activate fc_introspection
fi

if [ ${USER} == "spurneyma" ]; then
   echo " + User is spurneyma"
   echo "   conda activate dyneusr_prj_mas"
   conda activate dyneusr_prj_mas
fi

# Unset DISPLAY variable
echo "++ Unsetting DISPLAY environment variable"
unset DISPLAY

pwd 
# Run shape graphs script
echo "++ Calling Python script: SNYCQ_model_cv"
python ./SNYCQ_model_cv.py -f ${VAR_FILE} -d ${VAR_D} -s ${VAR_S} -r ${VAR_R} -j ${VAR_J}
