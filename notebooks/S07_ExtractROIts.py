# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: FC Introspection (Jan 2023)
#     language: python
#     name: fc_introspection
# ---

# # Description - Create Swarm File to run transformation to MNI pipeline on the preprocessed data
#
# This notebook contians the code to extract representative timeseries for ROIs using the 7 Networks / 200 ROIs version of the Yeo Atlas.
#
# First, we will prepare the atlas for its use in this work
#
# Next, we will run AFNI's program ```3dNetCorr``` to extract the represenative timeseries. This second step will be done via a swarm job.

# # 1. Atlas Preparation
#
# This project uses the 200 ROI version of the Schaefer Atlas sorted according to the 7 Yeo Networks. This atlas is publicly available [here](https://github.com/ThomasYeoLab/CBIG/tree/master/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal).
#
# To prepare this atlas for the project, please perform the following operations:
#
# 1. Create a local folder for brain parcellations: (e.g., ```/data/SFIMJGC_Introspec/2023_fc_intrsopection/atlases```)
#
# 2. Assign the full path of that folder to variable [```ATLASES_DIR``` in ```basics.py```](https://github.com/nimh-sfim/fc_introspection/blob/main/notebooks/utils/basics.py#L36).
#
# 3. Assign the name of the atlas (Schaefer2018_200Parcels_7Networks) to the [```ATLAS_NAME``` variable in ```basics.py```](https://github.com/nimh-sfim/fc_introspection/blob/main/notebooks/utils/basics.py#L37).
#
#    > **NOTE**: Unless you are using a different version of the Schaefer Atlas, you can skip this step.
#
# 4. Create a sub-folder for the 200 Schaefer Atlas:
# ```bash
# # cd ${ATLASES_DIR}
# # mkdir ${ATLAS_NAME}
# ```
#
# 5. Copy the following files from their original location in CBIG repo (link above) to your local Schaefer Atlas folder:
#
# * [```Schaefer2018_200Parcels_7Networks_order_FSLMNI152_2mm.nii.gz```](https://github.com/ThomasYeoLab/CBIG/tree/master/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/MNI/Schaefer2018_200Parcels_7Networks_order_FSLMNI152_2mm.nii.gz): 200 ROI Atlas / 7 Yeo Networks in MNI space on a 2mm grid.
#
# * [```Schaefer2018_200Parcels_7Networks_order.txt```](https://github.com/ThomasYeoLab/CBIG/tree/master/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/MNI/freeview_lut/Schaefer2018_200Parcels_7Networks_order.txt): list of all ROI, their network membership and color code (RGB).
#
# * [```Schaefer2018_200Parcels_7Networks_order.lut```](https://github.com/ThomasYeoLab/CBIG/tree/master/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/MNI/fsleyes_lut/Schaefer2018_200Parcels_7Networks_order.lut): same as above, but coded for FSL eyes.
#
# * [```Schaefer2018_200Parcels_7Networks_order_FSLMNI152_2mm.Centroid_RAS.csv```](https://github.com/ThomasYeoLab/CBIG/tree/master/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/MNI/Centroid_coordinates/Schaefer2018_200Parcels_7Networks_order_FSLMNI152_2mm.Centroid_RAS.csv): information about each ROI centroid coordinates.
#
#

import subprocess
from utils.basics import ATLAS_PATH, ATLAS_NAME, DATA_DIR
import os.path as osp
from sfim_lib.atlases.raking import correct_ranked_shaefer_atlas

# ## 1.1. Correct space, generate label table and attach it to atlas file

# Correct the space tag, generate a label table, attach it to the original atlas file.
command = """module load afni; \
   cd {ATLAS_PATH}; \
   3drefit -space MNI {ATLAS_NAME}_order_FSLMNI152_2mm.nii.gz; \
   @MakeLabelTable -lab_file {ATLAS_NAME}_order.txt 1 0 -labeltable {ATLAS_NAME}_order.niml.lt -dset {ATLAS_NAME}_order_FSLMNI152_2mm.nii.gz;""".format(ATLAS_PATH=ATLAS_PATH,ATLAS_NAME=ATLAS_NAME)
output  = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
print(output.strip().decode())

# ## 1.2. Convert Atlas file to final grid of all pre-processed data

# Put in the final grid that agress with that of all fully pre-processed functional scans
command = """module load afni; \
   cd {ATLAS_PATH}; \
   3dZeropad -overwrite -master {DATA_DIR}/PrcsData/ALL_SCANS/all_mean.box.nii.gz -prefix {ATLAS_NAME}_order_FSLMNI152_2mm.nii.gz {ATLAS_NAME}_order_FSLMNI152_2mm.nii.gz; \
   3drefit -labeltable {ATLAS_NAME}_order.niml.lt {ATLAS_NAME}_order_FSLMNI152_2mm.nii.gz""".format(ATLAS_PATH=ATLAS_PATH,ATLAS_NAME=ATLAS_NAME, DATA_DIR=DATA_DIR)
output  = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
print(output.strip().decode())

# ## 1.3. Remove regions with bad coverage from the atlas

# Remove a series of regions with bad coverage / low signal levels
# Limbic LH Temporal Pole 1-4: 57,58,59,60
# Limbic RH Temporal Pole 1-3: 162,163,164
# Limbic LH OFC 1-2: 55,56
# Limbic RH OFC 1-3: 159,160,161
command="""module load afni; \
           cd {ATLAS_PATH}; \
           3dcalc -overwrite \
                  -a {ATLAS_NAME}_order_FSLMNI152_2mm.nii.gz \
                  -expr 'equals(a,57)+equals(a,58)+equals(a,59)+equals(a,60)+equals(a,162)+equals(a,163)+equals(a,164)+equals(a,55)+equals(a,56)+equals(a,159)+equals(a,160)+equals(a,161)' \
                  -prefix {ATLAS_NAME}_Removed_ROIs.nii.gz; \
           3dcalc -overwrite \
                  -a      {ATLAS_NAME}_order_FSLMNI152_2mm.nii.gz \
                  -expr 'a-57*equals(a,57)-58*equals(a,58)-59*equals(a,59)-60*equals(a,60)-162*equals(a,162)-163*equals(a,163)-164*equals(a,164)-55*equals(a,55)-56*equals(a,56)-159*equals(a,159)-160*equals(a,160)-161*equals(a,161)' \
                  -prefix {ATLAS_NAME}_order_FSLMNI152_2mm.nii.gz; \
           3drefit -labeltable {ATLAS_NAME}_order.niml.lt {ATLAS_NAME}_order_FSLMNI152_2mm.nii.gz""".format(ATLAS_PATH=ATLAS_PATH,ATLAS_NAME=ATLAS_NAME, DATA_DIR=DATA_DIR)
output  = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
print(output.strip().decode())

# ## 1.4. Rank the atlas with missing ROIs

command = """module load afni; \
             cd {ATLAS_PATH}; \
             3dRank -prefix {ATLAS_NAME}_order_FSLMNI152_2mm.ranked.nii.gz -input {ATLAS_NAME}_order_FSLMNI152_2mm.nii.gz;""".format(ATLAS_PATH=ATLAS_PATH,ATLAS_NAME=ATLAS_NAME)
output  = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
print(output.strip().decode())

# ## 1.5. Create rank corrected Order & Centroid Files

path_to_order_file = osp.join(ATLAS_PATH,'Schaefer2018_200Parcels_7Networks_order.txt')
path_to_rank_file  = osp.join(ATLAS_PATH,'Schaefer2018_200Parcels_7Networks_order_FSLMNI152_2mm.ranked.nii.gz.rankmap.1D')
path_to_centroids_file = osp.join(ATLAS_PATH,'Schaefer2018_200Parcels_7Networks_order_FSLMNI152_2mm.Centroid_RAS.csv')
correct_ranked_shaefer_atlas(path_to_order_file,path_to_centroids_file,path_to_rank_file)

# ## 1.6. Add corrected label table to the ranked version of the atlas

command = """module load afni; \
             cd {ATLAS_PATH}; \
             @MakeLabelTable -lab_file {ATLAS_NAME}_order.ranked.txt 1 0 -labeltable {ATLAS_NAME}_order.ranked.niml.lt -dset {ATLAS_NAME}_order_FSLMNI152_2mm.ranked.nii.gz;""".format(ATLAS_PATH=ATLAS_PATH,ATLAS_NAME=ATLAS_NAME)
output  = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
print(output.strip().decode())