# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: FC Instrospection (2023 | 3.10)
#     language: python
#     name: fc_introspection_2023_py310
# ---

# # Description - Information and Code to prepare the atlases used in this work
#
# For this paper we use a combination of two altases: Schaeffer 400 ROI atlas (for cortical regions) and AAL v2 atlas (for subcortical regions).
#
# This notebook will help you prepare final versions of the atlases that contain ROIs well within the imaging FOV of the sample and with ROIs numbered contigously. It will also help you generate accessory files with color, ROI ID and ROI location info needed for plotting and for sorting ROI by network membership.
#
# By the end of this notebook, there will be two atlases of interest:
#
# * ```Schaefer2018_400Parcels_7Networks```: Atlase with 187 ROI distributed across 6 Yeo Networks (all ROIs from the Limbic network are removed due to FOV constrains)
# * ```Schaefer2018_400Parcels_7Networks_AAL2```: Atlas with 195 ROIs that include the 187 from the atlas above plus 8 extra subcortical ROIs: L/R Thalamus, L/R Pallidum, L/R Caudate, L/R Putamen
#
# Prior to running this notebook you need to make sure the following variables are correctly set:
#
# 1. Assign the full path to the atlases folder to variable [```ATLASES_DIR``` in ```basics.py```](https://github.com/nimh-sfim/fc_introspection/blob/main/notebooks/utils/basics.py#L65).

import subprocess
import getpass
import os
import pandas as pd
from datetime import datetime
from shutil import rmtree
from utils.basics import CORTICAL_400ROI_ATLAS_PATH, CORTICAL_400ROI_ATLAS_NAME, SUBCORTICAL_ATLAS_PATH, SUBCORTICAL_ATLAS_NAME, FB_400ROI_ATLAS_NAME, FB_400ROI_ATLAS_PATH
from utils.basics import DATA_DIR, PRJ_DIR, SCRIPTS_DIR, ATLASES_DIR
from utils.basics import get_sbj_scan_list, rgb2hex
import os.path as osp
from sfim_lib.atlases.raking import correct_ranked_atlas
import hvplot.pandas
import holoviews as hv
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
print('++ Pandas version (<2.0 needed becuase some functions still use pd.append) --> This kernel uses: %s' % str(pd.__version__))

# # 1. Atlas Preparation: Schaeffer 400 ROI Atlas
#
# This project uses the 400 ROI version of the Schaefer Atlas sorted according to the 7 Yeo Networks. This atlas is publicly available [here](https://github.com/ThomasYeoLab/CBIG/tree/master/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal).
#
# To prepare this atlas for the project, please perform the following operations:
#
# 1. Create a local folder for brain parcellations: (e.g., ```/data/SFIMJGC_Introspec/2023_fc_intrsopection/atlases```)
#
#
#    > **NOTE**: Unless you are using a different version of the Schaefer Atlas, you can skip this step.
#
#
# 3. Create a sub-folder for the 400 Schaefer Atlas:
# ```bash
# # cd ${ATLASES_DIR}
# # mkdir Schaefer2018_400Parcels_7Networks
# ```
#
# 3. Copy the following files from their original location in CBIG repo (link above) to your local Schaefer Atlas folder:
#
# * [```Schaefer2018_400Parcels_7Networks_order_FSLMNI152_2mm.nii.gz```](https://github.com/ThomasYeoLab/CBIG/tree/master/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/MNI/Schaefer2018_400Parcels_7Networks_order_FSLMNI152_2mm.nii.gz): 400 ROI Atlas / 7 Yeo Networks in MNI space on a 2mm grid.
#
# * [```Schaefer2018_400Parcels_7Networks_order.txt```](https://github.com/ThomasYeoLab/CBIG/tree/master/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/MNI/freeview_lut/Schaefer2018_400Parcels_7Networks_order.txt): list of all ROI, their network membership and color code (RGB).
#
# * [```Schaefer2018_400Parcels_7Networks_order.lut```](https://github.com/ThomasYeoLab/CBIG/tree/master/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/MNI/fsleyes_lut/Schaefer2018_400Parcels_7Networks_order.lut): same as above, but coded for FSL eyes.
#
# * [```Schaefer2018_400Parcels_7Networks_order_FSLMNI152_2mm.Centroid_RAS.csv```](https://github.com/ThomasYeoLab/CBIG/tree/master/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/MNI/Centroid_coordinates/Schaefer2018_400Parcels_7Networks_order_FSLMNI152_2mm.Centroid_RAS.csv): information about each ROI centroid coordinates.

# 4. Correct space, generate label table and attach it to atlas file

# Correct the space tag, generate a label table, attach it to the original atlas file.
command = """module load afni; \
   cd {ATLAS_PATH}; \
   mv {ATLAS_NAME}_order_FSLMNI152_2mm.Centroid_RAS.csv {ATLAS_NAME}.Centroid_RAS.csv; \
   3drefit -space MNI {ATLAS_NAME}_order_FSLMNI152_2mm.nii.gz; \
   3dcopy -overwrite  {ATLAS_NAME}_order_FSLMNI152_2mm.nii.gz {ATLAS_NAME}.nii.gz; \
   @MakeLabelTable -lab_file {ATLAS_NAME}_order.txt 1 0 -labeltable {ATLAS_NAME}.niml.lt -dset {ATLAS_NAME}.nii.gz;""".format(ATLAS_PATH=CORTICAL_400ROI_ATLAS_PATH,ATLAS_NAME=CORTICAL_400ROI_ATLAS_NAME)
output  = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
print(output.strip().decode())

# 5. Convert Atlas file to final grid of all pre-processed data

# Put in the final grid that agress with that of all fully pre-processed functional scans
command = """module load afni; \
   cd {ATLAS_PATH}; \
   3dZeropad -overwrite -master {DATA_DIR}/PrcsData/ALL_SCANS/all_mean.box.nii.gz -prefix {ATLAS_NAME}.nii.gz {ATLAS_NAME}.nii.gz; \
   3drefit -labeltable {ATLAS_NAME}.niml.lt {ATLAS_NAME}.nii.gz""".format(ATLAS_PATH=CORTICAL_400ROI_ATLAS_PATH,ATLAS_NAME=CORTICAL_400ROI_ATLAS_NAME, DATA_DIR=DATA_DIR)
output  = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
print(output.strip().decode())

# 6. Find problematic regions (limited coverage by FOV with good signal quality)

sbj_list, scan_list = get_sbj_scan_list(when='post_motion', return_snycq=False)

mask_list = ''
for sbj,run in scan_list:
    _,_,sid,_,rid,_,acq = run.split('-')
    scan_folder = f'_scan_id_ses-{sid}_task-rest_acq-{acq}_run-{rid}_bold'
    path = osp.join(sbj,'preprocessed','func','pb05_mni',scan_folder,'mask.FB.nii.gz')
    mask_list = mask_list + ' ' + path
command = f"""module load afni; \
             cd {DATA_DIR}/PrcsData/; \
             3dMean -overwrite -prefix {DATA_DIR}/PrcsData/ALL_SCANS/all_mean.mask.FB.nii.gz {mask_list}; \
             3dcalc -overwrite -a {DATA_DIR}/PrcsData/ALL_SCANS/all_mean.mask.FB.nii.gz -expr 'ispositive(a-0.95)' -prefix {DATA_DIR}/PrcsData/ALL_SCANS/all_mean.mask.FB.95pc.nii.gz; \
             3dROIstats -nomeanout -nzvoxels -mask {CORTICAL_400ROI_ATLAS_PATH}/{CORTICAL_400ROI_ATLAS_NAME}.nii.gz {DATA_DIR}/PrcsData/ALL_SCANS/all_mean.mask.FB.95pc.nii.gz \
                        > {CORTICAL_400ROI_ATLAS_PATH}/{CORTICAL_400ROI_ATLAS_NAME}.coverage.txt"""
output  = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
print(output.strip().decode())

path = osp.join(CORTICAL_400ROI_ATLAS_PATH,f"{CORTICAL_400ROI_ATLAS_NAME}.coverage.txt")
df = pd.read_csv(path, delimiter='\t')
df = df.drop(['File','Sub-brick'],axis=1)
df = df.T
df.columns = ['Coverage']
df.index = [i.split('_',2)[2] for i in df.index]
df.index.name = 'ROI_Name'
df.reset_index(inplace=True)
df.index = df.index + 1
df.index.name = 'ROI_ID'

df.hvplot.hist(y='Coverage',bins=100) * hv.VLine(50).opts(color='k', line_dash='dashed')

bad_rois = list(df[df['Coverage'] <=50].index)

bad_rois_minus = '-'.join([str(roi)+'*equals(a,'+str(roi)+')' for roi in bad_rois])
bad_rois_plus  = '+'.join([str(roi)+'*equals(a,'+str(roi)+')' for roi in bad_rois])

# 7. Remove regions with bad coverage from the atlas

command=f"""module load afni; \
           cd {CORTICAL_400ROI_ATLAS_PATH}; \
           3dcalc -overwrite \
                  -a {CORTICAL_400ROI_ATLAS_NAME}.nii.gz \
                  -expr '{bad_rois_plus}' \
                  -prefix {CORTICAL_400ROI_ATLAS_NAME}_Removed_ROIs.nii.gz; \
           3dcalc -overwrite \
                  -a      {CORTICAL_400ROI_ATLAS_NAME}.nii.gz \
                  -expr 'a-{bad_rois_minus}' \
                  -prefix {CORTICAL_400ROI_ATLAS_NAME}.nii.gz; \
           3drefit -labeltable {CORTICAL_400ROI_ATLAS_NAME}.niml.lt {CORTICAL_400ROI_ATLAS_NAME}.nii.gz"""
output  = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
print(output.strip().decode())

# 8. Rank the atlas with missing ROIs

command = f"""module load afni; \
             cd {CORTICAL_400ROI_ATLAS_PATH}; \
             3dRank -prefix {CORTICAL_400ROI_ATLAS_NAME}.ranked.nii.gz -input {CORTICAL_400ROI_ATLAS_NAME}.nii.gz;"""
output  = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
print(output.strip().decode())

# 9. Create rank corrected Order & Centroid Files

path_to_order_file = osp.join(CORTICAL_400ROI_ATLAS_PATH,f'{CORTICAL_400ROI_ATLAS_NAME}_order.txt')
path_to_rank_file  = osp.join(CORTICAL_400ROI_ATLAS_PATH,f'{CORTICAL_400ROI_ATLAS_NAME}.ranked.nii.gz.rankmap.1D')
path_to_centroids_file = osp.join(CORTICAL_400ROI_ATLAS_PATH,f'{CORTICAL_400ROI_ATLAS_NAME}.Centroid_RAS.csv')
correct_ranked_atlas(path_to_order_file,path_to_centroids_file,path_to_rank_file)

# 10. Add corrected label table to the ranked version of the atlas

command = f"""module load afni; \
             cd {CORTICAL_400ROI_ATLAS_PATH}; \
             @MakeLabelTable -lab_file {CORTICAL_400ROI_ATLAS_NAME}_order.ranked.txt 1 0 -labeltable {CORTICAL_400ROI_ATLAS_NAME}_order.ranked.niml.lt -dset {CORTICAL_400ROI_ATLAS_NAME}.ranked.nii.gz;"""
output  = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
print(output.strip().decode())

# 11. Create a dataframe with all the necessary info about this atlas
#
# We now create a single dataframe with all the info we need: roi number, roi label, network, hemisphere, colors codes and centroid position. We save this to disk so that it can be easily accessed by any other notebook

# Load the cetroid file for the ranked atlas in memory
centroids_info               = pd.read_csv(osp.join(ATLASES_DIR, CORTICAL_400ROI_ATLAS_NAME,f'{CORTICAL_400ROI_ATLAS_NAME}.Centroid_RAS.ranked.csv' ))
centroids_info['ROI Name']   = [label.split('7Networks_')[1] for label in centroids_info['ROI Name']]
centroids_info['Hemisphere'] = [item.split('_')[0] for item in centroids_info['ROI Name']]
centroids_info['Network']    = [item.split('_')[1] for item in centroids_info['ROI Name']]
# Load the color info file for the ranked atlas in memory
color_info = pd.read_csv(osp.join(ATLASES_DIR, CORTICAL_400ROI_ATLAS_NAME,f'{CORTICAL_400ROI_ATLAS_NAME}_order.ranked.txt'),sep='\t', header=None)
# Combine all the useful columns into a single new dataframe
df         = pd.concat([centroids_info[['ROI Label','Hemisphere','Network','ROI Name','R','A','S']],color_info[[2,3,4]]], axis=1)
df.columns = ['ROI_ID','Hemisphere','Network','ROI_Name','pos_R','pos_A','pos_S','color_R','color_G','color_B']
df['RGB']  = [rgb2hex(r,g,b) for r,g,b in df.set_index('ROI_ID')[['color_R','color_G','color_B']].values]
# Save the new data frame to disk
df.to_csv(osp.join(ATLASES_DIR,CORTICAL_400ROI_ATLAS_NAME,f'{CORTICAL_400ROI_ATLAS_NAME}.ranked.roi_info.csv'), index=False)

# 11. Clean-up folder and assign file file names to completely pre-processed atlas

# ```bash
# # cd /data/SFIMJGC_Introspec/2023_fc_introspection/atlases/Schaefer2018_400Parcels_7Networks
# # mkdir orig
# # mv Schaefer2018_400Parcels_7Networks.Centroid_RAS.csv orig/
# # mv Schaefer2018_400Parcels_7Networks.nii.gz orig/
# # mv Schaefer2018_400Parcels_7Networks_order.lut orig/
# # mv Schaefer2018_400Parcels_7Networks_order.txt orig/
# # mv Schaefer2018_400Parcels_7Networks.niml.lt orig/
# # mv Schaefer2018_400Parcels_7Networks_order_FSLMNI152_2mm.nii.gz orig/
#
# # mv Schaefer2018_400Parcels_7Networks.Centroid_RAS.ranked.csv Schaefer2018_400Parcels_7Networks.Centroid_RAS.csv
# # mv Schaefer2018_400Parcels_7Networks_order.ranked.niml.lt Schaefer2018_400Parcels_7Networks_order.niml.lt
# # mv Schaefer2018_400Parcels_7Networks_order.ranked.txt Schaefer2018_400Parcels_7Networks_order.txt
# # mv Schaefer2018_400Parcels_7Networks.ranked.nii.gz Schaefer2018_400Parcels_7Networks.nii.gz
# # mv Schaefer2018_400Parcels_7Networks.ranked.nii.gz.rankmap.1D Schaefer2018_400Parcels_7Networks.nii.gz.rankmap.1D
# # mv Schaefer2018_400Parcels_7Networks.ranked.roi_info.csv Schaefer2018_400Parcels_7Networks.roi_info.csv
# ```

# ***
# # 2. Prepare the AAL v2 Atlas
#
# We will use this atlas to obtain defintions for 8 subcortical regions not includes in the Schaeffer atlas, namely L/R caudate, L/R putamen, L/R pallidum and L/R thalamus.
#
# To prepare this atlas for the project, please preform the following operations:
#
# 1. Create a sub-folder within ```ATLASES_DIR``` for this second atlas
#
# ```bash
# # cd ${ATLASES_DIR}
# # mkdir aal2
# ```
#
# 2. Download the AAL v2 Atlas from [here](https://www.gin.cnrs.fr/en/tools/aal/)
#
# 3. Unzip the contents of the downloaded file into the AAL2 folder.
#
#    > **NOTE**: we only need two files ```aal2.nii.gz``` and ```aal2.nii.txt```. So make sure those are available in ```${ATLASES_DIR}/aal2``` 
#
# 4. We need to correct the space in ```aal2.nii.gz``` to have the correct MNI space tag for AFNI

# Correct the space tag, generate a label table, attach it to the original atlas file.
command = """module load afni; \
   cd {ATLAS_PATH}; \
   3drefit -space MNI {ATLAS_NAME}.nii.gz; \
   @MakeLabelTable -lab_file {ATLAS_NAME}.nii.txt 1 0 -labeltable {ATLAS_NAME}.niml.lt -dset {ATLAS_NAME}.nii.gz;""".format(ATLAS_PATH=SUBCORTICAL_ATLAS_PATH,ATLAS_NAME=SUBCORTICAL_ATLAS_NAME)
output  = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
print(output.strip().decode())

# 5. Convert Atlas to the same grid as the functional data

# Put in the final grid that agress with that of all fully pre-processed functional scans
command = """module load afni; \
   cd {ATLAS_PATH}; \
   3dZeropad -overwrite -master {DATA_DIR}/PrcsData/ALL_SCANS/all_mean.box.nii.gz -prefix {ATLAS_NAME}.nii.gz {ATLAS_NAME}.nii.gz; \
   3drefit -labeltable {ATLAS_NAME}.niml.lt {ATLAS_NAME}.nii.gz""".format(ATLAS_PATH=SUBCORTICAL_ATLAS_PATH,ATLAS_NAME=SUBCORTICAL_ATLAS_NAME, DATA_DIR=DATA_DIR)
output  = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
print(output.strip().decode())

# 6. Create new atlas file with only the 8 ROIs we need

command = """module load afni; \
             cd {ATLAS_PATH}; \
             3dcalc -overwrite -a {ATLAS_NAME}.nii.gz \
                    -expr '75*equals(a,75) + 76*equals(a,76) + 77*equals(a,77) + 78*equals(a,78) + 79*equals(a,79) + 80*equals(a,80) + 81 * equals(a,81) + 82*equals(a,82)' \
                    -prefix {ATLAS_NAME}.subcortical.nii.gz; \
             3dmask_tool -overwrite -inputs {ATLAS_NAME}.subcortical.nii.gz -dilate_inputs -1 -prefix rm.{ATLAS_NAME}.subcortical.nii.gz; \
             3dcalc -overwrite -a {ATLAS_NAME}.subcortical.nii.gz -b rm.{ATLAS_NAME}.subcortical.nii.gz -expr 'a*b' -prefix {ATLAS_NAME}.subcortical.nii.gz; \
             3drefit -labeltable {ATLAS_NAME}.niml.lt {ATLAS_NAME}.subcortical.nii.gz; \
             rm rm.{ATLAS_NAME}.subcortical.nii.gz;""".format(ATLAS_PATH=SUBCORTICAL_ATLAS_PATH,ATLAS_NAME=SUBCORTICAL_ATLAS_NAME)
output  = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
print(output.strip().decode())

# 7. Remove areas of overlap between the two atlases
#
# Upon visual inspection we realized that despite having eroded the subcortical ROIs from the AAL2 atlas, we could still observe one voxel that overlaps with the Schaeffer atlas. Given at a later stage we will be merging both atlases, it is best to remove this voxel from the ```aal2.subcortical.nii.gz``` file now.

command="""module load afni; \
           cd {SUBCORTICAL_ATLAS_PATH};  \
           3dcalc -overwrite -a {SUBCORTICAL_ATLAS_NAME}.subcortical.nii.gz -b {CORTICAL_ATLAS_PATH}/{CORTICAL_ATLAS_NAME}.nii.gz -expr 'equals(0,step(a*b))' -prefix rm.overlap_between_atlases.nii.gz; \
           3dcalc -overwrite -a {SUBCORTICAL_ATLAS_NAME}.subcortical.nii.gz -b rm.overlap_between_atlases.nii.gz -expr 'a*b' -prefix {SUBCORTICAL_ATLAS_NAME}.subcortical.nii.gz; \
           3drefit -labeltable {SUBCORTICAL_ATLAS_NAME}.niml.lt {SUBCORTICAL_ATLAS_NAME}.subcortical.nii.gz; \
           rm rm.overlap_between_atlases.nii.gz;""".format(SUBCORTICAL_ATLAS_PATH=SUBCORTICAL_ATLAS_PATH, SUBCORTICAL_ATLAS_NAME=SUBCORTICAL_ATLAS_NAME, CORTICAL_ATLAS_NAME=CORTICAL_400ROI_ATLAS_NAME, CORTICAL_ATLAS_PATH=CORTICAL_400ROI_ATLAS_PATH)
output  = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
print(output.strip().decode())

# 8. Rank the atlas with missing ROIs

command = """module load afni; \
             cd {ATLAS_PATH}; \
             3dRank -prefix {ATLAS_NAME}.subcortical.ranked.nii.gz -input {ATLAS_NAME}.subcortical.nii.gz;""".format(ATLAS_PATH=SUBCORTICAL_ATLAS_PATH,ATLAS_NAME=SUBCORTICAL_ATLAS_NAME)
output  = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
print(output.strip().decode())

# 9. Create two additional text files with info about centroids and colors that we will need later
#
# * ```aal2.subcortical_order.txt```: contains ROI_ID, ROI_Name, RGB Color, Size

command = """cd {ATLAS_PATH}; \
             cat {ATLAS_NAME}.nii.txt | grep -e Thalamus -e Pallidum -e Caudate -e Putamen | awk -F '[ _]' '{{print $1"\taal2_"$3"H_Subcortical_"$2"\t255\t255\t0\t0"}}' > {ATLAS_NAME}.subcortical_order.txt;
             """.format(ATLAS_PATH=SUBCORTICAL_ATLAS_PATH,ATLAS_NAME=SUBCORTICAL_ATLAS_NAME)
output  = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
print(output.strip().decode())

# * ```aal2.subcortical.Centroid_RAS.csv```: contains ROI_ID, ROI_Name , Centroid coordinates

# ```bash
# # cd /data/SFIMJGC_Introspec/2023_fc_introspection/atlases/aal2
# module load afni
# # cat aal2.subcortical_order.txt | awk -F '\t' '{print $1","$2","}' > rm01.aal2.subcortical.Centroid_RAS.csv
# 3dCM -Icent -all_rois -Icent aal2.subcortical.nii.gz | grep -v '#' | tail -n 8 | awk '{print int($1)*-1","int($2)*-1","int($3)*-1}' > rm02.aal2.subcortical.Centroid_RAS.csv
# paste -d'\0' rm01.aal2.subcortical.Centroid_RAS.csv rm02.aal2.subcortical.Centroid_RAS.csv > aal2.subcortical.Centroid_RAS.csv
# sed -i '1s/^/ROI Label,ROI Name,R,A,S\n/' aal2.subcortical.Centroid_RAS.csv;
# # rm rm01.aal2.subcortical.Centroid_RAS.csv rm02.aal2.subcortical.Centroid_RAS.csv
# ```

# 10. Create Rank corrected Order and Centroid Files

path_to_order_file = osp.join(SUBCORTICAL_ATLAS_PATH,'aal2.subcortical_order.txt')
path_to_rank_file  = osp.join(SUBCORTICAL_ATLAS_PATH,'aal2.subcortical.ranked.nii.gz.rankmap.1D')
path_to_centroids_file = osp.join(SUBCORTICAL_ATLAS_PATH,'aal2.subcortical.Centroid_RAS.csv')
correct_ranked_atlas(path_to_order_file,path_to_centroids_file,path_to_rank_file)

# 11. Add corrected label table to the ranked version of the atlas

command = """module load afni; \
             cd {ATLAS_PATH}; \
             @MakeLabelTable -lab_file aal2.subcortical_order.ranked.txt 1 0 -labeltable aal2.subcortical_order.ranked.niml.lt -dset aal2.subcortical.ranked.nii.gz;""".format(ATLAS_PATH=SUBCORTICAL_ATLAS_PATH)
output  = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
print(output.strip().decode())

# 10. Create a dataframe with all the necessary info about this atlas
#
# We now create a single dataframe with all the info we need: roi number, roi label, network, hemisphere, colors codes and centroid position. We save this to disk so that it can be easily accessed by any other notebook

# Load the cetroid file for the ranked atlas in memory
centroids_info               = pd.read_csv(osp.join(ATLASES_DIR, SUBCORTICAL_ATLAS_NAME,'aal2.subcortical.Centroid_RAS.ranked.csv' ))
centroids_info['ROI Name']   = [label.split('aal2_')[1] for label in centroids_info['ROI Name']]
centroids_info['Hemisphere'] = [item.split('_')[0] for item in centroids_info['ROI Name']]
centroids_info['Network']    = [item.split('_')[1] for item in centroids_info['ROI Name']]
# Load the color info file for the ranked atlas in memory
color_info = pd.read_csv(osp.join(ATLASES_DIR, SUBCORTICAL_ATLAS_NAME,'aal2.subcortical_order.ranked.txt'),sep='\t', header=None)
# Combine all the useful columns into a single new dataframe
df         = pd.concat([centroids_info[['ROI Label','Hemisphere','Network','ROI Name','R','A','S']],color_info[[2,3,4]]], axis=1)
df.columns = ['ROI_ID','Hemisphere','Network','ROI_Name','pos_R','pos_A','pos_S','color_R','color_G','color_B']
df['RGB']  = [rgb2hex(r,g,b) for r,g,b in df.set_index('ROI_ID')[['color_R','color_G','color_B']].values]
# Save the new data frame to disk
df.to_csv(osp.join(ATLASES_DIR,SUBCORTICAL_ATLAS_NAME,'aal2.subcortical.ranked.roi_info.csv'), index=False)

# ***
# # 3. Combine Cortical and Subcortical Atlas
#
# 1. Create new folder for the combined atlas

if osp.exists(FB_400ROI_ATLAS_PATH):
    rmtree(FB_400ROI_ATLAS_PATH)
    print('++ WARNING: Removing pre-existing folder for combined atlas [%s]' % FB_400ROI_ATLAS_PATH)
os.makedirs(FB_400ROI_ATLAS_PATH)

# Downstream code has some expectations regarding ROI sorting:
#
# * All ROIs in the left hemisphere first, followed by all ROIs on the right hemisphere
# * Within each hemisphere, ROIs from the same network has contigous IDs
# * The subcortical regions will be added as its own network at the end of each hemisphere

# 2. Create un-ranked combined atlas NII file
# ```bash
# # Create a NIFTI files with all ROIs in order, but not contigous yet.
# # cd /data/SFIMJGC_Introspec/2023_fc_introspection/atlases/Schaefer2018_400Parcels_7Networks_AAL2
# 3dcalc -overwrite -a ../Schaefer2018_400Parcels_7Networks/orig/Schaefer2018_400Parcels_7Networks.nii.gz -expr 'within(a,0,200)*a' -prefix rm.Schaefer2018_400Parcels_7Networks.LH.nii.gz
# 3dcalc -overwrite -a ../Schaefer2018_400Parcels_7Networks/orig/Schaefer2018_400Parcels_7Networks.nii.gz -expr 'within(a,201,400)*a' -prefix rm.Schaefer2018_400Parcels_7Networks.RH.nii.gz
# 3dcalc -overwrite -a ../aal2/aal2.subcortical.ranked.nii.gz -expr [`cat ../aal2/aal2.subcortical_order.ranked.txt | grep LH | awk '{print "(a*equals(a,"$1"))"}' | tr -s '\n' '+' | sed 's/+$//g'`] -overwrite -prefix rm.aal2_subcortical.LH.nii
# 3dcalc -overwrite -a ../aal2/aal2.subcortical.ranked.nii.gz -expr [`cat ../aal2/aal2.subcortical_order.ranked.txt | grep RH | awk '{print "(a*equals(a,"$1"))"}' | tr -s '\n' '+' | sed 's/+$//g'`] -overwrite -prefix rm.aal2_subcortical.RH.nii
# 3dcalc -overwrite -a rm.Schaefer2018_400Parcels_7Networks.LH.nii.gz -b rm.aal2_subcortical.LH.nii -c rm.Schaefer2018_400Parcels_7Networks.RH.nii.gz -d rm.aal2_subcortical.RH.nii -expr 'a+step(b)*(1000+b)+step(c)*(2000+c)+step(d)*(3000+d)' -prefix rm.combined.nii.gz
# ```

# 3. Create un-ranked order file that matches the nii file just generated
#
# ```bash
# # Create order file that matches the nii above file.
# grep LH ../Schaefer2018_400Parcels_7Networks/orig/Schaefer2018_400Parcels_7Networks_order.txt > rm.combined_order.txt
# grep LH ../aal2/aal2.subcortical_order.ranked.txt                                             | awk '{print 1000+int($1)"\t"$2"\t"$3"\t"$4"\t"$5"\t"$6}' >> rm.combined_order.txt
# grep RH ../Schaefer2018_400Parcels_7Networks/orig/Schaefer2018_400Parcels_7Networks_order.txt | awk '{print 2000+int($1)"\t"$2"\t"$3"\t"$4"\t"$5"\t"$6}' >> rm.combined_order.txt
# grep RH ../aal2/aal2.subcortical_order.ranked.txt                                             | awk '{print 3000+int($1)"\t"$2"\t"$3"\t"$4"\t"$5"\t"$6}' >> rm.combined_order.txt
# sed -i 's/7Networks/8Networks/g' rm.combined_order.txt
# sed -i 's/aal2/8Networks/g'      rm.combined_order.txt
# ```

# 4. Create un-ranked Centroids file that matches the nii file two cells up
# ```bash
# # Create Centroid file that matches the nii file above.
# # echo "ROI Label,ROI Name,R,A,S" > rm.combined.Centroid_RAS.csv
# grep LH ../Schaefer2018_400Parcels_7Networks/orig/Schaefer2018_400Parcels_7Networks.Centroid_RAS.csv >> rm.combined.Centroid_RAS.csv
# grep LH ../aal2/aal2.subcortical.Centroid_RAS.ranked.csv                                        | awk -F ',' '{print 1000+int($1)","$2","$3","$4","$5}' >> rm.combined.Centroid_RAS.csv
# grep RH ../Schaefer2018_400Parcels_7Networks/orig/Schaefer2018_400Parcels_7Networks.Centroid_RAS.csv | awk -F ',' '{print 2000+int($1)","$2","$3","$4","$5}' >> rm.combined.Centroid_RAS.csv
# grep RH ../aal2/aal2.subcortical.Centroid_RAS.ranked.csv                                        | awk -F ',' '{print 3000+int($1)","$2","$3","$4","$5}' >> rm.combined.Centroid_RAS.csv
# sed -i 's/7Networks/8Networks/g' rm.combined.Centroid_RAS.csv
# sed -i 's/aal2/8Networks/g'      rm.combined.Centroid_RAS.csv
# ```
#
# 5.  Add the label information to the nifti file

command = """module load afni; \
             cd {ATLAS_PATH}; \
             @MakeLabelTable -lab_file rm.combined_order.txt 1 0 -labeltable rm.combined.niml.lt -dset rm.combined.nii.gz;""".format(ATLAS_PATH=FB_400ROI_ATLAS_PATH)
output  = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
print(output.strip().decode())

# 6. Rank the atlas with missing ROIs

command = """module load afni; \
             cd {ATLAS_PATH}; \
             3dRank -prefix Schaefer2018_400Parcels_7Networks_AAL2.nii.gz -input rm.combined.nii.gz;""".format(ATLAS_PATH=FB_400ROI_ATLAS_PATH,ATLAS_NAME=FB_400ROI_ATLAS_NAME)
output  = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
print(output.strip().decode())

# 6. Create Rank corrected Order and Centroid Files

path_to_order_file = osp.join(FB_400ROI_ATLAS_PATH,'rm.combined_order.txt')
path_to_rank_file  = osp.join(FB_400ROI_ATLAS_PATH,'Schaefer2018_400Parcels_7Networks_AAL2.nii.gz.rankmap.1D')
path_to_centroids_file = osp.join(FB_400ROI_ATLAS_PATH,'rm.combined.Centroid_RAS.csv')
correct_ranked_atlas(path_to_order_file,path_to_centroids_file,path_to_rank_file, new_atlas_name='Schaefer2018_400Parcels_7Networks_AAL2')

# 11. Add corrected label table to the ranked version of the atlas

command = """module load afni; \
             cd {ATLAS_PATH}; \
             @MakeLabelTable -lab_file {ATLAS_NAME}_order.ranked.txt 1 0 -labeltable {ATLAS_NAME}_order.ranked.niml.lt -dset {ATLAS_NAME}.nii.gz;
             """.format(ATLAS_PATH=FB_400ROI_ATLAS_PATH, ATLAS_NAME=FB_400ROI_ATLAS_NAME)
output  = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
print(output.strip().decode())

# 10. Create a dataframe with all the necessary info about this atlas
#
# We now create a single dataframe with all the info we need: roi number, roi label, network, hemisphere, colors codes and centroid position. We save this to disk so that it can be easily accessed by any other notebook

# Load the cetroid file for the ranked atlas in memory
centroids_info               = pd.read_csv(osp.join(ATLASES_DIR, FB_400ROI_ATLAS_NAME,'{ATLAS_NAME}.Centroid_RAS.ranked.csv'.format(ATLAS_NAME=FB_400ROI_ATLAS_NAME) ))
centroids_info['ROI Name']   = [label.split('8Networks_')[1] for label in centroids_info['ROI Name']]
centroids_info['Hemisphere'] = [item.split('_')[0] for item in centroids_info['ROI Name']]
centroids_info['Network']    = [item.split('_')[1] for item in centroids_info['ROI Name']]
# Load the color info file for the ranked atlas in memory
color_info = pd.read_csv(osp.join(ATLASES_DIR, FB_400ROI_ATLAS_NAME,'{ATLAS_NAME}_order.ranked.txt'.format(ATLAS_NAME=FB_400ROI_ATLAS_NAME)),sep='\t', header=None)
# Combine all the useful columns into a single new dataframe
df         = pd.concat([centroids_info[['ROI Label','Hemisphere','Network','ROI Name','R','A','S']],color_info[[2,3,4]]], axis=1)
df.columns = ['ROI_ID','Hemisphere','Network','ROI_Name','pos_R','pos_A','pos_S','color_R','color_G','color_B']
df['RGB']  = [rgb2hex(r,g,b) for r,g,b in df.set_index('ROI_ID')[['color_R','color_G','color_B']].values]
# Save the new data frame to disk
df.to_csv(osp.join(ATLASES_DIR,FB_400ROI_ATLAS_NAME,'{ATLAS_NAME}.roi_info.csv'.format(ATLAS_NAME=FB_400ROI_ATLAS_NAME)), index=False)

# 11. Clean-up atlas folder
#
# ```bash
# # cd /data/SFIMJGC_Introspec/2023_fc_introspection/atlases/Schaefer2018_400Parcels_7Networks_AAL2
# # rm rm.*
# # mv Schaefer2018_400Parcels_7Networks_AAL2.Centroid_RAS.ranked.csv Schaefer2018_400Parcels_7Networks_AAL2.Centroid_RAS.csv
# # mv Schaefer2018_400Parcels_7Networks_AAL2_order.ranked.niml.lt Schaefer2018_400Parcels_7Networks_AAL2_order.niml.lt
# # mv Schaefer2018_400Parcels_7Networks_AAL2_order.ranked.txt Schaefer2018_400Parcels_7Networks_AAL2_order.txt
# ```
