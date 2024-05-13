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

# +
from utils.basics import RESOURCES_NIMARE_DIR, ATLASES_DIR, FB_400ROI_ATLAS_NAME, PRJ_DIR, SCRIPTS_DIR, RESOURCES_NBS_DIR
from utils.plotting import create_graph_from_matrix
import os
import os.path as osp
from nimare.extract import fetch_neurosynth
from nimare.io import convert_neurosynth_to_dataset
import nibabel as nib
from nilearn.image import load_img
from nilearn.plotting import plot_roi, plot_stat_map
from nilearn import masking
import numpy as np
import pandas as pd
from utils.basics import FB_400ROI_ATLAS_PATH
import subprocess
from nimare.decode import discrete
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import shutil
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from matplotlib.colors import LinearSegmentedColormap

import getpass
from datetime import datetime
from glob import glob
# -

import wordcloud
print(wordcloud.__version__)


def my_orange_color_func(dictionary):
    def my_orange_color_func_inner(word, font_size, position, orientation, random_state=None, **kwargs):
        freq_as_int = int(dictionary[word])
        color_list = sns.color_palette('Oranges',100).as_hex()
        return color_list[freq_as_int]
    return my_orange_color_func_inner
def my_blue_color_func(dictionary):
    def my_blue_color_func_inner(word, font_size, position, orientation, random_state=None, **kwargs):
        freq_as_int = int(dictionary[word])
        color_list = sns.color_palette('Blues',100).as_hex()
        return color_list[freq_as_int]
    return my_blue_color_func_inner


# # 1. NiMare Setup
# ## 1.1. Folder Setup

vocab = 'LDA50'
ATLAS_NAME = FB_400ROI_ATLAS_NAME

# +
VOCAB_DIR            = osp.join(RESOURCES_NIMARE_DIR,vocab)
METAMAPS_ORIG_DIR    = osp.join(VOCAB_DIR,"meta-analyses-orig")  # where to save meta-analysis maps
METAMAPS_RPI_DIR     = osp.join(VOCAB_DIR,"meta-analyses-RPI")  # where to save meta-analysis maps

ns_dset_path         = osp.join(VOCAB_DIR, f"neurosynth_dataset_{vocab}.pkl.gz")
ns_dset_mask_path    = osp.join(VOCAB_DIR, f"neurosynth_dataset_{vocab}_mask.nii")
#ns_dset_mask_path    = osp.join(METAMAPS_ORIG_DIR, f"neurosynth_dataset_{vocab}_mask.nii")
#lda_model_path       = osp.join(VOCAB_DIR, f'lda_model.pkl.gz')

print('++ INFO: Resource Folder for NiMare Analyses                              : %s' % RESOURCES_NIMARE_DIR)
print('++ INFO: Folder for this vocabulary                                       : %s' % VOCAB_DIR)
print('++ INFO: Folder for meta-maps in original orientation as written by NiMare: %s' % METAMAPS_ORIG_DIR)
print('++ INFO: Folder for meta-maps in RPI orientation (the one our data has)   : %s' % METAMAPS_RPI_DIR)
print('++ ------------------------------------------------------------------------')
print('++ INFO: Path for NeuroSynth Dataset in NiMare format                     : %s' % ns_dset_path)
print('++ INFO: Path for NeuroSynth Dataset mask                                 : %s' % ns_dset_mask_path)
#print('++ INFO: Path for locally trained LDA model.                              : %s' % lda_model_path)
# -

# Create Empty Output Folders
# ===========================
print("++ INFO: Setting up all necessary folders")
for folder_path in [RESOURCES_NIMARE_DIR, VOCAB_DIR, METAMAPS_ORIG_DIR, METAMAPS_RPI_DIR]:
    if osp.exists(folder_path):
        print(" + WARNING: Removing folder [%s]" % folder_path)
        shutil.rmtree(folder_path)
    print(" + INFO: Generating/Regenerating output folder [%s]" % folder_path)
    os.mkdir(folder_path)

# ## 1.2. Download Neurosynth 7 database
#
# First, we need to download the Neurosynth database (version 7) for the 50 Topic Vocabulary

# Download NeuroSynth database
print("++ INFO: Fetching neurosynth dataset for this vocabulary...")
files = fetch_neurosynth(data_dir=VOCAB_DIR, version="7", overwrite=False, vocab=vocab, source="abstract")

# ## 1.3. Convert Neurosynth Database to NiMare Dataset
#
# Next, we need to convert it into a format NiMare can understand

# %%time
# Convert to NiMare Dataset
neurosynth_db = files[0]
neurosynth_dset = convert_neurosynth_to_dataset(
        coordinates_file=neurosynth_db['coordinates'],
        metadata_file=neurosynth_db['metadata'],
        annotations_files=neurosynth_db['features'],
        )

# To avoid having to do these two steps continously, we will save the NiMare version of the NeuroSynth Database to disk. If we need it again, we just have to load this file.

# Save the dataset as a pickle file to the Resources directory
print ("++ INFO: Saving NeuroSynth Dataset to disk: %s" % ns_dset_path)
neurosynth_dset.save(ns_dset_path)

# As a sanity check, we print the labels for the first 10 topics and count how many topics in total are in the database.

# Extract Topic Names
topics_ORIG = neurosynth_dset.get_labels()
print('++ INFO: First few topics      : %s' % str(topics_ORIG[0:5]))
print('++ INFO: Total number of topics: %d' % len(topics_ORIG))

# ## 1.4. Extract Dset Mask
#
# This is necessary to ensure that any dataset that we decode is sitting on the same space and grid that NiMare expects

nib.save(neurosynth_dset.masker.mask_img,ns_dset_mask_path)
print('++ INFO: Neurosynth Dataset mask saved to disk: %s' % ns_dset_mask_path)

dset_mask = load_img(ns_dset_mask_path)
plot_roi(dset_mask, draw_cross=False);

command = f'''module load afni; \
              3dinfo -space -orient -header_name -header_line {ns_dset_mask_path}; \
        '''
output  = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
print(output.strip().decode())

# ## 1.5. Make a mask restricted to GM ribbon for decoding
#
# Although it is possible to decode using a full brain mask, this tends to not work so well becuase WM voxels cloud the correlations. 

# First, we create a version of the atlas (Which covers the GM ribbon) in the same grid as the mask file provided by nimare. which as you can see above is a full brain mask

FB_400ROI_ATLAS_ORIG_GRID_FILE   = osp.join(FB_400ROI_ATLAS_PATH,f'{ATLAS_NAME}.nii.gz')
FB_400ROI_ATLAS_NIMARE_GRID_FILE = osp.join(RESOURCES_NIMARE_DIR,f'{ATLAS_NAME}_NiMareGrid.nii.gz')
command = f'''module load afni; \
              3dresample -overwrite -rmode NN \
                         -input  {FB_400ROI_ATLAS_ORIG_GRID_FILE} \
                         -prefix {FB_400ROI_ATLAS_NIMARE_GRID_FILE} \
                         -master {ns_dset_mask_path}'''
output  = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
print(output.strip().decode())
print('++ New File: %s' % FB_400ROI_ATLAS_NIMARE_GRID_FILE)

atlas_img = load_img(FB_400ROI_ATLAS_NIMARE_GRID_FILE)
plot_roi(atlas_img, draw_cross=False)

# Create final mask for decoding (Neurosynth grid but restricted to the GM ribbon on which we computed the CPCAs)

NIMARE_DECODING_RIBBON_MASK = osp.join(RESOURCES_NIMARE_DIR,'NiMare_Decoding_Mask_GMribbon.nii.gz')
command = f'''module load afni; \
              3dcalc -overwrite \
                         -a {FB_400ROI_ATLAS_NIMARE_GRID_FILE} \
                         -b {ns_dset_mask_path} \
                         -expr "step(a)*b" \
                         -prefix {NIMARE_DECODING_RIBBON_MASK}'''
output  = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
print(output.strip().decode())

ribbon_mask_img = load_img(NIMARE_DECODING_RIBBON_MASK)
plot_roi(ribbon_mask_img, draw_cross=False)

# # 1.5 Generate Nifti Files for Topic Maps in original orientation
#
# It is advisable to have the topic maps available as .nii files in our local computer so that we can explore them in AFNI and also use them for computations outside the nimare environment.
#
# Because this is a time intensive step, we will do it via a set of batch jobs. Next, we create the swarm files needed for this step.

username = getpass.getuser()
print('++ INFO: user working now --> %s' % username)

# +
#user specific folders
#=====================
swarm_folder   = osp.join(PRJ_DIR,'SwarmFiles.{username}'.format(username=username))
logs_folder    = osp.join(PRJ_DIR,'Logs.{username}'.format(username=username))

swarm_path     = osp.join(swarm_folder,f'S15_NiMareTopics_{vocab}.SWARM.sh')
logdir_path    = osp.join(logs_folder, f'S15_NiMareTopics_{vocab}.logs')
print('++ INFO: swarm_path = %s' % swarm_path)
print('++ INFO: logs dir   = %s' % logdir_path)
# -

# create user specific folders if needed
# ======================================
if not osp.exists(swarm_folder):
    os.makedirs(swarm_folder)
    print('++ INFO: New folder for swarm files created [%s]' % swarm_folder)
if not osp.exists(logdir_path):
    os.makedirs(logdir_path)
    print('++ INFO: New folder for log files created [%s]' % logdir_path)

# +
# Open the file
swarm_file = open(swarm_path, "w")
# Log the date and time when the SWARM file is created
swarm_file.write('#Create Time: %s' % datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
swarm_file.write('\n')
# Insert comment line with SWARM command
swarm_file.write('#swarm -f {swarm_path} -g 32 -t 8 --partition quick,norm -b 5 --time 00:48:00 --merge-output --logdir {logdir_path}'.format(swarm_path=swarm_path,logdir_path=logdir_path))
swarm_file.write('\n')

# Insert one line per subject
for topic in topics_ORIG:
    topic = topic.replace(' ','-')
    swarm_file.write(f"export VOCAB={vocab} TOPIC={topic}; sh {SCRIPTS_DIR}/S15_NiMare_Create_TopicMaps.sh")
    swarm_file.write('\n')
swarm_file.close()
# -

print(swarm_path)

# Submit this swarm file to generate nii files for all topics in vocabulary.
#
# Once all jobs are completed, check that the topic maps are now available in ```/data/SFIMJGC_Introspec/2023_fc_introspecion/code/fc_intrisoection/nimare/LDA50/meta-analyses-orig```

command = f'''module load afni; \
              3dinfo -space -orient -header_name -header_line {ns_dset_mask_path} {VOCAB_DIR}/meta-analyses-orig/*language*specificity*; \
        '''
output  = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
print(output.strip().decode())

from nilearn.plotting import view_img_on_surf

view_img_on_surf(osp.join(VOCAB_DIR,'meta-analyses-orig','LDA50_abstract_weight__37_language_reading_word_z_desc-specificity.nii.gz'))

# + active=""
# FB_400ROI_ATLAS_RPI_GRID_CENTER_FILE   = osp.join(FB_400ROI_ATLAS_PATH,f'{ATLAS_NAME}.Rm2.nii.gz')
# command = f'''module load afni; \
#               3dinfo -space -orient -n4 -d3 -same_all_grid -header_name -header_line {ns_dset_mask_path} {FB_400ROI_ATLAS_RPI_GRID_CENTER_FILE}; \
#         '''
# output  = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
# print(output.strip().decode())

# + active=""
# FB_400ROI_ATLAS_LPI_NIMARE_GRID_FILE = osp.join(METAMAPS_ORIG_DIR,f'{ATLAS_NAME}_NiMareGrid.nii.gz')
# command = f'''module load afni; \
#               3dLRflip -LR -overwrite -prefix {FB_400ROI_ATLAS_LPI_NIMARE_GRID_FILE} {FB_400ROI_ATLAS_RPI_GRID_CENTER_FILE}; \
#               3drefit -orient LPI -space ORIG {FB_400ROI_ATLAS_LPI_NIMARE_GRID_FILE}; \
#               3dresample -overwrite -rmode NN \
#                          -input  {FB_400ROI_ATLAS_LPI_NIMARE_GRID_FILE} \
#                          -prefix {FB_400ROI_ATLAS_LPI_NIMARE_GRID_FILE} \
#                          -master {ns_dset_mask_path}'''
# output  = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
# print(output.strip().decode())
# #3dLRflip -LR -overwrite -prefix {FB_400ROI_ATLAS_LPI_NIMARE_GRID_FILE} {FB_400ROI_ATLAS_RPI_GRID_FILE};\
# #3drefit -orient LPI -space ORIG {FB_400ROI_ATLAS_LPI_NIMARE_GRID_FILE}; \
#

# + active=""
# command = f'''module load afni; \
#               3dinfo -space -orient -n4 -d3 -same_all_grid -header_name -header_line {ns_dset_mask_path} {FB_400ROI_ATLAS_LPI_NIMARE_GRID_FILE} {METAMAPS_ORIG_DIR}/*37*language*z_desc-speci*; \
#         '''
# output  = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
# print(output.strip().decode())

# + active=""
# plot_roi(FB_400ROI_ATLAS_LPI_NIMARE_GRID_FILE)
# -

# # 1.6 Flip them to RPI 
#
# Here, we are gonna do some "AFNI magic" to generate the same maps with a different orientation (RPI), so that they completely match our data. This is needed so that both AFNI and Python see the correct orientation when trying to match our data, with the NS maps
#
# ### First, let's flip the NS mask

ns_dset_mask_path_RPI = osp.join(METAMAPS_RPI_DIR,f'neurosynth_dataset_{vocab}_mask.nii')
command = f'''module load afni; \
              if [ ! -d {METAMAPS_RPI_DIR} ]; then mkdir {METAMAPS_RPI_DIR}; fi; \
              echo "hello"; \
              3dLRflip -LR -overwrite -prefix {ns_dset_mask_path_RPI} {ns_dset_mask_path}; \
              3drefit -orient RPI -space MNI {ns_dset_mask_path_RPI}; \
           '''
output  = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
print(output.strip().decode())

dset_mask = load_img(ns_dset_mask_path_RPI)
plot_roi(dset_mask, draw_cross=False);

command = f'''module load afni; \
              3dinfo -space -orient -header_name -header_line {ns_dset_mask_path_RPI}; \
        '''
output  = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
print(output.strip().decode())

# ### Second,  let's flip the maps using a bash script that we generate here first and then we run also through the notebook.
#
# Let's get a list of all the maps we want to transform. Becuase reverse-inference is based on the ```z_desc-specificity``` maps, we will only manipulate these to save time and disk space.

nimare_metamaps_orig        = sorted(glob(os.path.join(METAMAPS_ORIG_DIR, f"{vocab}_*z_desc-specificity.nii.gz")))
print(len(nimare_metamaps_orig))

#Path to script that will generate the corrected version of the meta-analytic maps
nimare_flip_metamaps_script_path = osp.join(SCRIPTS_DIR,'S15b_NiMare_Flip_TopicMaps.{vocab}.sh'.format(vocab=vocab))
print (nimare_flip_metamaps_script_path)

nimare_flip_metamaps_script = open(nimare_flip_metamaps_script_path, "w")
nimare_flip_metamaps_script.write('# Script to create flipped version of NiMare outputs\n')
nimare_flip_metamaps_script.write('module load afni\n')
nimare_flip_metamaps_script.write('\n')
nimare_flip_metamaps_script.write('set -e\n')
nimare_flip_metamaps_script.write('\n')
nimare_flip_metamaps_script.write('# Create output folder (if needed)\n')
nimare_flip_metamaps_script.write('if [ ! -d {out_folder} ]; then mkdir {out_folder}; fi\n\n'.format(out_folder=METAMAPS_RPI_DIR))
for orig_path in nimare_metamaps_orig:
    orig_file = osp.basename(orig_path)
    #orig_file = orig_file.replace(' ','\ ')
    new_path  = osp.join(METAMAPS_RPI_DIR,orig_file)
    #new_path  = new_path.replace(' ','-')
    nimare_flip_metamaps_script.write('3dLRflip -LR -overwrite -prefix "{out_file}" "{in_file}"\n'.format(in_file=orig_path, out_file=new_path))
    nimare_flip_metamaps_script.write('3drefit -orient RPI -space MNI "{in_file}"\n'.format(in_file=new_path))
    nimare_flip_metamaps_script.write('3dresample -overwrite -master {master_path} -input "{in_file}" -prefix "{out_file}"\n\n'.format(master_path = ns_dset_mask_path_RPI,
                                                                                                                               in_file = new_path,
                                                                                                                               out_file = new_path))
nimare_flip_metamaps_script.close()

# Now, open a terminal and run the script

command = f'''module load afni; \
              3dinfo -space -orient -header_name -header_line {VOCAB_DIR}/meta-analyses-RPI/*language*specificity*; \
        '''
output  = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
print(output.strip().decode())

view_img_on_surf(osp.join(VOCAB_DIR,'meta-analyses-RPI','LDA50_abstract_weight__37_language_reading_word_z_desc-specificity.nii.gz'))

# ### Third, let's create a GM restricted mask to be used in all decoding efforts (in RPI format)
#
# As we can see, although we have ensured the same orient and space, the grid is still differnet between our atlas and the nimare mask

FB_400ROI_ATLAS_ORIG_GRID_FILE   = osp.join(FB_400ROI_ATLAS_PATH,f'{ATLAS_NAME}.nii.gz')
command = f'''module load afni; \
              3dinfo -space -orient -n4 -d3 -same_all_grid -header_name -header_line {ns_dset_mask_path_RPI} {FB_400ROI_ATLAS_ORIG_GRID_FILE}; \
        '''
output  = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
print(output.strip().decode())

plot_roi(FB_400ROI_ATLAS_ORIG_GRID_FILE, draw_cross=False)

# To correct that, we need to run 3dresample using as input the atlas, and the RPI corrected version of the data

FB_400ROI_ATLAS_NIMARE_GRID_FILE = osp.join(METAMAPS_RPI_DIR,f'{ATLAS_NAME}_NiMareGrid.nii.gz')
command = f'''module load afni; \
              3dresample -overwrite -rmode NN \
                         -input  {FB_400ROI_ATLAS_ORIG_GRID_FILE} \
                         -prefix {FB_400ROI_ATLAS_NIMARE_GRID_FILE} \
                         -master {ns_dset_mask_path_RPI}'''
output  = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
print(output.strip().decode())

# After the correction, everything is now on the same grid, space and orientation

command = f'''module load afni; \
              3dinfo -space -orient -n4 -d3 -same_all_grid -header_name -header_line {ns_dset_mask_path_RPI} {FB_400ROI_ATLAS_NIMARE_GRID_FILE}; \
        '''
output  = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
print(output.strip().decode())

atlas_img = load_img(FB_400ROI_ATLAS_NIMARE_GRID_FILE)
plot_roi(atlas_img, draw_cross=False)

# ### Forth, we create the final decoding mask that only looks at GM voxels

NIMARE_DECODING_RIBBON_MASK = osp.join(METAMAPS_RPI_DIR,'NiMare_Decoding_Mask_GMribbon.nii.gz')
command = f'''module load afni; \
              3dcalc -overwrite \
                         -a {FB_400ROI_ATLAS_NIMARE_GRID_FILE} \
                         -b {ns_dset_mask_path_RPI} \
                         -expr "step(a)*b" \
                         -prefix {NIMARE_DECODING_RIBBON_MASK}'''
output  = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
print(output.strip().decode())

ribbon_mask_img = load_img(NIMARE_DECODING_RIBBON_MASK)
plot_roi(ribbon_mask_img, draw_cross=False)

# # 2. Extract ROIs with highest degree for both NBS contrasts
#
# # 2.1. Load Atlas Information so that we can annotate Pandas Dataframes

NBS_THRESHOLD        = 'NBS_3p1'

ATLASINFO_PATH = osp.join(ATLASES_DIR,ATLAS_NAME,'{ATLAS_NAME}.roi_info.csv'.format(ATLAS_NAME=ATLAS_NAME))
roi_info       = pd.read_csv(ATLASINFO_PATH)

# ## 2.2. Load NBS Results

NBS_constrasts = {}

# Load the connections that are significantly stronger for the contrast: $$Image-Pos-Others > Surr-Neg-Self$$

#data_f1GTf2 = np.loadtxt(f'/data/SFIMJGC_Introspec/2023_fc_introspection/code/fc_introspection/resources/nbs/{ATLAS_NAME}/NBS_CL02_Results/NBS_CL02_Image-Pos-Others_gt_Surr-Neg-Self.edge')
data_f1GTf2 = np.loadtxt(osp.join(RESOURCES_NBS_DIR,ATLAS_NAME,'All_Scans','NBS_CL02_Results',NBS_THRESHOLD,'NBS_CL02_Image-Pos-Others_gt_Surr-Neg-Self.edge'))
NBS_constrasts['f1GTf2'] = pd.DataFrame(data_f1GTf2,
                           index   = roi_info.set_index(['Hemisphere','Network','ROI_Name','ROI_ID','RGB']).index, 
                           columns = roi_info.set_index(['Hemisphere','Network','ROI_Name','ROI_ID','RGB']).index)
del data_f1GTf2

# Load the connections that are significantly stronger for the contrast: $$Surr-Neg-Self > Image-Pos-Others$$

#data_f2GTf1 = np.loadtxt(f'/data/SFIMJGC_Introspec/2023_fc_introspection/code/fc_introspection/resources/nbs/{ATLAS_NAME}/NBS_CL02_Results/NBS_CL02_Surr-Neg-Self_gt_Image-Pos-Others.edge')
data_f2GTf1 = np.loadtxt(osp.join(RESOURCES_NBS_DIR,ATLAS_NAME,'All_Scans','NBS_CL02_Results',NBS_THRESHOLD,'NBS_CL02_Surr-Neg-Self_gt_Image-Pos-Others.edge'))
NBS_constrasts['f2GTf1'] = pd.DataFrame(data_f2GTf1,
                           index   = roi_info.set_index(['Hemisphere','Network','ROI_Name','ROI_ID','RGB']).index, 
                           columns = roi_info.set_index(['Hemisphere','Network','ROI_Name','ROI_ID','RGB']).index)
del data_f2GTf1

# ## 2.3. Create Graph Models needed to compute Degree

# %%time
NBS_Gs,NBS_Gatts = {},{}
for contrast in NBS_constrasts.keys():
    # Create Graph Model
    NBS_Gs[contrast],NBS_Gatts[contrast]      = create_graph_from_matrix(NBS_constrasts[contrast])
    NBS_Gatts[contrast] =  NBS_Gatts[contrast].set_index(['Hemisphere','Network','ROI_Name','ROI_ID','RGB']).sort_index(level='ROI_ID')
N_rois = NBS_Gatts[contrast].shape[0]
print('++ INFO: Number of ROIs: %d' % N_rois)

# ## 2.4. Write Graph Metric-per-ROI Results to disk as NifTi Files

# We load the atlas (the version on the grid that aligns with NiMare) as a vector.

atlas_vector = masking.apply_mask(FB_400ROI_ATLAS_NIMARE_GRID_FILE, dset_mask)
print('++ INFO: Atlas Dimensions as a vector = %s' % str(atlas_vector.shape))

# %%time
for contrast in NBS_constrasts.keys():
    for metric in ['Degree','Eigenvector_Centrality','Page_Rank']:
        # Create Empty Vector with same dimensions as atlas
        output_vector = np.zeros(atlas_vector.shape)
        # For each ROI extract the Graph Metric of interest
        for hm,nw,roi_name,roi_id,rgb in NBS_Gatts[contrast].index:
            output_vector[atlas_vector==roi_id] = NBS_Gatts[contrast].loc[(hm,nw,roi_name,roi_id,rgb),metric]
        # Write to disk just in case we want to look at it later
        output_img = masking.unmask(output_vector,dset_mask)
        output_path = osp.join(RESOURCES_NIMARE_DIR,f'{NBS_THRESHOLD}_{contrast}_{metric}.nii')
        output_img.to_filename(output_path)
        print('++ INFO: File written to disk [%s]' % output_path)
        if metric == 'Degree':
            plot_stat_map(output_path, draw_cross=False)

# ## 2.5. Also write file for the top degree ROI for each contrast (this is the one we will decode)

for contrast in NBS_constrasts.keys():
    input_path = osp.join(RESOURCES_NIMARE_DIR,f'{NBS_THRESHOLD}_{contrast}_Degree.nii')
    output_path = osp.join(RESOURCES_NIMARE_DIR,f'{NBS_THRESHOLD}_{contrast}_Degree_TopROI.nii')
    input_img  = load_img(input_path)
    top_degree_value = int(input_img.get_fdata().max())
    command = f'''module load afni; \
                  3dcalc -overwrite -a {input_path} -expr "equals(a,{top_degree_value})" -prefix {output_path}'''
    output  = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
    print(output.strip().decode())

# # 3. Preparations for WordCloud & ROI plot generation
#
# Neurosynth topics and terms contain words that are not necessarily that meaningful when looking for relationships to cognitive processes. Examples of such words include those about anatomical structures, anatomical location, tissue types, imaging modalities, etc. We will remove those prior to generating wordclouds

my_stopwords = list(STOPWORDS) + ['resonance','magnetic','medial','lateral','dorsal','ventral','anterior','posterior','primary','secondary',
                                  'contralateral','prefrontal','temporal','occipital','parietal','frontal','network','cortex',
                                  'sii','mns','neuron','pre','md','gm','volume','matter','white','gray','hemispheric','state','mpfc','dmn','default',
                                  'fc','rest', 'temporoparietal','hippocampus','insula','premotor','supplementary','resting']

# In this study we work with the 50 Topic dictionary. Each topic is associated with a set of terms that appear in unison in the neuroimaing literature. The next cell gathers the 40 top terms associated with each topic. We will only pick only those assoicated with topics that show significant correlations with our ROIs later when generating wordclouds. For now we load them all and have then ready on a pandas dataframe.

path            = osp.join(VOCAB_DIR,'neurosynth',f'data-neurosynth_version-7_vocab-{vocab}_keys.tsv')
words_per_topic = pd.read_csv(path, sep='\t', header=None)
words_per_topic.index = neurosynth_dset.get_labels()
words_per_topic.index.name = 'Topic Label'
words_per_topic.columns = ['Topic ID','Unknown','Terms']
words_per_topic.head(5)

# Finally, we will want to show the ROI with the highest degree on each direction of the contrast. To make nilearn plot the countour in black color, we need to generate a fake colormap with black being the first color.

roi_cmap = LinearSegmentedColormap.from_list('black',['#000000','#ffffff'],10)

# # 4. ROI decoding on location with highest degree
#
# ## 4.1. Images-Pos-Others > Surr-Neg-Self
#
# ### 4.1.1. Load the target ROI

top_degree_roi_path =  osp.join(RESOURCES_NIMARE_DIR,f'{NBS_THRESHOLD}_f1GTf2_Degree_TopROI.nii')
top_degree_roi      = load_img(top_degree_roi_path)

f = plot_roi(top_degree_roi,draw_cross=False, display_mode='ortho', linewidths=3, cut_coords=[-5,-60,30], view_type='contours', cmap=roi_cmap)

f.savefig('./figures/S15a_f1GTf2_TopDegree_ROI.png')

# ### 4.1.2. Gather the studies with coordinates that overlap with the ROI

# Get studies with voxels in the mask
ids = neurosynth_dset.get_studies_by_mask(top_degree_roi)
print('++INFO: Number of studies that overlap with the ROI: %d stduies' % len(ids))

# ### 4.1.3. Decode using the Chi-Method

# Run the decoder
decoder = discrete.NeurosynthDecoder(u=0.05, correction='bonferroni')
decoder.fit(neurosynth_dset)
decoded_df = decoder.transform(ids=ids)

selected_topics = decoded_df[decoded_df['pReverse']<0.05].sort_values(by='probReverse', ascending=False)
print('++ List of topics that correlate significantly with the provided ROI (pBONF<0.05)')
selected_topics

# ### 4.1.4. Generate a wordcloud
#
# If we are not dealing with terms directly, which is the case with the topic dictionaries, we need to first create a dictionary with the invidual terms assoicated with each topic.
#
# That dictorionary will contain the term, and a weight that corresponds to the inverse rank of the term within the topic. As we consider only the top 40 terms associated with each topic (what Neurosynth makes available), weights will be integers in the range 1 to 40.

if vocab != 'terms':
    term_weights_per_topic={}
    for topic in words_per_topic.index:
        this_topic_words              = words_per_topic.loc[topic]['Terms']
        this_topic_words_top40        = this_topic_words.split(' ')[0:40][::-1]
        term_weights_per_topic[topic] = {word:weight+1 for weight,word in enumerate(this_topic_words_top40)}

# Next, to generate final weights per term used in the wordcloud formation, we will multiply each term by the reverse probability of the topic to which they belong. Then for each term we will compute the final weight as the sum of all such topic specific weights (just in case a term appears in more than one selected topic.)

freqs_df = pd.Series(dtype=float)
if vocab == 'terms':
    for term_long,row in selected_topics.iterrows():
        term = term_long.split('__')[1]
        term_prob = row['probReverse']
        if term in freqs_df.index:
            freqs_df[term] = freqs_df[term] + term_prob
        else:
            freqs_df[term] = term_prob
else:
    for topic in selected_topics.index:
        this_topic_prob = selected_topics.loc[topic,'probReverse']
        for word,weight in term_weights_per_topic[topic].items():
            if word in freqs_df.index:
                freqs_df[word] = freqs_df[word] + (this_topic_prob * weight)
            else:
                freqs_df[word] = (this_topic_prob * weight)

# Finally, we will select the top 30 terms for the wordcloud. Size of words will be directly related to the weights. In addition, to make sure color of the words is also associated with the weights, we need to do a bit more meddling so that we can crease a color scale that gives more emphasis (e.g., darker colors) to the terms with the highest weights, yet other words also have an intesnsity that allows us to read the words.

freqs_df.drop(my_stopwords,errors='ignore', inplace=True)
freqs_df  = freqs_df.sort_values(ascending=False)[0:30]

# Compute values constrained between 0 and 100 (ONLY FOR COLORSCALE PURPOSES)
freqs_arr = freqs_df.values
freqs_arr = freqs_arr.reshape(-1,1)
freqs_arr = MinMaxScaler((25,99)).fit_transform(freqs_arr)
freqs_df_color = pd.Series(freqs_arr.flatten(),index=freqs_df.index)

# As the wordcloud API takes as inputs dictonaries, we will transform the dataframes with the weights for word_size and word_color to python dictionary structures

freqs_dict = freqs_df.to_dict()
freqs_color_dict = freqs_df_color.to_dict()

wc = WordCloud(max_font_size=40,min_font_size=9, stopwords=set(my_stopwords),
                   contour_color='black', contour_width=3, 
                   background_color='white', color_func=my_orange_color_func(freqs_color_dict),
                   repeat=False).generate_from_frequencies(freqs_dict )
plt.imshow(wc,interpolation='bilinear')
plt.axis('off')
plt.tight_layout()
plt.savefig(f'./figures/S15a_WordCloud_{NBS_THRESHOLD}_f1GTf2_TopDegree_NeuosynthDeconding.png')

# ## 4.2. Surr-Neg-Self > Images-Pos-Others
#
# ### 4.2.1. Load the target ROI

top_degree_roi_path =  osp.join(RESOURCES_NIMARE_DIR,f'{NBS_THRESHOLD}_f2GTf1_Degree_TopROI.nii')
top_degree_roi      = load_img(top_degree_roi_path)

f = plot_roi(top_degree_roi,draw_cross=False, display_mode='ortho', linewidths=3, cut_coords=[-61,-36,33], view_type='contours', cmap=roi_cmap)

f.savefig(f'./figures/S15a_{NBS_THRESHOLD}_f2GTf1_TopDegree_ROI.png')

# ### 4.2.2. Gather the studies with coordinates that overlap with the ROI

# Get studies with voxels in the mask
ids = neurosynth_dset.get_studies_by_mask(top_degree_roi)
print('++INFO: Number of studies that overlap with the ROI: %d stduies' % len(ids))

# ### 4.2.3. Decode using the Chi-Method

# Run the decoder
decoder = discrete.NeurosynthDecoder(u=0.05, correction='bonferroni')
decoder.fit(neurosynth_dset)
decoded_df = decoder.transform(ids=ids)

selected_topics = decoded_df[decoded_df['pReverse']<0.05].sort_values(by='probReverse', ascending=False)
print('++ List of topics that correlate significantly with the provided ROI (pBONF<0.05)')
selected_topics

# ### 4.2.4. Generate a wordcloud
#
# If we are not dealing with terms directly, which is the case with the topic dictionaries, we need to first create a dictionary with the invidual terms assoicated with each topic.
#
# That dictorionary will contain the term, and a weight that corresponds to the inverse rank of the term within the topic. As we consider only the top 40 terms associated with each topic (what Neurosynth makes available), weights will be integers in the range 1 to 40.

if vocab != 'terms':
    term_weights_per_topic={}
    for topic in words_per_topic.index:
        this_topic_words              = words_per_topic.loc[topic]['Terms']
        this_topic_words_top40        = this_topic_words.split(' ')[0:40][::-1]
        term_weights_per_topic[topic] = {word:weight+1 for weight,word in enumerate(this_topic_words_top40)}

# Next, to generate final weights per term used in the wordcloud formation, we will multiply each term by the reverse probability of the topic to which they belong. Then for each term we will compute the final weight as the sum of all such topic specific weights (just in case a term appears in more than one selected topic.)

freqs_df = pd.Series(dtype=float)
if vocab == 'terms':
    for term_long,row in selected_topics.iterrows():
        term = term_long.split('__')[1]
        term_prob = row['probReverse']
        if term in freqs_df.index:
            freqs_df[term] = freqs_df[term] + term_prob
        else:
            freqs_df[term] = term_prob
else:
    for topic in selected_topics.index:
        this_topic_prob = selected_topics.loc[topic,'probReverse']
        for word,weight in term_weights_per_topic[topic].items():
            if word in freqs_df.index:
                freqs_df[word] = freqs_df[word] + (this_topic_prob * weight)
            else:
                freqs_df[word] = (this_topic_prob * weight)

# Finally, we will select the top 30 terms for the wordcloud. Size of words will be directly related to the weights. In addition, to make sure color of the words is also associated with the weights, we need to do a bit more meddling so that we can crease a color scale that gives more emphasis (e.g., darker colors) to the terms with the highest weights, yet other words also have an intesnsity that allows us to read the words.

freqs_df.drop(my_stopwords,errors='ignore', inplace=True)
freqs_df  = freqs_df.sort_values(ascending=False)[0:30]

# Compute values constrained between 0 and 100 (ONLY FOR COLORSCALE PURPOSES)
freqs_arr = freqs_df.values
freqs_arr = freqs_arr.reshape(-1,1)
freqs_arr = MinMaxScaler((25,99)).fit_transform(freqs_arr)
freqs_df_color = pd.Series(freqs_arr.flatten(),index=freqs_df.index)

# As the wordcloud API takes as inputs dictonaries, we will transform the dataframes with the weights for word_size and word_color to python dictionary structures

freqs_dict = freqs_df.to_dict()
freqs_color_dict = freqs_df_color.to_dict()

wc = WordCloud(max_font_size=40,min_font_size=9, stopwords=set(my_stopwords),
                   contour_color='black', contour_width=3, 
                   background_color='white', color_func=my_blue_color_func(freqs_color_dict),
                   repeat=False).generate_from_frequencies(freqs_dict )
plt.imshow(wc,interpolation='bilinear')
plt.axis('off')
plt.tight_layout()
plt.savefig(f'./figures/S14b_WordCloud_{NBS_THRESHOLD}_f2GTf1_TopDegree_NeuosynthDeconding.png')


