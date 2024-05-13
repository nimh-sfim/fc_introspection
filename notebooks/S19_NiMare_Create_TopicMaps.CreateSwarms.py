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
#     display_name: FC Introspection (Jan 2023)
#     language: python
#     name: fc_introspection
# ---

import os.path as osp
import shutil
import os
import nimare
from nimare.extract import fetch_neurosynth
from nimare.io import convert_neurosynth_to_dataset
from glob import glob
import numpy as np
import pickle
from utils.basics import PRJ_DIR, RESOURCES_NIMARE_DIR

print(nimare.__version__)

# # 1. Folder Setup

vocab = 'LDA50'

# +
#RESOURCE_NIMARE_DIR  = osp.join(PRJ_DIR,'nimare')
VOCAB_DIR            = osp.join(RESOURCES_NIMARE_DIR,vocab)
METAMAPS_ORIG_DIR    = osp.join(VOCAB_DIR,"meta-analyses-orig")  # where to save meta-analysis maps
METAMAPS_RPI_DIR     = osp.join(VOCAB_DIR,"meta-analyses-RPI")  # where to save meta-analysis maps

ns_dset_path         = osp.join(VOCAB_DIR, f"neurosynth_dataset_{vocab}.pkl.gz")
lda_model_path       = osp.join(VOCAB_DIR, f'lda_model.pkl.gz')

print('++ INFO: Resource Folder for NiMare Analyses                              : %s' % RESOURCES_NIMARE_DIR)
print('++ INFO: Folder for this vocabulary                                       : %s' % VOCAB_DIR)
print('++ INFO: Folder for meta-maps in original orientation as written by NiMare: %s' % METAMAPS_ORIG_DIR)
print('++ INFO: Folder for meta-maps in RPI orientation (the one our data has)   : %s' % METAMAPS_RPI_DIR)
print('++ ------------------------------------------------------------------------')
print('++ INFO: Path for NeuroSynth Dataset in NiMare format                     : %s' % ns_dset_path)
print('++ INFO: Path for locally trained LDA model.                              : %s' % lda_model_path)
# -

# Create Empty Output Folders
# ===========================
print("++ INFO: Setting up all necessary folders")
for folder_path in [VOCAB_DIR, METAMAPS_ORIG_DIR, METAMAPS_RPI_DIR]:
    if osp.exists(folder_path):
        print(" + WARNING: Removing folder [%s]" % folder_path)
        shutil.rmtree(folder_path)
    print(" + INFO: Generating/Regenerating output folder [%s]" % folder_path)
    os.mkdir(folder_path)

# # 2. Download Neurosynth 7 database
#
# First, we need to download the Neurosynth database (version 7) for the 400 Topic Vocabulary

# Download NeuroSynth database
print("++ INFO: Fetching neurosynth dataset for this vocabulary...")
files = fetch_neurosynth(data_dir=VOCAB_DIR, version="7", overwrite=False, vocab=vocab, source="abstract")

# # 3. Convert Neurosynth Database to NiMare Dataset
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
print('++ INFO: First few topics      : %s' % str(topics_ORIG[0:10]))
print('++ INFO: Total number of topics: %d' % len(topics_ORIG))

# # 4. Train LDA Model
#
# The topic model downloaded from NeuroSynth does not include the weigths associated with how each term relates to the topic. If we want these (which can be quite useful for generating wordclouds), then we would have to train our own Latent Dirichlet allocation (LDA) topic model using NiMare.
#
# This is a very computing and memory intensive process. I was able to run it using spersist nodes with 100GB of memory and setting ```NUMEXPR_MAX_THREADS=24```. For the 50 topic model it takes several hours, for the 400 topic model it takes a few days.
#
# Here is what to do to run the models
#
# ```bash
# # Once you are on a tmux/no machine terminal in an spersist node with the suggested configuration, do the following:
#
# # Enter the notebook directory
# # cd /data/SFIMJGC_Introspec/2023_fc_introspection/code/fc_introspection/notebooks/
# # Set NUMEXPR_MAX_THREADS to use 24 cpus
# export NUMEXPR_MAX_THREADS=24
# # Active the correct conda environment
# source /data/SFIMJGC_HCP7T/Apps/miniconda38/etc/profile.d/conda.sh && conda activate fc_introspection
# # Run the LDA model for the 50 topics
# python ./S18_NiMare_Compute_LDA50_Models.py
# # Run the LDA model for the 400 topics
# python ./S18_NiMare_Compute_LDA400_Models.py
# ```
#
# By then end, you should have two new files:
#
# * ```/data/SFIMJGC_Introspec/2023_fc_introspection/nimare/LDA50/lda_model_LDA50.pkl.gz```
# * ```/data/SFIMJGC_Introspec/2023_fc_introspection/nimare/LDA400/lda_model_LDA400.pkl.gz```
#
# Execution time for LDA50
#
# ```
# real 341m48.019s
# user 1061m37.913s
# sys 26m45.867s
# ```
#
# Execution time for LDA400
#
# ```
#
# ```

# ## 4.1. Add new topic definitions to the NeuroSynth Database object
#
# Next, we will add the new topic models to the NeuroSynth dataset. What this means is that we will have a second version of the NeuroSynth Database object that contains not only the original topics (those available in the Neurosynth website) but also a novel version of the topics generated locally by running NiMare implementation of the LDA algorithm. 
#
# Later in the code, we will be able to work with one or the other.

lda_model_path = osp.join(VOCAB_DIR,f'lda_model_{vocab}.pkl.gz')
with open(lda_model_path,'rb') as f:
    model_results = pickle.load(f)

neurosynth_EXTRA_dset = model_results['new_dset']

topics       = neurosynth_dset.get_labels()
topics_EXTRA = neurosynth_EXTRA_dset.get_labels()
print(len(topics), len(topics_EXTRA))

# Save the dataset as a pickle file to the Resources directory
ns_dset_EXTRA_path         = os.path.join(VOCAB_DIR, f"neurosynth_dataset_{vocab}_EXTRA.pkl.gz")
print (" + Saving dataset to %s" % ns_dset_EXTRA_path)
neurosynth_EXTRA_dset.save(ns_dset_EXTRA_path)

# # 5. Generate Topic Maps

# +
import pandas as pd
import os.path as osp
import os
from datetime import datetime
import getpass
import subprocess

from utils.basics import get_sbj_scan_list

from utils.basics import PRJ_DIR, DATA_DIR, SCRIPTS_DIR #NOTEBOOKS_DIR, RESOURCES_DINFO_DIR, PREPROCESSING_NOTES_DIR, 
print('++ INFO: Project Dir:                  %s' % PRJ_DIR) 
#print('++ INFO: Notebooks Dir:                %s' % NOTEBOOKS_DIR) 
print('++ INFO: Bash Scripts Dir:             %s' % SCRIPTS_DIR)
#print('++ INFO: Resources (Dataset Info) Dir: %s' % RESOURCES_DINFO_DIR)
#print('++ INFO: Pre-processing Notes Dir:     %s' % PREPROCESSING_NOTES_DIR)
print('++ INFO: Data Dir:                     %s' % DATA_DIR)
# -

username = getpass.getuser()
print('++ INFO: user working now --> %s' % username)

# +
#user specific folders
#=====================
swarm_folder   = osp.join(PRJ_DIR,'SwarmFiles.{username}'.format(username=username))
logs_folder    = osp.join(PRJ_DIR,'Logs.{username}'.format(username=username))

swarm_path     = osp.join(swarm_folder,f'S19_NiMareTopics_{vocab}.SWARM.sh')
logdir_path    = osp.join(logs_folder, f'S19_NiMareTopics_{vocab}.logs')
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
swarm_file.write('#swarm -f {swarm_path} -g 32 -t 8 --partition quick,norm -b 5 --time 00:48:00 --logdir {logdir_path}'.format(swarm_path=swarm_path,logdir_path=logdir_path))
swarm_file.write('\n')

# Insert one line per subject
for topic in topics_EXTRA:
    topic = topic.replace(' ','-')
    swarm_file.write(f"export VOCAB={vocab} TOPIC={topic}; sh {SCRIPTS_DIR}/S19_NiMare_Create_TopicMaps.sh")
    swarm_file.write('\n')
swarm_file.close()
# -

# # 5. Flip them to RPI
#
# > Need code to generate the master file

# +
# Path to script that will generate the corrected version of the meta-analytic maps
nimare_flip_metamaps_script_path = osp.join(SCRIPTS_DIR,'STEP19_NiMare_Flip_TopicMaps.{vocab}.sh'.format(vocab=vocab))

# Path to fMRI Dataset to be used as reference when resampling. We pick the final data in MNI space from one random subject. All other subjects should be in the same space/grid
master_path = '/data/SFIMJGC_Introspec/2023_fc_introspection/nimare/NiMare_Decoding_Mask_GMribbon_2023.nii.gz'
# -

nimare_metamaps_orig        = sorted(glob(os.path.join(METAMAPS_ORIG_DIR, f"{vocab}_*z_desc-specificity.nii.gz")))
print(len(nimare_metamaps_orig))

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
    nimare_flip_metamaps_script.write('3dresample -overwrite -master {master_path} -input "{in_file}" -prefix "{out_file}"\n\n'.format(master_path = master_path,
                                                                                                                               in_file = new_path,
                                                                                                                               out_file = new_path))
nimare_flip_metamaps_script.close()

nimare_flip_metamaps_script_path


