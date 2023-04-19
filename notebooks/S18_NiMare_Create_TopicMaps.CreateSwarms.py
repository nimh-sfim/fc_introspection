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

import os.path as osp
import shutil
import os
import nimare
from nimare.extract import fetch_neurosynth
from nimare.io import convert_neurosynth_to_dataset
from glob import glob
import numpy as np
import pickle

print(nimare.__version__)

# # 1. Folder Setup

PRJDIR = "/data/SFIMJGC_Introspec/2023_fc_introspection"
vocab = 'LDA400'

# +
RESOURCE_NIMARE_DIR  = osp.join(PRJDIR,'nimare')
VOCAB_DIR            = osp.join(RESOURCE_NIMARE_DIR,vocab)
METAMAPS_ORIG_DIR    = osp.join(VOCAB_DIR,"meta-analyses-orig")  # where to save meta-analysis maps
METAMAPS_RPI_DIR     = osp.join(VOCAB_DIR,"meta-analyses-RPI")  # where to save meta-analysis maps

ns_dset_path         = os.path.join(VOCAB_DIR, f"neurosynth_dataset_{vocab}.pkl.gz")
print(ns_dset_path)
# -

# Create Empty Output Folders
# ===========================
print("++ INFO: Setting up all necessary folders")
for folder_path in [VOCAB_DIR, METAMAPS_ORIG_DIR]:
    if osp.exists(folder_path):
        print(" + WARNING: Removing folder [%s]" % folder_path)
        shutil.rmtree(folder_path)
    print(" + INFO: Generating/Regenerating output folder [%s]" % folder_path)
    os.mkdir(folder_path)

# # 2. Download Neurosynth 7 database

# Download NeuroSynth database
print("++ INFO: Fetching neurosynth dataset for this vocabulary...")
files = fetch_neurosynth(data_dir=VOCAB_DIR, version="7", overwrite=False, vocab=vocab, source="abstract")

# # 3. Convert Neurosynth Database to NiMare Dataset

# %%time
# Convert to NiMare Dataset
neurosynth_db = files[0]
neurosynth_dset = convert_neurosynth_to_dataset(
        coordinates_file=neurosynth_db['coordinates'],
        metadata_file=neurosynth_db['metadata'],
        annotations_files=neurosynth_db['features'],
        )

# Save the dataset as a pickle file to the Resources directory
print (" + Saving dataset to %s" % ns_dset_path)
neurosynth_dset.save(ns_dset_path)

# Extract Topic Names
topics_ORIG = neurosynth_dset.get_labels()
print(topics_ORIG[0:10])
print(len(topics_ORIG))

# ***

# # 4. Train LDA Model
#
# > Call S20s for 50 and 400 topics. This takes days to run and about 100GB of memory. Can be run on spersist node with 24 cpus using the NUMEXPR_MAX_THREADS environment variable.

lda_model_path = f'/data/SFIMJGC_Introspec/prj2021_dyneusr/Resources_NiMare/{vocab}/lda_model.pkl.gz'
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


