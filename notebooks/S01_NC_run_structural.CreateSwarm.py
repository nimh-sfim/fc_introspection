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
#     display_name: FC Instrospection py 3.10 | 2023b
#     language: python
#     name: fc_introspection_2023b_py310
# ---

# # Description - Create Swarm File to run structural pipeline on NC dataset
#
# This script creates the swarm file to run freesurfer on the NC dataset. 
#
# Becuase this dataset is linked to the MPI LEMON dataset, sometimes subjects have an anatomical, but on other occasions the anatomical needs to be grabbed from the LEMON dataset.
#
# In Notebooks/SNYCQ01_CleanDownloadedData we created a dataframe that contains the final list of subjects to be analyzed (i.e., only those with resting runs accompanied by SNYCQ) and for each of these subjects, the dataframe also contains the path of the anatomical for each subject.
#
# ***

import pandas as pd
import os.path as osp
import os
from datetime import datetime
import getpass

from utils.basics import PRJ_DIR, SCRIPTS_DIR, ANAT_PATHINFO_PATH

username = getpass.getuser()
print('++ INFO: user working now --> %s' % username)

# +
#user specific folders
#=====================
swarm_folder   = osp.join(PRJ_DIR,'SwarmFiles.{username}'.format(username=username))
logs_folder    = osp.join(PRJ_DIR,'Logs.{username}'.format(username=username))         
                          
swarm_path     = osp.join(swarm_folder,'S01_NC_run_structural.SWARM.sh')
# -

# create user specific folders if needed
# ======================================
if not osp.exists(swarm_folder):
    os.makedirs(swarm_folder)
if not osp.exists(logs_folder):
    os.makedirs(logs_folder)

# ***
# # 1. Load DataFrame with subject list and path to anatomical

anat_info = pd.read_csv(ANAT_PATHINFO_PATH, index_col='subject')

anat_info.head()

# # 2. Create Log Directory for swarm jobs

logdir_path = osp.join(logs_folder,'S01_NC_run_structural.logs')
if not osp.exists(logdir_path):
    os.mkdir(logdir_path)
    print("++ INFO: Log folder created [%s]" % logdir_path)

# ***
# # 3. Create Swarm File

# This will create a swarm file with one line call to S01_NC_run_structural.sh per subject. The inputs to that bash script are:
#
# * SBJ = subject ID
# * ANAT_PREFIX = 'ses-01' or 'ses-02' depending on where the anatomical data resides. This information will be used by ```structural.py``` and ```mp2rage.py``` within the lemon pipeline.
# * ANAT_PATH = folder containing the anatomical scans. They will be also used by the two pipeline files mentioned above

# Open the file
swarm_file = open(swarm_path, "w")
# Log the date and time when the SWARM file is created
swarm_file.write('#Create Time: %s' % datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
swarm_file.write('\n')
# Insert comment line with SWARM command
swarm_file.write('#swarm -f {swarm_path} -g 32 -t 32 --time 48:00:00 --logdir {logdir}'.format(swarm_path=swarm_path,logdir=logdir_path))
swarm_file.write('\n')
# Insert one line per subject
for sbj,row in anat_info.iterrows():
    anat_prefix = 'missing'
    if row['ses-01'] == True:
        anat_prefix = 'ses-01'
    if row['ses-02'] == True:
        anat_prefix = 'ses-02'
    swarm_file.write("export SBJ={sbj} ANAT_PREFIX={anat_prefix} ANAT_PATH={anat_path}; sh {scripts_folder}/S01_NC_run_structural.sh".format(sbj=sbj,
                                                                                                                                             anat_prefix=anat_prefix,
                                                                                                                                             anat_path=row['anat_path'],
                                                                                                                                             scripts_folder=SCRIPTS_DIR))
    swarm_file.write('\n')
swarm_file.close()


