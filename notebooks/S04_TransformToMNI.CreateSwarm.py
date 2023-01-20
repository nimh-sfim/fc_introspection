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
# This script creates the SWARM file to run the pipeline that will transform the data to MNI space, which is necessary for the following step of this project.

# +
import pandas as pd
import os.path as osp
import os
from datetime import datetime
import getpass

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

swarm_path     = osp.join(swarm_folder,'S04_TransformToMNI.SWARM.sh')
logdir_path    = osp.join(logs_folder, 'S04_TransformToMNI.logs')
# -

# create user specific folders if needed
# ======================================
if not osp.exists(swarm_folder):
    os.makedirs(swarm_folder)
    print('++ INFO: New folder for swarm files created [%s]' % swarm_folder)
if not osp.exists(logs_folder):
    os.makedirs(logs_folder)
    print('++ INFO: New folder for log files created [%s]' % logs_folder)

# # 1. Load list of scans that completed struct and func pre-processing and have low motion

sbj_list, scan_list, SNYCQ_data = get_sbj_scan_list('post_motion')

# ***
# # 2. Create SWARM file
#
# This will create a swarm file with one line call per subject. The inputs to that bash script are:
#
# * SBJ = subject ID

# +
# Open the file
swarm_file = open(swarm_path, "w")
# Log the date and time when the SWARM file is created
swarm_file.write('#Create Time: %s' % datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
swarm_file.write('\n')
# Insert comment line with SWARM command
swarm_file.write('#swarm -f {swarm_path} -g 32 -t 32 --partition quick,norm --logdir {logdir_path}'.format(swarm_path=swarm_path,logdir_path=logdir_path))
swarm_file.write('\n')

for sbj in sbj_list:
    swarm_file.write("export SBJ={sbj}; sh {scripts_folder}/S04_TransformToMNI.sh".format(sbj=sbj, scripts_folder = SCRIPTS_DIR))
    swarm_file.write('\n')
swarm_file.close()
