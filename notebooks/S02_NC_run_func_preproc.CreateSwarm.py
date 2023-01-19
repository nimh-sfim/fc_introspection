# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.12.0
#   kernelspec:
#     display_name: FC Instrospection (Jan 2023)
#     language: python
#     name: fc_introspection
# ---

# # Description - Create Swarm File to run lsd functional pre-processing pipeline on NC dataset
#
# This script creates the swarm file to run the functional pre-processing pipeline on the lsd (NC) portion of the lemon dataset. 
#
# ***

import pandas as pd
import os.path as osp
import os
from datetime import datetime
import getpass
from utils.basics import PRJ_DIR, NOTEBOOKS_DIR, SCRIPTS_DIR, RESOURCES_DINFO_DIR, PREPROCESSING_NOTES_DIR
print('++ INFO: Project Dir:                  %s' % PRJ_DIR) 
print('++ INFO: Notebooks Dir:                %s' % NOTEBOOKS_DIR) 
print('++ INFO: Bash Scripts Dir:             %s' % SCRIPTS_DIR)
print('++ INFO: Resources (Dataset Info) Dir: %s' % RESOURCES_DINFO_DIR)
print('++ INFO: Pre-processing Notes Dir:     %s' % PREPROCESSING_NOTES_DIR)

username = getpass.getuser()
print('++ INFO: user working now --> %s' % username)

#user specific folders
#=====================
swarm_folder   = osp.join(PRJ_DIR,'SwarmFiles.{username}'.format(username=username))
logs_folder    = osp.join(PRJ_DIR,'Logs.{username}'.format(username=username))
logdir_path    = osp.join(logs_folder,'S02_NC_run_func_preproc.logs')

# create user specific folders if needed
# ======================================
if not osp.exists(swarm_folder):
    os.makedirs(swarm_folder)
    print('++ INFO: New folder for swarm files created [%s]' % swarm_folder)
if not osp.exists(logs_folder):
    os.makedirs(logs_folder)
    print('++ INFO: New folder for log files created [%s]' % logs_folder)

anat_info_path           = osp.join(RESOURCES_DINFO_DIR,'NC_anat_info.pkl')
bad_struct_subjects_path = osp.join(RESOURCES_DINFO_DIR,'NC_struc_fail_list.csv')
swarm_path               = osp.join(swarm_folder,'S02_NC_run_func_preproc.SWARM.sh')

# ***
# # 1. Load list of subjects with at least one rest run with accompanying SNYQ data

sbj_list = (pd.read_csv(osp.join(RESOURCES_DINFO_DIR,'NC_withSNYCQ_subjects.txt'), header=None)[0]).tolist()
print("++ INFO: Number of subjects: %s" % len(sbj_list))

# ***
# # 2. Load list of subjects that failed structural pre-processing

bad_struct_sbj_df   = pd.read_csv(osp.join(PREPROCESSING_NOTES_DIR,'NC_struct_fail_list.csv'))
bad_struct_sbj_list = list(bad_struct_sbj_df['Subject'].values)
bad_struct_sbj_df.head()
print("++ INFO: Number of subjects with incomplete structural pre-processing:                          %d subjects" % len(bad_struct_sbj_list))
print("++ INFO: Number of rest scans that will be removed due to incomplete structural pre-processing: %d scans " % bad_struct_sbj_df['func_scans'].sum())

# ***
# # 3. Don't attempt functional pre-processing on subjects that failed the structural
#
# If the structual pre-processing failed for a subject, we will not be able to complete our analysis. For that reason, we will not attempt functional pre-processing of scans from subjects with failed anatomical scans

sbj_list = [sbj for sbj in sbj_list if sbj not in bad_struct_sbj_list]
print('++ INFO: Number of subjects for which we will attempt functional pre-processing: %d' % len(sbj_list))

# ***
# # 2. Create Log Directory for swarm jobs

if not osp.exists(logdir_path):
    os.mkdir(logdir_path)
    print("++ INFO: Log folder created [%s]" % logdir_path)

# ***
# ### 2. Create Swarm File

# +
# Open the file
swarm_file = open(swarm_path, "w")
# Log the date and time when the SWARM file is created
swarm_file.write('#Create Time: %s' % datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
swarm_file.write('\n')
# Insert comment line with SWARM command
swarm_file.write('#swarm -f {swarm_path} -g 32 -t 32 --time 32:00:00 --logdir {logdir_path}'.format(swarm_path=swarm_path, logdir_path=logdir_path))
swarm_file.write('\n')

# Insert one line per subject
for sbj in sbj_list:
    swarm_file.write("export SBJ={sbj}; sh {scripts_folder}/S02_NC_run_func_preproc.sh".format(sbj=sbj,scripts_folder=SCRIPTS_DIR))
    swarm_file.write('\n')
swarm_file.close()
# -
print('++ INFO: Swarm file available at: %s' % swarm_path)

