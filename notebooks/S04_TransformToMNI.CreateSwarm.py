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

# # Description - Create Swarm File to run transformation to MNI pipeline on the preprocessed data
#
# This script creates the SWARM file to run the pipeline that will transform the data to MNI space, which is necessary for the following step of this project.

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

# # 1. Load list of scans that completed struct and func pre-processing and have low motion

sbj_list, scan_list, SNYCQ_data = get_sbj_scan_list('post_motion')

# # 2. Create Output Folder for all subjects

for sbj in sbj_list:
    output_path = osp.join(DATA_DIR,'PrcsData',sbj,'preprocessed','func','pb05_mni')
    if not osp.exists(output_path):
        os.makedirs(output_path)

# ***
# # 3. Create SWARM file
#
# This will create a swarm file with one line call per subject. The inputs to that bash script are:
#
# * SBJ = subject ID

# +
#user specific folders
#=====================
swarm_folder   = osp.join(PRJ_DIR,'SwarmFiles.{username}'.format(username=username))
logs_folder    = osp.join(PRJ_DIR,'Logs.{username}'.format(username=username))

swarm_path     = osp.join(swarm_folder,'S04_TransformToMNI.pass01.SWARM.sh')
logdir_path    = osp.join(logs_folder, 'S04_TransformToMNI.pass01.logs')
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
swarm_file.write('#swarm -f {swarm_path} -g 32 -t 32 --partition quick,norm --logdir {logdir_path}'.format(swarm_path=swarm_path,logdir_path=logdir_path))
swarm_file.write('\n')

# Insert one line per subject
for sbj,run in scan_list:
    run = run[-2:] + "_" + run[12:18]
    swarm_file.write("export SBJ={sbj} RUN={RUN}; sh {scripts_folder}/S04_TransformToMNI.pass01.sh".format(sbj=sbj, RUN=run, scripts_folder = SCRIPTS_DIR))
    swarm_file.write('\n')
swarm_file.close()
# -

# By the end of these jobs, we will have two new files in ```DATA_DIR/PrcsData/<SBJ>/preprocessed/func/```
#
# * ```pb05_mni/<SCAN_ID>/rest2mni.nii.gz``` MNI Version of the motion corrected resting-state scan.
# * ```pb05_mni/<SCAN_ID>/rest_mean_2mni.nii.gz``` MNI Version of the temporal mean of the file above. 
#
# Becuase those files are very large (~2GB per scan), we decided to trim the corners of the files that contain no brain tissue. This required the following additional steps:
#
# 1. Create a common grid that would accomodate all scans
#
# 2. Cut the scans to be on that grid.
#
# The following cells help us accomplish these two tasks

# # 4. Compute common small size grid (only brain tissue)

command = """module load afni; \
             sh {PRJ_DIR}/code/fc_introspection/bash/S04_TransformToMNI.pass02.sh""".format(PRJ_DIR=PRJ_DIR)
output  = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
print(output.strip().decode())

# >**NOTE**: Automask sometimes leaves a bit of skull on the left side of the brain. We will manually correct this error on all_mean.mask.boxed.nii.gz using AFNI.

# # 5. Enforce new grid on files generated during step 3

#user specific folders
#=====================
swarm_path     = osp.join(swarm_folder,'S04_TransformToMNI.pass03.SWARM.sh')
logdir_path    = osp.join(logs_folder, 'S04_TransformToMNI.pass03.logs')

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
swarm_file.write('#swarm -f {swarm_path} -g 32 -t 32 --partition quick,norm --logdir {logdir_path}'.format(swarm_path=swarm_path,logdir_path=logdir_path))
swarm_file.write('\n')

# Insert one line per subject
for sbj,run in scan_list:
    run = run[-2:] + "_" + run[12:18]
    swarm_file.write("export SBJ={sbj} RUN={RUN}; sh {scripts_folder}/S04_TransformToMNI.pass03.sh".format(sbj=sbj, RUN=run, scripts_folder = SCRIPTS_DIR))
    swarm_file.write('\n')
swarm_file.close()
