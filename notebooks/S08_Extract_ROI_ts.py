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

# # Description - Extract Representative ROI Timseries
#
# This notebook contians the code to extract representative timeseries for the two different atlas prepared in the previous notebook
#
# We rely on AFNI's program [```3dNetCorr```](https://afni.nimh.nih.gov/pub/dist/doc/program_help/3dNetCorr.html) to extract the represenative timeseries. This second step will be done via a swarm job.

import subprocess
import getpass
import os
import pandas as pd
from datetime import datetime
from shutil import rmtree
from utils.basics import CORTICAL_400ROI_ATLAS_PATH, CORTICAL_400ROI_ATLAS_NAME, SUBCORTICAL_ATLAS_PATH, SUBCORTICAL_ATLAS_NAME, FB_400ROI_ATLAS_NAME, FB_400ROI_ATLAS_PATH
from utils.basics import DATA_DIR, PRJ_DIR, SCRIPTS_DIR, ATLASES_DIR
from utils.basics import get_sbj_scan_list
import os.path as osp

#CORTICAL_ATLAS_NAME = CORTICAL_400ROI_ATLAS_NAME
#CORTICAL_ATLAS_PATH = CORTICAL_400ROI_ATLAS_PATH
ATLAS_NAME       = FB_400ROI_ATLAS_NAME
ATLAS_PATH       = FB_400ROI_ATLAS_PATH

# ***
# 1. Retrieve user ID

username = getpass.getuser()
print('++ INFO: user working now --> %s' % username)

# 2. Load list of scans that passed all QAs

sbj_list, scan_list, SNYCQ_data = get_sbj_scan_list('post_motion')

# + [markdown] tags=[]
# 3. Create output folder for static FC matrices
# -

for sbj in sbj_list:
    output_path = osp.join(DATA_DIR,'PrcsData',sbj,'preprocessed','func','pb06_staticFC')
    if not osp.exists(output_path):
        os.makedirs(output_path)

# 4. Create Swarm jobs

# +
#user specific folders
#=====================
swarm_folder   = osp.join(PRJ_DIR,f'SwarmFiles.{username}','S08')
logs_folder    = osp.join(PRJ_DIR,f'Logs.{username}','S08')

swarm_path     = osp.join(swarm_folder,f'S08_Extract_ROI_ts_{ATLAS_NAME}.SWARM.sh')
logdir_path    = osp.join(logs_folder, f'S08_Extract_ROI_ts_{ATLAS_NAME}.logs')
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
    swarm_file.write(f"export SBJ={sbj} RUN={run}; sh {SCRIPTS_DIR}/S08_ExtractROIts.sh {ATLAS_NAME} \n")
swarm_file.close()
# -
print(swarm_path)

next(os.walk('/data/SFIMJGC_Introspec/2023_fc_introspection/code/fc_introspection/resources/cpm/swarm_outputs/real/Schaefer2018_200Parcels_7Networks_AAL2/subject_aware/conf_not_residualized/spearman_sum/'))[1]


