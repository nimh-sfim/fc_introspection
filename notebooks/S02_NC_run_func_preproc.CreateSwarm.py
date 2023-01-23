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
from utils.basics import PRJ_DIR, SCRIPTS_DIR, RESOURCES_DINFO_DIR
from utils.basics import get_sbj_scan_list
print('++ INFO: Project Dir:                  %s' % PRJ_DIR) 
print('++ INFO: Bash Scripts Dir:             %s' % SCRIPTS_DIR)
print('++ INFO: Resources (Dataset Info) Dir: %s' % RESOURCES_DINFO_DIR)

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

swarm_path               = osp.join(swarm_folder,'S02_NC_run_func_preproc.SWARM.sh')

# ***
#
# # 1. Gather information about what scans/subjects completed structural pre-processing

sbj_list, scan_list, SNYCQ_data = get_sbj_scan_list('post_struct')

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

