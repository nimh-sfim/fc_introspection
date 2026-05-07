#!/usr/bin/env python
# coding: utf-8

# # Description - Create Swarm File to extrtact GM, WM and CSF tissue masks
# 
# This script will call AFNI program `3dSeg` on each subject's T1 already in MNI space to extract masks for GM, WM and CSF.

# In[7]:


import pandas as pd
import os.path as osp
import os
from datetime import datetime
import getpass
import subprocess

from utils.basics import get_sbj_scan_list

from utils.basics import PRJ_DIR, DATA_DIR, SCRIPTS_DIR
print('++ INFO: Project Dir:                  %s' % PRJ_DIR) 
print('++ INFO: Bash Scripts Dir:             %s' % SCRIPTS_DIR)
print('++ INFO: Data Dir:                     %s' % DATA_DIR)


# In[8]:


username = getpass.getuser()
print('++ INFO: user working now --> %s' % username)


# # 1. Load list of scans that completed struct and func pre-processing and have low motion

# In[9]:


sbj_list, scan_list, SNYCQ_data = get_sbj_scan_list('post_motion')


# ***
# # 2. Create SWARM file
# 
# This will create a swarm file with one line call per subject. The inputs to that bash script are:
# 
# * SBJ = subject ID

# In[10]:


#user specific folders
#=====================
swarm_folder   = osp.join(PRJ_DIR,'SwarmFiles.{username}'.format(username=username))
logs_folder    = osp.join(PRJ_DIR,'Logs.{username}'.format(username=username))

swarm_path     = osp.join(swarm_folder,'S05_SegmentT1.SWARM.sh')
logdir_path    = osp.join(logs_folder, 'S05_SegmentT1.logs')


# In[11]:


# create user specific folders if needed
# ======================================
if not osp.exists(swarm_folder):
    os.makedirs(swarm_folder)
    print('++ INFO: New folder for swarm files created [%s]' % swarm_folder)
if not osp.exists(logdir_path):
    os.makedirs(logdir_path)
    print('++ INFO: New folder for log files created [%s]' % logdir_path)


# In[14]:


# Open the file
swarm_file = open(swarm_path, "w")
# Log the date and time when the SWARM file is created
swarm_file.write('#Create Time: %s' % datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
swarm_file.write('\n')
# Insert comment line with SWARM command
swarm_file.write('#swarm -f {swarm_path} -g 32 -t 32 -b 4 --time 00:30:00 --partition quick,norm --logdir {logdir_path}'.format(swarm_path=swarm_path,logdir_path=logdir_path))
swarm_file.write('\n')

# Insert one line per subject
for sbj in sbj_list:
    swarm_file.write("export SBJ={sbj}; sh {scripts_folder}/S05_SegmentT1.sh".format(sbj=sbj, scripts_folder = SCRIPTS_DIR))
    swarm_file.write('\n')
swarm_file.close()

