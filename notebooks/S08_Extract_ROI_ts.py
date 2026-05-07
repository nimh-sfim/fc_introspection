#!/usr/bin/env python
# coding: utf-8

# # Description - Extract Representative ROI Timseries
# 
# This notebook contians the code to extract representative timeseries for all ROIs in our atlas of interest.
# 
# We rely on AFNI's program [```3dNetCorr```](https://afni.nimh.nih.gov/pub/dist/doc/program_help/3dNetCorr.html) to extract the represenative timeseries. This second step will be done via a swarm job.

# In[1]:


import getpass
import os
import pandas as pd
from datetime import datetime
from utils.basics import FB_400ROI_ATLAS_NAME, FB_400ROI_ATLAS_PATH
from utils.basics import DATA_DIR, PRJ_DIR, SCRIPTS_DIR
from utils.basics import get_sbj_scan_list
import os.path as osp


# In[2]:


ATLAS_NAME       = FB_400ROI_ATLAS_NAME
ATLAS_PATH       = FB_400ROI_ATLAS_PATH
print(ATLAS_NAME)


# ***
# 1. Retrieve user ID

# In[3]:


username = getpass.getuser()
print('++ INFO: user working now --> %s' % username)


# 2. Load list of scans that passed all QAs

# In[4]:


sbj_list, scan_list, SNYCQ_data = get_sbj_scan_list('post_motion')


# 3. Create output folder for static FC matrices

# In[5]:


for sbj in sbj_list:
    output_path = osp.join(DATA_DIR,'PrcsData',sbj,'preprocessed','func','pb06_staticFC')
    if not osp.exists(output_path):
        os.makedirs(output_path)


# 4. Create Swarm jobs

# In[6]:


#user specific folders
#=====================
swarm_folder   = osp.join(PRJ_DIR,f'SwarmFiles.{username}','S08')
logs_folder    = osp.join(PRJ_DIR,f'Logs.{username}','S08')

swarm_path     = osp.join(swarm_folder,f'S08_Extract_ROI_ts_{ATLAS_NAME}.SWARM.sh')
logdir_path    = osp.join(logs_folder, f'S08_Extract_ROI_ts_{ATLAS_NAME}.logs')


# In[7]:


# create user specific folders if needed
# ======================================
if not osp.exists(swarm_folder):
    os.makedirs(swarm_folder)
    print('++ INFO: New folder for swarm files created [%s]' % swarm_folder)
if not osp.exists(logdir_path):
    os.makedirs(logdir_path)
    print('++ INFO: New folder for log files created [%s]' % logdir_path)


# In[8]:


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


# In[9]:


print(swarm_path)

