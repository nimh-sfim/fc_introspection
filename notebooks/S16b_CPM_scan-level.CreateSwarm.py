#!/usr/bin/env python
# coding: utf-8

# # Description: Scan-Level Cross-Valudation CPM | Swarm jobs for real and null distributions
# 
# This notebook create the swarm jobs to run the **scan-level cross-validation** CPM on 100 iterations over the real data and 10,000 iterations over randomized data. 
# 
# This happens separately for each question in the SNYCQ
# 
# > NOTE: The results associated with these analyses are reported only in the form of a supplementary figure.
# 
# > NOTE: This notebook assumes you already run ```S16a_CPM_subject_aware.CreateSwarm.ipynb```. This is needed to ensure FC matrices, prediction targets and motion estimates are available for the CPM jobs.

# In[1]:


import os.path as osp
import os
from datetime import datetime
import getpass
#from utils.basics import get_sbj_scan_list
from utils.basics import PRJ_DIR, SCRIPTS_DIR, RESOURCES_CPM_DIR, FB_400ROI_ATLAS_NAME
#from utils.io import read_fc_matrices
import pandas as pd


# In[2]:


CPM_NITERATIONS      = 100             # Number of iterations on real data (to evaluate robustness against fold generation)
CPM_NULL_NITERATIONS = 10000           # Number of iterations used to build a null distribution
CORR_TYPE            = 'pearson'       # Correlation type to use on the edge-selection step
E_SUMMARY_METRIC     = 'sum'           # How to summarize across selected edges on the final model
E_THR_P              = 0.01            # Threshold used on the edge-selection step
E_THR_R              = None
SPLIT_MODE           = 'basic'         # Split mode for cross validation
MODEL_TYPE           = CORR_TYPE+'_'+E_SUMMARY_METRIC
CONFOUNDS            = 'conf_residualized' # Options: conf_residualized, conf_not_residualized
ATLAS_NAME           = FB_400ROI_ATLAS_NAME


# In[3]:


username = getpass.getuser()
print('++ INFO: user working now --> %s' % username)


# ***
# 
# # 1. **Scan-level** Cross Validation CPM
# 

# Get list of prediction targets

# In[4]:


pred_targets_path = osp.join(RESOURCES_CPM_DIR,'behav_data.csv')
behav_df          = pd.read_csv(pred_targets_path, index_col=[0,1])
targets           = list(behav_df.columns)
print('++ INFO: Prediction Targets: %s' % str(targets))
print('++ INFO: Number of prediction targets: %d' % len(targets))


# 
# Generate log folders and folder to contain the different swarm files

# In[5]:


#user specific folders
#=====================
swarm_folder = osp.join(PRJ_DIR,'SwarmFiles.{username}'.format(username=username),f'S16_CPM_{SPLIT_MODE}')
logs_folder  = osp.join(PRJ_DIR,'Logs.{username}'.format(username=username),f'S16_CPM_{SPLIT_MODE}.logs')
swarm_path,logdir_path={},{}

if not osp.exists(swarm_folder):
    os.makedirs(swarm_folder)
    print('++ INFO: New folder for swarm files created [%s]' % swarm_folder)

if not osp.exists(logs_folder):
    os.makedirs(logs_folder)
    print('++ INFO: New folder for log files created [%s]' % logs_folder)

for TARGET in targets:    
    swarm_path[TARGET]  = osp.join(swarm_folder,'S16_CPM-{atlas}-real-{sm}-{conf}-{mt}-{target}.SWARM.sh'.format(atlas=ATLAS_NAME,sm=SPLIT_MODE,conf=CONFOUNDS,mt=MODEL_TYPE, target=TARGET.replace(' ','')))
    logdir_path[TARGET] = osp.join(logs_folder, 'S16_CPM-{atlas}-real-{sm}-{conf}-{mt}-{target}.logs'.format(atlas=ATLAS_NAME,sm=SPLIT_MODE,conf=CONFOUNDS,mt=MODEL_TYPE, target=TARGET))


# In[6]:


# create specific folders if needed
# ======================================
for TARGET in targets:
    if not osp.exists(logdir_path[TARGET]):
        os.makedirs(logdir_path[TARGET])
        print('++ INFO: New folder for log files created [%s]' % logdir_path[TARGET])


# Create one swarm file per prediction target

# In[7]:


for TARGET in targets:
    # Open the file
    swarm_file = open(swarm_path[TARGET], "w")
    # Log the date and time when the SWARM file is created
    swarm_file.write('#Create Time: %s' % datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
    swarm_file.write('\n')
    # Insert comment line with SWARM command
    swarm_file.write('#swarm -f {swarm_path} -g 8 -t 8 -b 10 --time 00:24:00 --partition quick,norm --logdir {logdir_path}'.format(swarm_path=swarm_path[TARGET],logdir_path=logdir_path[TARGET]))
    swarm_file.write('\n')
    for n_iter in range(CPM_NITERATIONS):
        out_dir = osp.join(RESOURCES_CPM_DIR,'swarm_outputs','real',ATLAS_NAME,SPLIT_MODE,CONFOUNDS,MODEL_TYPE,TARGET)
        if not osp.exists(out_dir):
            print("++ INFO: Creating output dir %s" % out_dir)
            os.makedirs(out_dir)
        swarm_file.write("export BEHAV_PATH={behav_path} FC_PATH={fc_path} OUT_DIR={output_dir} BEHAVIOR={behavior} NUM_FOLDS={k} NUM_ITER={n_iter} CORR_TYPE={corr_type} E_SUMMARY_METRIC={e_summary_metric} E_THR_R={e_thr_r} E_THR_P={e_thr_p} SPLIT_MODE={split_mode} VERBOSE=True RANDOMIZE_BEHAVIOR=False CONFOUNDS={confounds} CONFOUNDS_PATH={confounds_path}; sh {scripts_folder}/S16_cpm_batch.sh".format(scripts_folder = SCRIPTS_DIR,
                           behav_path       = osp.join(RESOURCES_CPM_DIR,'behav_data.csv'),
                           fc_path          = osp.join(RESOURCES_CPM_DIR,f'fc_data_{ATLAS_NAME}.csv'),
                           confounds_path   = osp.join(RESOURCES_CPM_DIR,'confounds.csv'),                        
                           output_dir       = out_dir,
                           behavior         = TARGET,
                           k                = 10,
                           n_iter           = n_iter + 1,
                           corr_type        = CORR_TYPE,
                           split_mode       = SPLIT_MODE,
                           e_summary_metric = E_SUMMARY_METRIC,
                           e_thr_r          = E_THR_R,
                           e_thr_p          = E_THR_P,
                           confounds        = CONFOUNDS == 'conf_residualized'))
        swarm_file.write('\n')
    swarm_file.close()
    print('++ INFO: Swarm file created for target %s: %s' % (TARGET, swarm_path[TARGET]))


# Next, jobs need to be submitted from a terminal.
# 
# Once all the jobs have successfully completed, you should run the following command to compile all the outputs into a single file.
# 
# ```bash
# conda generic_2025a
# 
# # Compile together the results over the 100 real permutations
# python /data/SFIMJGC_Introspec/2023_fc_introspection/code/fc_introspection/notebooks/S16b_GatherSwarmResults.py \
#    -i /data/SFIMJGC_Introspec/2023_fc_introspection/code/fc_introspection/resources/cpm/swarm_outputs/real/Schaefer2018_400Parcels_7Networks_AAL2/basic/conf_residualized/pearson_sum/ \
#    -o /data/SFIMJGC_Introspec/2023_fc_introspection/code/fc_introspection/resources/cpm/real-Schaefer2018_400Parcels_7Networks_AAL2-basic-conf_residualized-pearson_sum.pkl \
#    -n 100
# ```

# ***
# ## 2. **Scan-level** Cross Validation CPM | Null Distributions
# 
# Create folders for logs and swarm files

# In[8]:


for TARGET in targets:    
    swarm_path[TARGET]  = osp.join(swarm_folder,'S16_CPM-{atlas}-null-{sm}-{conf}-{mt}-{target}.SWARM.sh'.format(atlas=ATLAS_NAME,sm=SPLIT_MODE,conf=CONFOUNDS,mt=MODEL_TYPE, target=TARGET))
    logdir_path[TARGET] = osp.join(logs_folder, 'S16_CPM-{atlas}-null-{sm}-{conf}-{mt}-{target}.logs'.format(atlas=ATLAS_NAME,sm=SPLIT_MODE,conf=CONFOUNDS,mt=MODEL_TYPE, target=TARGET))


# In[9]:


# create specific folders if needed
# ======================================
for TARGET in targets:
    if not osp.exists(logdir_path[TARGET]):
        os.makedirs(logdir_path[TARGET])
        print('++ INFO: New folder for log files created [%s]' % logdir_path[TARGET])


# Create one swarm file for each target

# In[10]:


for TARGET in targets:
    # Open the file
    swarm_file = open(swarm_path[TARGET], "w")
    # Log the date and time when the SWARM file is created
    swarm_file.write('#Create Time: %s' % datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
    swarm_file.write('\n')
    # Insert comment line with SWARM command
    swarm_file.write('#swarm -f {swarm_path} -g 8 -t 8 -b 15 --time 00:16:00 --partition quick,norm --logdir {logdir_path}'.format(swarm_path=swarm_path[TARGET],logdir_path=logdir_path[TARGET]))
    swarm_file.write('\n')
    for n_iter in range(CPM_NULL_NITERATIONS):
        out_dir = osp.join(RESOURCES_CPM_DIR,'swarm_outputs','null',ATLAS_NAME,SPLIT_MODE,CONFOUNDS,MODEL_TYPE,TARGET)
        if not osp.exists(out_dir):
            os.makedirs(out_dir)
        swarm_file.write("export BEHAV_PATH={behav_path} FC_PATH={fc_path} OUT_DIR={output_dir} BEHAVIOR={behavior} NUM_FOLDS={k} NUM_ITER={n_iter} CORR_TYPE={corr_type} E_SUMMARY_METRIC={e_summary_metric} E_THR_R={e_thr_r} E_THR_P={e_thr_p} SPLIT_MODE={split_mode} VERBOSE=True RANDOMIZE_BEHAVIOR=True CONFOUNDS={confounds} CONFOUNDS_PATH={confounds_path}; sh {scripts_folder}/S16_cpm_batch.sh".format(scripts_folder = SCRIPTS_DIR,
                           behav_path       = osp.join(RESOURCES_CPM_DIR,'behav_data.csv'),
                           fc_path          = osp.join(RESOURCES_CPM_DIR,f'fc_data_{ATLAS_NAME}.csv'),
                           confounds_path   = osp.join(RESOURCES_CPM_DIR,'confounds.csv'), 
                           output_dir       = out_dir,
                           behavior         = TARGET,
                           k                = 10,
                           n_iter           = n_iter + 1,
                           corr_type        = CORR_TYPE,
                           split_mode       = SPLIT_MODE,
                           e_summary_metric = E_SUMMARY_METRIC,
                           e_thr_r          = E_THR_R,
                           e_thr_p          = E_THR_P,
                           confounds        = CONFOUNDS == 'conf_residualized'))
        swarm_file.write('\n')
    swarm_file.close()
    print('++ INFO: Swarm file created for target %s: %s' % (TARGET, swarm_path[TARGET]))


# Submit all jobs
# 
# As jobs complete, it is important to ensure they all did successfully. The following code call should help in finding what (if any) jobs did not finish correctly, so that you re-attempt to run such jobs. This usually happens due to time constrains.
# 
# ```bash
# conda generic_2025a
# 
# # Compile together the results over the 10000 null permutations
# python /data/SFIMJGC_Introspec/2023_fc_introspection/code/fc_introspection/notebooks/S16b_GatherSwarmResults.py \
#    -i /data/SFIMJGC_Introspec/2023_fc_introspection/code/fc_introspection/resources/cpm/swarm_outputs/null/Schaefer2018_400Parcels_7Networks_AAL2/basic/conf_residualized/pearson_sum/ \
#    -o /data/SFIMJGC_Introspec/2023_fc_introspection/code/fc_introspection/resources/cpm/null-Schaefer2018_400Parcels_7Networks_AAL2-basic-conf_residualized-pearson_sum.pkl \
#    -n 10000 -T -t Factor1
# ```
# 
# Once you have ensured all jobs have finished correctly, the following code must be run to gather all outputs and save them on a single pkl file
# 
# ```bash
# python /data/SFIMJGC_Introspec/2023_fc_introspection/code/fc_introspection/notebooks/S16b_GatherSwarmResults.py  \
#    -i /data/SFIMJGC_Introspec/2023_fc_introspection/code/fc_introspection/resources/cpm/swarm_outputs/null/Schaefer2018_400Parcels_7Networks_AAL2/basic/conf_residualized/pearson_sum/ \
#    -o /data/SFIMJGC_Introspec/2023_fc_introspection/code/fc_introspection/resources/cpm/null-Schaefer2018_400Parcels_7Networks_AAL2-basic-conf_residualized-pearson_sum.pkl \
#    -n 10000
# ```
# 
# At this point, it is advisable to delete the swarm output folders given that all required information is not on the pkl files.
# 
# ```bash
# rm -rf /data/SFIMJGC_Introspec/2023_fc_introspection/code/fc_introspection/resources/cpm/swarm_outputs/
# ```
