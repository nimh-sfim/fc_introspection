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

# # Description
#
# This notebook create the swarm jobs to run CPM on 100 iterations over the real data and 10,000 iterations over randomized data. 
#
# This happens separately for each question in the SNYCQ

import os.path as osp
import os
from datetime import datetime
import getpass
from utils.basics import get_sbj_scan_list
from utils.basics import PRJ_DIR, SCRIPTS_DIR, RESOURCES_CPM_DIR, DATA_DIR, FB_ATLAS_NAME
from cpm.cpm import read_fc_matrices

CPM_NITERATIONS      = 100       # Number of iterations on real data (to evaluate robustness against fold generation)
CPM_NULL_NITERATIONS = 10000     # Number of iterations used to build a null distribution
CORR_TYPE            = 'pearson' # Correlation type to use on the edge-selection step
E_SUMMARY_METRIC     = 'sum'     # How to summarize across selected edges on the final model
E_THR_P              = 0.01      # Threshold used on the edge-selection step

sbj_list, scan_list, snycq = get_sbj_scan_list(when='post_motion', return_snycq=True)

behaviors = list(snycq.columns)
print(behaviors)

username = getpass.getuser()
print('++ INFO: user working now --> %s' % username)

# ***
#
# ## 0. Prepare all necessary data for the CPM
#
# 1. Create resources folder for CPM analyses

if not osp.exists(RESOURCES_CPM_DIR):
    print('++ INFO: Creating resources folder for CPM analyses [%s]' % RESOURCES_CPM_DIR)
    os.makedirs(RESOURCES_CPM_DIR)

# 2. Load list of scans that passed all QAs

sbj_list, scan_list, snycq_df = get_sbj_scan_list(when='post_motion', return_snycq=True)

# 3. Load FC data into memory

fc_data = read_fc_matrices(scan_list,DATA_DIR,FB_ATLAS_NAME)

# 4. Save FC data in vectorized form for all scans into a single file for easy access for batch jobs

out_path = osp.join(RESOURCES_CPM_DIR,'fc_data.csv')
fc_data.to_csv(out_path)
print('++ INFO: FC data saved to disk [%s]' % out_path)

# 5. Save SNYCQ in the cpm resources folder

out_path = osp.join(RESOURCES_CPM_DIR,'behav_data.csv')
snycq_df.to_csv(out_path)
print('++ INFO: Behavioral data saved to disk [%s]' % out_path)

# ***
# ## 1. Swarm Jobs for the real data

# We will generate separate swarm files per question. Similarly we will separate the swarm jobs that are for computations on real data and those that are for the generation of the null distribution.

#user specific folders
#=====================
swarm_folder = osp.join(PRJ_DIR,'SwarmFiles.{username}'.format(username=username))
logs_folder  = osp.join(PRJ_DIR,'Logs.{username}'.format(username=username))
swarm_path,logdir_path={},{} 
for behavior in behaviors:    
    swarm_path[behavior]  = osp.join(swarm_folder,'S15_CPM_{beh}.SWARM.sh'.format(beh=behavior))
    logdir_path[behavior] = osp.join(logs_folder, 'S15_CPM_{beh}.logs'.format(beh=behavior))

# create user specific folders if needed
# ======================================
if not osp.exists(swarm_folder):
    os.makedirs(swarm_folder)
    print('++ INFO: New folder for swarm files created [%s]' % swarm_folder)
for behavior in behaviors:
    if not osp.exists(logdir_path[behavior]):
        os.makedirs(logdir_path[behavior])
        print('++ INFO: New folder for log files created [%s]' % logdir_path[behavior])

for behavior in behaviors:
    # Open the file
    swarm_file = open(swarm_path[behavior], "w")
    # Log the date and time when the SWARM file is created
    swarm_file.write('#Create Time: %s' % datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
    swarm_file.write('\n')
    # Insert comment line with SWARM command
    swarm_file.write('#swarm -f {swarm_path} -g 8 -t 8 -b 10 --time 00:15:00 --partition quick,norm --logdir {logdir_path}'.format(swarm_path=swarm_path[behavior],logdir_path=logdir_path[behavior]))
    swarm_file.write('\n')
    for n_iter in range(CPM_NITERATIONS):
        out_dir = osp.join(RESOURCES_CPM_DIR,'results',behavior,CORR_TYPE,E_SUMMARY_METRIC)
        if not osp.exists(out_dir):
            os.makedirs(out_dir)
        swarm_file.write("export BEHAV_PATH={behav_path} FC_PATH={fc_path} OUT_DIR={output_dir} BEHAVIOR={behavior} NUM_FOLDS={k} NUM_ITER={n_iter} CORR_TYPE={corr_type} E_SUMMARY_METRIC={e_summary_metric} E_THR_P={e_thr_p} VERBOSE=True RANDOMIZE_BEHAVIOR=False; sh {scripts_folder}/S15_cpm_batch.sh".format(scripts_folder = SCRIPTS_DIR,
                           behav_path       = osp.join(RESOURCES_CPM_DIR,'behav_data.csv'),
                           fc_path          = osp.join(RESOURCES_CPM_DIR,'fc_data.csv'),
                           output_dir       = out_dir,
                           behavior         = behavior,
                           k                = 10,
                           n_iter           = n_iter + 1,
                           corr_type        = CORR_TYPE,
                           e_summary_metric = E_SUMMARY_METRIC,
                           e_thr_p          = E_THR_P))
        swarm_file.write('\n')
    swarm_file.close()

# ***
# ## 2. Swarm jobs for the Null Distributions

#user specific folders
#=====================
swarm_folder = osp.join(PRJ_DIR,'SwarmFiles.{username}'.format(username=username))
logs_folder  = osp.join(PRJ_DIR,'Logs.{username}'.format(username=username))
swarm_path,logdir_path={},{} 
for behavior in behaviors:    
    swarm_path[behavior]  = osp.join(swarm_folder,'S15_CPM_null_{beh}.SWARM.sh'.format(beh=behavior))
    logdir_path[behavior] = osp.join(logs_folder, 'S15_CPM_null_{beh}.logs'.format(beh=behavior))

# create user specific folders if needed
# ======================================
if not osp.exists(swarm_folder):
    os.makedirs(swarm_folder)
    print('++ INFO: New folder for swarm files created [%s]' % swarm_folder)
for behavior in behaviors:
    if not osp.exists(logdir_path[behavior]):
        os.makedirs(logdir_path[behavior])
        print('++ INFO: New folder for log files created [%s]' % logdir_path[behavior])

for behavior in behaviors:
    # Open the file
    swarm_file = open(swarm_path[behavior], "w")
    # Log the date and time when the SWARM file is created
    swarm_file.write('#Create Time: %s' % datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
    swarm_file.write('\n')
    # Insert comment line with SWARM command
    swarm_file.write('#swarm -f {swarm_path} -g 8 -t 8 -b 50 --time 00:04:30 --partition quick,norm --logdir {logdir_path}'.format(swarm_path=swarm_path[behavior],logdir_path=logdir_path[behavior]))
    swarm_file.write('\n')
    for n_iter in range(CPM_NULL_NITERATIONS):
        out_dir = osp.join(RESOURCES_CPM_DIR,'null_distribution',behavior,CORR_TYPE,E_SUMMARY_METRIC)
        if not osp.exists(out_dir):
            os.makedirs(out_dir)
        swarm_file.write("export BEHAV_PATH={behav_path} FC_PATH={fc_path} OUT_DIR={output_dir} BEHAVIOR={behavior} NUM_FOLDS={k} NUM_ITER={n_iter} CORR_TYPE={corr_type} E_SUMMARY_METRIC={e_summary_metric} E_THR_P={e_thr_p} VERBOSE=True RANDOMIZE_BEHAVIOR=True; sh {scripts_folder}/S15_cpm_batch.sh".format(scripts_folder = SCRIPTS_DIR,
                           behav_path       = osp.join(RESOURCES_CPM_DIR,'behav_data.csv'),
                           fc_path          = osp.join(RESOURCES_CPM_DIR,'fc_data.csv'),
                           output_dir       = out_dir,
                           behavior         = behavior,
                           k                = 10,
                           n_iter           = n_iter + 1,
                           corr_type        = CORR_TYPE,
                           e_summary_metric = E_SUMMARY_METRIC,
                           e_thr_p          = E_THR_P))
        swarm_file.write('\n')
    swarm_file.close()