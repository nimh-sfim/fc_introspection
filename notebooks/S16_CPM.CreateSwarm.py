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

# +
import os.path as osp
import os
from datetime import datetime
import getpass
from utils.basics import get_sbj_scan_list
from utils.basics import PRJ_DIR, SCRIPTS_DIR, RESOURCES_CPM_DIR, DATA_DIR, FB_400ROI_ATLAS_NAME, FB_200ROI_ATLAS_NAME, ATLASES_DIR
from cpm.cpm import read_fc_matrices

import numpy as np
import pandas as pd
import hvplot.pandas
# -

CPM_NITERATIONS      = 100       # Number of iterations on real data (to evaluate robustness against fold generation)
CPM_NULL_NITERATIONS = 10000     # Number of iterations used to build a null distribution
CORR_TYPE            = 'pearson'      # Correlation type to use on the edge-selection step
E_SUMMARY_METRIC     = 'sum'           # How to summarize across selected edges on the final model
E_THR_P              = 0.01            # Threshold used on the edge-selection step
E_THR_R              = None
SPLIT_MODE           = 'subject_aware' # Split mode for cross validation
MODEL_TYPE           = CORR_TYPE+'_'+E_SUMMARY_METRIC
CONFOUNDS            = 'conf_residualized'
ATLAS_NAME           = FB_400ROI_ATLAS_NAME

username = getpass.getuser()
print('++ INFO: user working now --> %s' % username)

# ***
#
# # 1. Scan-Level CPM
#
# ## 1.1. Prepare Inputs and Folders
#
# 1. Create resources folder for CPM analyses

if not osp.exists(RESOURCES_CPM_DIR):
    print('++ INFO: Creating resources folder for CPM analyses [%s]' % RESOURCES_CPM_DIR)
    os.makedirs(RESOURCES_CPM_DIR)

# 2. Load list of scans that passed all QAs

sbj_list, scan_list, snycq_df = get_sbj_scan_list(when='post_motion', return_snycq=True)

# 3. Add Factors

factorization_path = './mlt/output/factorization/factorization_fulldata_confound.npz'
factorization_results = np.load(factorization_path)
W = pd.DataFrame(factorization_results['W'],index=snycq_df.index, columns=['Factor1', 'Factor2'])

behav_df = pd.concat([snycq_df,W],axis=1)
behav_df.head(5)

# 4. Get list of all variables to predict

targets = list(behav_df.columns)
print(targets)
print(len(targets))

# 5. Load FC data into memory

fc_data = read_fc_matrices(scan_list,DATA_DIR,ATLAS_NAME)

# 6. Save FC data in vectorized form for all scans into a single file for easy access for batch jobs

out_path = osp.join(RESOURCES_CPM_DIR,f'fc_data_{ATLAS_NAME}.csv')
fc_data.to_csv(out_path)
print('++ INFO: FC data saved to disk [%s]' % out_path)

# 7. Save SNYCQ in the cpm resources folder

out_path = osp.join(RESOURCES_CPM_DIR,'behav_data.csv')
behav_df.to_csv(out_path)
print('++ INFO: Behavioral data saved to disk [%s]' % out_path)

#
# ## 1.2. Create Swarm Jobs for the real data

# We will generate separate swarm files per question. Similarly we will separate the swarm jobs that are for computations on real data and those that are for the generation of the null distribution.

# +
#user specific folders
#=====================
swarm_folder = osp.join(PRJ_DIR,'SwarmFiles.{username}'.format(username=username),'S15')
logs_folder  = osp.join(PRJ_DIR,'Logs.{username}'.format(username=username),'S15')
swarm_path,logdir_path={},{}

if not osp.exists(swarm_folder):
    os.makedirs(swarm_folder)
    print('++ INFO: New folder for swarm files created [%s]' % swarm_folder)

if not osp.exists(logs_folder):
    os.makedirs(logs_folder)
    print('++ INFO: New folder for log files created [%s]' % swarm_folder)

for TARGET in targets:    
    swarm_path[TARGET]  = osp.join(swarm_folder,'S15_CPM-{atlas}-real-{sm}-{conf}-{mt}-{target}.SWARM.sh'.format(atlas=ATLAS_NAME,sm=SPLIT_MODE,conf=CONFOUNDS,mt=MODEL_TYPE, target=TARGET))
    logdir_path[TARGET] = osp.join(logs_folder, 'S15_CPM-{atlas}-real-{sm}-{conf}-{mt}-{target}.logs'.format(atlas=ATLAS_NAME,sm=SPLIT_MODE,conf=CONFOUNDS,mt=MODEL_TYPE, target=TARGET))
# -

# create specific folders if needed
# ======================================
for TARGET in targets:
    if not osp.exists(logdir_path[TARGET]):
        os.makedirs(logdir_path[TARGET])
        print('++ INFO: New folder for log files created [%s]' % logdir_path[TARGET])

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
            os.makedirs(out_dir)
        swarm_file.write("export BEHAV_PATH={behav_path} FC_PATH={fc_path} OUT_DIR={output_dir} BEHAVIOR={behavior} NUM_FOLDS={k} NUM_ITER={n_iter} CORR_TYPE={corr_type} E_SUMMARY_METRIC={e_summary_metric} E_THR_R={e_thr_r} E_THR_P={e_thr_p} SPLIT_MODE={split_mode} VERBOSE=True RANDOMIZE_BEHAVIOR=False CONFOUNDS={confounds}; sh {scripts_folder}/S15_cpm_batch.sh".format(scripts_folder = SCRIPTS_DIR,
                           behav_path       = osp.join(RESOURCES_CPM_DIR,'behav_data.csv'),
                           fc_path          = osp.join(RESOURCES_CPM_DIR,f'fc_data_{ATLAS_NAME}.csv'),
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

# ## 1.3. Create Swarm jobs for the Null Distributions

for TARGET in targets:    
    swarm_path[TARGET]  = osp.join(swarm_folder,'S15_CPM-{atlas}-null-{sm}-{conf}-{mt}-{target}.SWARM.sh'.format(atlas=ATLAS_NAME,sm=SPLIT_MODE,conf=CONFOUNDS,mt=MODEL_TYPE, target=TARGET))
    logdir_path[TARGET] = osp.join(logs_folder, 'S15_CPM-{atlas}-null-{sm}-{conf}-{mt}-{target}.logs'.format(atlas=ATLAS_NAME,sm=SPLIT_MODE,conf=CONFOUNDS,mt=MODEL_TYPE, target=TARGET))

# create specific folders if needed
# ======================================
for TARGET in targets:
    if not osp.exists(logdir_path[TARGET]):
        os.makedirs(logdir_path[TARGET])
        print('++ INFO: New folder for log files created [%s]' % logdir_path[TARGET])

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
        swarm_file.write("export BEHAV_PATH={behav_path} FC_PATH={fc_path} OUT_DIR={output_dir} BEHAVIOR={behavior} NUM_FOLDS={k} NUM_ITER={n_iter} CORR_TYPE={corr_type} E_SUMMARY_METRIC={e_summary_metric} E_THR_R={e_thr_r} E_THR_P={e_thr_p} SPLIT_MODE={split_mode} VERBOSE=True RANDOMIZE_BEHAVIOR=True CONFOUNDS={confounds}; sh {scripts_folder}/S15_cpm_batch.sh".format(scripts_folder = SCRIPTS_DIR,
                           behav_path       = osp.join(RESOURCES_CPM_DIR,'behav_data.csv'),
                           fc_path          = osp.join(RESOURCES_CPM_DIR,f'fc_data_{ATLAS_NAME}.csv'),
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

# ## 1.4 Command Lines to load everything into a single dataframe

# ```bash
# # 
# python /data/SFIMJGC_Introspec/2023_fc_introspection/code/fc_introspection/notebooks/S15b_GatherSwarmResults.py \
#        -i /data/SFIMJGC_Introspec/2023_fc_introspection/code/fc_introspection/resources/cpm/swarm_outputs/real/Schaefer2018_400Parcels_7Networks_AAL2/subject_aware/conf_residualized/pearson_sum/ \
#        -o /data/SFIMJGC_Introspec/2023_fc_introspection/code/fc_introspection/resources/cpm/real-Schaefer2018_400Parcels_7Networks_AAL2-subject_aware-conf_residualized-pearson_sum.pkl \
#        -n 100
#
#
# python ./S15b_GatherSwarmResults.py  -i /data/SFIMJGC_Introspec/2023_fc_introspection/code/fc_introspection/resources/cpm/swarm_outputs/real/Schaefer2018_400Parcels_7Networks_AAL2/subject_aware/conf_not_residualized/pearson_sum/ -o /data/SFIMJGC_Introspec/2023_fc_introspection/code/fc_introspection/resources/cpm/real-Schaefer2018_400Parcels_7Networks_AAL2-subject_aware-conf_not_residualized-pearson_sum.pkl -n 100 -c pearson
#
# python ./S15b_GatherSwarmResults.py  -i /data/SFIMJGC_Introspec/2023_fc_introspection/code/fc_introspection/resources/cpm/swarm_outputs/null/Schaefer2018_400Parcels_7Networks_AAL2/subject_aware/conf_not_residualized/pearson_sum/ -o /data/SFIMJGC_Introspec/2023_fc_introspection/code/fc_introspection/resources/cpm/null-Schaefer2018_400Parcels_7Networks_AAL2-subject_aware-conf_not_residualized-pearson_sum.pkl -n 10000 -c pearson
# ```
