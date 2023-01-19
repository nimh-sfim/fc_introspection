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
#     display_name: Dyneusr (Nov 2021)
#     language: python
#     name: dyneusr_nov2021
# ---

# # Description
#
# This notebook looks at the scans that have completed the functional pre-processing, generates censoring files for them and finally generates a list of scans that we will not use any further due to excessive head motion

import pandas as pd
import numpy as np
import os.path as osp
import hvplot.pandas
import holoviews as hv

from utils.basics import DATA_DIR, RESOURCES_DINFO_DIR, PREPROCESSING_NOTES_DIR
from utils.basics import REL_MOT_THRESHOLD, FINAL_NUM_VOLS, MAX_CENSOR_PERCENT

# # 1. Load Original SNYCQ Dataframe with all 693 entries

SNYCQ_data=pd.read_csv(osp.join(RESOURCES_DINFO_DIR, 'SNYCQ_Preproc.csv'), index_col=['Subject','Run'])
print('++ INFO: SNYCQ Dataframe shape = %s' % str(SNYCQ_data.shape))
SNYCQ_data.sample(4)

# # 2. Remove entries for scans associated with subjects that failed anatomical pre-processing

struct_fail_list = pd.read_csv(osp.join(PREPROCESSING_NOTES_DIR, 'NC_struct_fail_list.csv'))
sbj_to_remove    = struct_fail_list['Subject'].unique()
print('++ INFO: Number of subjects to remove due to incomplete struc preproc = %d' % len(sbj_to_remove))

SNYCQ_data = SNYCQ_data.drop(SNYCQ_data.loc[sbj_to_remove,:].index)
print('++ INFO: Number of remaining scans: %d scans' % SNYCQ_data.shape[0])
print('++ INFO: Number of subjects with at least one scan still available: %d subjects' % len(SNYCQ_data.reset_index()['Subject'].unique()))

# ***
# # 3. Remove entries for scans with issues during functional pre-preprocessing

func_fail_list             = pd.read_csv(osp.join(PREPROCESSING_NOTES_DIR, 'NC_func_fail_list.csv'), index_col=['Subject','Run'])
print('++ INFO: Number of scans to remove due to failed func preproc = %d' % len(func_fail_list))

SNYCQ_data = SNYCQ_data.drop(func_fail_list.index)
print('++ INFO: Number of remaining scans:                                 %d scans' % SNYCQ_data.shape[0])
print('++ INFO: Number of subjects with at least one scan still available: %d subjects' % len(SNYCQ_data.reset_index()['Subject'].unique()))

# ***
# # 4. Discard scans with excessive motion
#
# ## 4.1 Load Relative Motion Traces

motion_df = pd.DataFrame(index=SNYCQ_data.index,columns=np.arange(651))
for sbj,run in motion_df.index:
    _,_,sID,_,rID,_,aID = run.split('-')
    mot_path = osp.join(DATA_DIR,'PrcsData',sbj,'preprocessed','func','pb01_moco','_scan_id_ses-{sID}_task-rest_acq-{aID}_run-{rID}_bold'.format(sID=sID,rID=rID,aID=aID),'rest_realigned_rel.rms')
    aux_mot  = np.loadtxt(mot_path)
    if aux_mot.shape[0] < 651:
        print('++ WARNING: Run has less than expected acquisitions [%s, %s]' % (aux_mot.shape[0],mot_path))
    motion_df.loc[(sbj,run),:] = np.loadtxt(mot_path)

print('++ INFO: Shape of motion_df = %s' % str(motion_df.shape))

# ## 4.2 Search for volumes with excessive motion
#
# For every volume with relative motion above the threshold, we mark as volumes to be censored that volume, the one before and the following two.

num_rem_vols = pd.DataFrame(columns=['Num Removed','PC Removed'], index=motion_df.index)
for sbj,run in motion_df.index:
    _,_,sID,_,rID,_,aID = run.split('-')
    aux          = motion_df.loc[sbj,run]
    # Mark bad volumes
    bad_volumes  = np.unique(np.concatenate([np.array(list(aux.where(aux >= REL_MOT_THRESHOLD).dropna().index))-1, 
                                             np.array(list(aux.where(aux >= REL_MOT_THRESHOLD).dropna().index)), 
                                             np.array(list(aux.where(aux >= REL_MOT_THRESHOLD).dropna().index))+1, 
                                             np.array(list(aux.where(aux >= REL_MOT_THRESHOLD).dropna().index)) + 2]))
    # Correct if first volume was above threshold
    if -1 in bad_volumes:
        bad_volumes = np.delete(bad_volumes,np.where(bad_volumes==-1)) # This happens if the first volume has high motion (then I would attempt to censor the -1 volume)
    # Obtain list of good volumes
    all_volumes  = np.arange(FINAL_NUM_VOLS)
    good_volumes = np.array([i for i in all_volumes if i not in bad_volumes])
    # Save list of good volumes to disk
    output_path  = osp.join(DATA_DIR,'PrcsData',sbj,'preprocessed','func','pb01_moco','_scan_id_ses-{sID}_task-rest_acq-{aID}_run-{rID}_bold'.format(sID=sID,rID=rID,aID=aID),'motion_good_volume_list.csv')
    np.savetxt(output_path, [good_volumes], delimiter=",", fmt='%d')
    # Generate a censor file [0 = bad volume not to be included | 1 = good volume to be included]
    censor = np.zeros(FINAL_NUM_VOLS)
    if good_volumes.shape[0]>0:
        censor[good_volumes] = 1
    else:
        print('++ WARNING: No good volumes for (%s,%s)' % (sbj,run))
    censor = censor.astype(int)
    # Save Censor file to disk
    censor_path = osp.join(DATA_DIR,'PrcsData',sbj,'preprocessed','func','pb01_moco','_scan_id_ses-{sID}_task-rest_acq-{aID}_run-{rID}_bold'.format(sID=sID,rID=rID,aID=aID),'censor.1D')
    np.savetxt(censor_path, censor, delimiter='\n', fmt='%d')
    # Add information about the numbner of volumes / percentage of volumes being censored per scan. 
    num_rem_vols.loc[(sbj,run),'Num Removed'] = bad_volumes.shape[0]
    num_rem_vols.loc[(sbj,run),'PC Removed'] = 100*bad_volumes.shape[0]/FINAL_NUM_VOLS

num_rem_vols.hvplot.hist(bins=100, y='PC Removed', ylabel='Number of Scans', xlabel='Percentage of censored volumes per scan') * hv.VLine(30).opts(line_dash='dashed', line_color='k')

scans_to_remove_due_to_excessive_motion = num_rem_vols[num_rem_vols['PC Removed'] > MAX_CENSOR_PERCENT]
SNYCQ_data = SNYCQ_data.drop(scans_to_remove_due_to_excessive_motion.index)

print('++ INFO: Number of scans to remove due to excessive motion:                         %s scans' %    len(scans_to_remove_due_to_excessive_motion))
print('++ INFO: Number of remaining scans:                                                 %d scans' % SNYCQ_data.shape[0])
print('++ INFO: Number of subjects remaining after removal of scans with excessive motion: %d subjects' % len(SNYCQ_data.reset_index()['Subject'].unique()))

# ## 4.3. Write list of scans with excessive motion to disk

output_path = osp.join(PREPROCESSING_NOTES_DIR,'NC_func_too_much_motion_list.csv')
scans_to_remove_due_to_excessive_motion.to_csv(output_path)
print('++ INFO: List of scans with excessive motion written to disk: [%s]' % output_path)
