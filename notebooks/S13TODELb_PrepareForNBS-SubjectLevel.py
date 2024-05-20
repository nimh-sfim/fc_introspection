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
#     display_name: FC Instrospection (2023 | 3.10)
#     language: python
#     name: fc_introspection_2023_py310
# ---

# # Description - Prepare Data for NBS Analysis on Matlab
#
# This notebook prepares connectivity matrices for doing the NBS analyses on Matlab. At this point we have selected scans corresponding to the two populations that we want to compare and all that's left is to generate the design matrices and create a copy of the FC matrices in a format that NBS can understand.

import os.path as osp
import os
import pandas as pd
import numpy as np
import xarray as xr
from sfim_lib.io.afni import load_netcc
from shutil import rmtree
from tqdm import tqdm
from IPython import display

from utils.basics import RESOURCES_NBS_DIR, ATLASES_DIR, DATA_DIR, SNYCQ_CLUSTERS_INFO_PATH, SNYCQ_W_PATH, RESOURCES_DINFO_DIR
from utils.basics import FB_400ROI_ATLAS_NAME, FB_400ROI_ATLAS_PATH, FB_400ROI_BRAINNET_NODES_PATH
from utils.basics import FB_200ROI_ATLAS_NAME, FB_200ROI_ATLAS_PATH, FB_200ROI_BRAINNET_NODES_PATH
from utils.basics import get_sbj_scan_list

import matplotlib.pyplot as plt
import hvplot.pandas
print(hvplot.__version__)

print("++ INFO: RESOURCES_NBS_DIR=%s" % RESOURCES_NBS_DIR)

ATLAS_NAME = FB_400ROI_ATLAS_NAME

roi_info, Nrois = {},{}
ATLASINFO_PATH       = osp.join(ATLASES_DIR,ATLAS_NAME,'{ATLAS_NAME}.roi_info.csv'.format(ATLAS_NAME=ATLAS_NAME))
roi_info[ATLAS_NAME] = pd.read_csv(ATLASINFO_PATH)
Nrois[ATLAS_NAME]    = roi_info[ATLAS_NAME].shape[0]
print('++ Number of ROIs: %d ROIs' % Nrois[ATLAS_NAME])

# ***
# # 2. Create Design Matrix for NBS
#
# For this version of the NBS analyses, we will select a single scan per subject/cluster if more than one is available. 
#
# We will select whichever scan is closest to the extreme corners. 
#
# To start, we will load the W matrix for all 471 scans. This matrix contains the location of each scan in the 2D space estimated by sbcNNMF

print(pd.__version__)
W = pd.read_csv(SNYCQ_W_PATH,index_col=['Subject','Run'])

# Load information about cluter membership for all 471 scans

clusters_info = pd.read_csv(SNYCQ_CLUSTERS_INFO_PATH, index_col=['Subject','Run'])

# Next, we merge both dataframes into a single dataframe that has both cluster membership and W coordinates

mot_info = pd.read_csv(osp.join(RESOURCES_DINFO_DIR,'motion_confounds.csv'),index_col=['Subject','Run'])

W_plus        = pd.concat([W,clusters_info,mot_info],axis=1)

# We now remove all scans in the Intermediate group. So far this is equivalent to the original NBS scan selection (Scan-Level)

W_plus = W_plus[W_plus['Cluster Label'] != 'Intermediate']

# Get a list of subjects that were in the two clusters of interest

sbj_list       = W_plus.index.get_level_values('Subject').unique()
print(len(sbj_list))

# We define the extreme corners for each cluster. We need those to compute the euclidean distance

# ## Full Sample

final_scan_list = {}
final_scan_list['All_Scans'] = W_plus.copy()
final_scan_list['All_Scans']['Cluster Label'].value_counts()

# ## Random Smaller Sample

final_scan_list['sample_150'] = pd.concat([W_plus.reset_index().set_index('Cluster Label').loc['Image-Pos-Others'].sample(75).reset_index().set_index(['Subject','Run']).sort_index(level=['Subject','Run']),
                                           W_plus.reset_index().set_index('Cluster Label').loc['Surr-Neg-Self'].sample(75).reset_index().set_index(['Subject','Run']).sort_index(level=['Subject','Run'])])
final_scan_list['sample_150']['Cluster Label'].value_counts()

final_scan_list['sample_140'] = pd.concat([W_plus.reset_index().set_index('Cluster Label').loc['Image-Pos-Others'].sample(70).reset_index().set_index(['Subject','Run']).sort_index(level=['Subject','Run']),
                                           W_plus.reset_index().set_index('Cluster Label').loc['Surr-Neg-Self'].sample(70).reset_index().set_index(['Subject','Run']).sort_index(level=['Subject','Run'])])
final_scan_list['sample_140']['Cluster Label'].value_counts()

final_scan_list['sample_120'] = pd.concat([W_plus.reset_index().set_index('Cluster Label').loc['Image-Pos-Others'].sample(60).reset_index().set_index(['Subject','Run']).sort_index(level=['Subject','Run']),
                                           W_plus.reset_index().set_index('Cluster Label').loc['Surr-Neg-Self'].sample(60).reset_index().set_index(['Subject','Run']).sort_index(level=['Subject','Run'])])
final_scan_list['sample_120']['Cluster Label'].value_counts()

final_scan_list['sample_100'] = pd.concat([W_plus.reset_index().set_index('Cluster Label').loc['Image-Pos-Others'].sample(50).reset_index().set_index(['Subject','Run']).sort_index(level=['Subject','Run']),
                                           W_plus.reset_index().set_index('Cluster Label').loc['Surr-Neg-Self'].sample(50).reset_index().set_index(['Subject','Run']).sort_index(level=['Subject','Run'])])
final_scan_list['sample_100']['Cluster Label'].value_counts()

final_scan_list['sample_80'] = pd.concat([W_plus.reset_index().set_index('Cluster Label').loc['Image-Pos-Others'].sample(40).reset_index().set_index(['Subject','Run']).sort_index(level=['Subject','Run']),
                                           W_plus.reset_index().set_index('Cluster Label').loc['Surr-Neg-Self'].sample(40).reset_index().set_index(['Subject','Run']).sort_index(level=['Subject','Run'])])
final_scan_list['sample_80']['Cluster Label'].value_counts()

# ## Extreme Selection of Scans

CORNERS = {'Image-Pos-Others':pd.Series([1,0],['Factor 1','Factor 2']).astype(float),
           'Surr-Neg-Self':pd.Series([0,1],['Factor 1','Factor 2']).astype(float)}

final_scan_list['extremes'] = W_plus.copy()
for sbj in sbj_list:
    this_sbj_scans = W_plus.loc[sbj].copy()
    clusters_sbj_has_scans = list(this_sbj_scans['Cluster Label'].unique())
    #####
    if len(clusters_sbj_has_scans) > 1:
        final_scan_list['extremes'].drop(sbj,axis=0, level='Subject', inplace=True)
        continue
    #####
    for cluster in clusters_sbj_has_scans:
        this_sbj_scans_in_cluster = this_sbj_scans.reset_index().set_index('Cluster Label').loc[cluster].copy()
        if isinstance(this_sbj_scans_in_cluster,pd.Series):
            # There is only one scan --> nothing needs to be removed
            continue
        # I can now safely assume, I am dealing with a this_sbj_scans_in_cluster of type dataframe
        number_of_scans           = this_sbj_scans_in_cluster.shape[0]
        this_sbj_scans_in_cluster = this_sbj_scans_in_cluster.reset_index().set_index('Run')
        
        corner    = CORNERS[cluster]
        best_scan = np.sqrt(((this_sbj_scans_in_cluster[['Factor 1','Factor 2']] - corner)*(this_sbj_scans_in_cluster[['Factor 1','Factor 2']] - corner)).sum(axis=1).sort_values(ascending=True)).index[0]
        all_scans = list(this_sbj_scans_in_cluster.index)
        for scan in all_scans:
            if scan != best_scan:
                # Anything that it is not a top scan, needs to be removed
                final_scan_list['extremes'].drop((sbj,scan),axis=0, inplace=True)

final_scan_list['extremes'] = final_scan_list['extremes'].sort_values('Cluster Label')

# ## Extreme Selection of Scans (remove further away)

CORNERS = {'Image-Pos-Others':pd.Series([1,0],['Factor 1','Factor 2']).astype(float),
           'Surr-Neg-Self':pd.Series([0,1],['Factor 1','Factor 2']).astype(float)}

final_scan_list['extremes2'] = W_plus.copy()
for sbj in sbj_list:
    this_sbj_scans = W_plus.loc[sbj].copy()
    clusters_sbj_has_scans = list(this_sbj_scans['Cluster Label'].unique())
    #####
    if len(clusters_sbj_has_scans) > 1:
        final_scan_list['extremes2'].drop(sbj,axis=0, level='Subject', inplace=True)
        continue
    #####
    for cluster in clusters_sbj_has_scans:
        this_sbj_scans_in_cluster = this_sbj_scans.reset_index().set_index('Cluster Label').loc[cluster].copy()
        if isinstance(this_sbj_scans_in_cluster,pd.Series):
            # There is only one scan --> nothing needs to be removed
            continue
        # I can now safely assume, I am dealing with a this_sbj_scans_in_cluster of type dataframe
        number_of_scans           = this_sbj_scans_in_cluster.shape[0]
        this_sbj_scans_in_cluster = this_sbj_scans_in_cluster.reset_index().set_index('Run')
        
        corner     = CORNERS[cluster]
        worse_scan = np.sqrt(((this_sbj_scans_in_cluster[['Factor 1','Factor 2']] - corner)*(this_sbj_scans_in_cluster[['Factor 1','Factor 2']] - corner)).sum(axis=1).sort_values(ascending=False)).index[0]
        final_scan_list['extremes2'].drop(index=(sbj,worse_scan), inplace=True)

final_scan_list['extremes2'] = final_scan_list['extremes2'].sort_values('Cluster Label')

# ## Closest to Centroids

CORNERS = {cluster:W_plus[W_plus['Cluster Label']==cluster][['Factor 1','Factor 2']].mean() for cluster in ['Image-Pos-Others','Surr-Neg-Self']}
CORNERS

final_scan_list['centroids'] = W_plus.copy()
for sbj in sbj_list:
    this_sbj_scans = W_plus.loc[sbj].copy()
    clusters_sbj_has_scans = list(this_sbj_scans['Cluster Label'].unique())
    #####
    if len(clusters_sbj_has_scans) > 1:
        final_scan_list['centroids'].drop(sbj,axis=0, level='Subject', inplace=True)
        continue
    #####
    for cluster in clusters_sbj_has_scans:
        this_sbj_scans_in_cluster = this_sbj_scans.reset_index().set_index('Cluster Label').loc[cluster].copy()
        if isinstance(this_sbj_scans_in_cluster,pd.Series):
            # There is only one scan --> nothing needs to be removed
            continue
        # I can now safely assume, I am dealing with a this_sbj_scans_in_cluster of type dataframe
        number_of_scans           = this_sbj_scans_in_cluster.shape[0]
        this_sbj_scans_in_cluster = this_sbj_scans_in_cluster.reset_index().set_index('Run')
        
        corner    = CORNERS[cluster]
        best_scan = np.sqrt(((this_sbj_scans_in_cluster[['Factor 1','Factor 2']] - corner)*(this_sbj_scans_in_cluster[['Factor 1','Factor 2']] - corner)).sum(axis=1).sort_values(ascending=True)).index[0]
        all_scans = list(this_sbj_scans_in_cluster.index)
        for scan in all_scans:
            if scan != best_scan:
                # Anything that it is not a top scan, needs to be removed
                final_scan_list['centroids'].drop((sbj,scan),axis=0, inplace=True)

final_scan_list['centroids'] = final_scan_list['centroids'].sort_values('Cluster Label')

# ## Remove scans from subjects with scans on both clusters

final_scan_list['both_clusters_out'] = W_plus.copy()
for sbj in sbj_list:
    this_sbj_scans = W_plus.loc[sbj].copy()
    clusters_sbj_has_scans = list(this_sbj_scans['Cluster Label'].unique())
    #####
    # If subject has scans in both cluster, then all the scans for the subject get removed
    if len(clusters_sbj_has_scans) > 1:
        final_scan_list['both_clusters_out'].drop(sbj,axis=0, level='Subject', inplace=True)
        num_scans = this_sbj_scans.shape[0]
        num_scans_in_F1 = this_sbj_scans[this_sbj_scans['Cluster Label']=='Image-Pos-Others'].shape[0]
        num_scans_in_F2 = this_sbj_scans[this_sbj_scans['Cluster Label']=='Surr-Neg-Self'].shape[0]
        print(f'++ WARNING [{sbj} | {num_scans}] This subject has scans in both groups [{num_scans_in_F1} + {num_scans_in_F2}]')
        continue
    #####

# ## Centroid-based Selection of Scans (remove further away)

CORNERS = {cluster:W_plus[W_plus['Cluster Label']==cluster][['Factor 1','Factor 2']].mean() for cluster in ['Image-Pos-Others','Surr-Neg-Self']}
CORNERS

cluster

final_scan_list['centroids2'] = W_plus.copy()
for sbj in sbj_list:
    this_sbj_scans = W_plus.loc[sbj].copy()
    clusters_sbj_has_scans = list(this_sbj_scans['Cluster Label'].unique())
    # If subject has scans in both cluster, then all the scans for the subject get removed
    if len(clusters_sbj_has_scans) > 1:
        final_scan_list['centroids2'].drop(sbj,axis=0, level='Subject', inplace=True)
        num_scans = this_sbj_scans.shape[0]
        num_scans_in_F1 = this_sbj_scans[this_sbj_scans['Cluster Label']=='Image-Pos-Others'].shape[0]
        num_scans_in_F2 = this_sbj_scans[this_sbj_scans['Cluster Label']=='Surr-Neg-Self'].shape[0]
        print(f'++ WARNING [{sbj} | {num_scans}] This subject has scans in both groups [{num_scans_in_F1} + {num_scans_in_F2}]')
        continue
    # Past this point I know all scans from this subject are on the same and only cluster
    cluster = clusters_sbj_has_scans[0]
    this_sbj_scans_in_cluster = this_sbj_scans.reset_index().set_index('Cluster Label').loc[cluster].copy()
    if isinstance(this_sbj_scans_in_cluster,pd.Series):
        # There is only one scan --> nothing needs to be removed
        continue
    # I can now safely assume, I am dealing with a this_sbj_scans_in_cluster of type dataframe
    this_sbj_scans_in_cluster = this_sbj_scans_in_cluster.reset_index().set_index('Run')
        
    corner     = CORNERS[cluster]
    worse_scan = np.sqrt(((this_sbj_scans_in_cluster[['Factor 1','Factor 2']] - corner)*(this_sbj_scans_in_cluster[['Factor 1','Factor 2']] - corner)).sum(axis=1).sort_values(ascending=False)).index[0]
    print(f'++ INFO: Removed scan [{sbj},{worse_scan}]')
    final_scan_list['centroids2'].drop(index=(sbj,worse_scan), inplace=True)

final_scan_list['centroids2'] = final_scan_list['centroids2'].sort_values('Cluster Label')

# ## Minimum Motion

this_sbj_scans.reset_index().set_index('Cluster Label').loc[cluster]

final_scan_list['Subject-Level_Min-Motion'] = W_plus.copy()
for sbj in sbj_list:
    this_sbj_scans = W_plus.loc[sbj].copy()
    clusters_sbj_has_scans = list(this_sbj_scans['Cluster Label'].unique())
    #####
    # If subject has scans in both cluster, then all the scans for the subject get removed
    if len(clusters_sbj_has_scans) > 1:
        final_scan_list['Subject-Level_Min-Motion'].drop(sbj,axis=0, level='Subject', inplace=True)
        num_scans = this_sbj_scans.shape[0]
        num_scans_in_F1 = this_sbj_scans[this_sbj_scans['Cluster Label']=='Image-Pos-Others'].shape[0]
        num_scans_in_F2 = this_sbj_scans[this_sbj_scans['Cluster Label']=='Surr-Neg-Self'].shape[0]
        print(f'++ WARNING [{sbj} | {num_scans}] This subject has scans in both groups [{num_scans_in_F1} + {num_scans_in_F2}]')
        continue
    #####
    if this_sbj_scans.shape[0] == 1:
        # If this subject only has one scan, I do not need to discard anything
        #print(f'[{sbj},{cluster}] --> Only one scan for this subject')
        continue
    clusters_sbj_has_scans = list(this_sbj_scans['Cluster Label'].unique())
    for cluster in clusters_sbj_has_scans:
        if isinstance(this_sbj_scans.reset_index().set_index('Cluster Label').loc[cluster],pd.Series):
            # If there is only one scan for this cluster and subject --> do nothing
            #print(f'[{sbj},{cluster}] --> Only one scan for this subject & cluster')
            continue
        this_sbj_scans_in_cluster = this_sbj_scans.reset_index().set_index('Cluster Label').loc[cluster].copy().reset_index().set_index('Run')
        best_scan = this_sbj_scans_in_cluster.sort_values(by='Mean Rel Motion', ascending=True).index[0]
        all_scans = list(this_sbj_scans_in_cluster.index)
        #print(f'[{sbj},{cluster}]' + str(all_scans) + ' --> ' + best_scan)
        for scan in all_scans:
            if scan != best_scan:
                # Anything that it is not a top scan, needs to be removed
                final_scan_list['Subject-Level_Min-Motion'].drop(index=(sbj,scan), inplace=True)

final_scan_list['Subject-Level_Min-Motion'] = final_scan_list['Subject-Level_Min-Motion'].sort_values('Cluster Label')

# ## Subjects than only have 1 scan

final_scan_list['one_scan_only'] = W_plus.copy()
for sbj in sbj_list:
    if W_plus.loc[sbj].shape[0] < 2:
        final_scan_list['one_scan_only'].drop(index=sbj,level='Subject', inplace=True)

final_scan_list['one_scan_only'] = final_scan_list['one_scan_only'].sort_values('Cluster Label')

# This next cell will loop over all subjects:
# * Check in how many clusters the subject has scans
# * For each cluster, select only the scan closest to the corner of interest

# # Visualize Selection

FINAL_SELECTION = 'All_Scans'
SUBJECT_LEVEL_DIR = osp.join(RESOURCES_NBS_DIR,ATLAS_NAME,FINAL_SELECTION)
print('++ INFO: Files specific to subject-level NBS analyses --> %s' % SUBJECT_LEVEL_DIR)

if osp.exists(SUBJECT_LEVEL_DIR):
    rmtree(SUBJECT_LEVEL_DIR)
    print('++ WARNING: Removing pre-existing folder [%s]' % SUBJECT_LEVEL_DIR)
os.makedirs(SUBJECT_LEVEL_DIR)
print('++ INFO: Regenerating empty folder [%s]' % SUBJECT_LEVEL_DIR)

print('++ Number of scans across both clustes: %d scans' % final_scan_list[FINAL_SELECTION].index.get_level_values('Subject').shape[0])
print('++ Number of subjects across both clusters: %d subjects' % final_scan_list[FINAL_SELECTION].index.get_level_values('Subject').unique().shape[0])

final_scan_list[FINAL_SELECTION]['Cluster Label'].value_counts()

# This is a plot to ensure that the selection process worked as expected

pd.DataFrame(CORNERS['Image-Pos-Others']).T.hvplot.scatter(x='Factor 1', y='Factor 2', marker='square',size=100, color='red', alpha=.5, aspect='square') * \
pd.DataFrame(CORNERS['Surr-Neg-Self']).T.hvplot.scatter(x='Factor 1', y='Factor 2', marker='square',size=100, color='blue', alpha=.5, aspect='square') * \
W_plus[W_plus['Cluster Label']=='Image-Pos-Others'].hvplot.scatter(x='Factor 1',y='Factor 2', hover_cols=['Subject','Run'], aspect='square', s=50, c='orange', frame_width=600) * \
W_plus[W_plus['Cluster Label']=='Surr-Neg-Self'].hvplot.scatter(x='Factor 1',y='Factor 2', hover_cols=['Subject','Run'], aspect='square', s=50, c='lightblue', frame_width=600) * \
final_scan_list[FINAL_SELECTION][final_scan_list[FINAL_SELECTION]['Cluster Label']=='Image-Pos-Others'].hvplot.scatter(x='Factor 1',y='Factor 2', aspect='square', size=5, c='k') * \
final_scan_list[FINAL_SELECTION][final_scan_list[FINAL_SELECTION]['Cluster Label']=='Surr-Neg-Self'].hvplot.scatter(x='Factor 1',y='Factor 2', aspect='square', size=5, c='k')  * \
W_plus.loc['sub-010058'].hvplot.scatter(x='Factor 1',y='Factor 2', color='r',s=600, alpha=.5, line_width=5, fill_color=None, aspect='square')
#W_plus.loc['sub-010098'].hvplot.scatter(x='Factor 1',y='Factor 2', color='y',s=600, alpha=.5, line_width=5, fill_color=None, aspect='square') * \
#W_plus.loc['sub-010060'].hvplot.scatter(x='Factor 1',y='Factor 2', color='m',s=600, alpha=.5, line_width=5, fill_color=None, aspect='square')

final_scan_list[FINAL_SELECTION].sort_index(level=['Subject','Run']).reset_index().groupby('Subject').count().sort_values(by='Run').hvplot.bar(x='Subject',y='Run', width=1500).opts(xrotation=90)

# +
fig, ax = plt.subplots(1,1,figsize=(7,7))
W_plus.plot.scatter(x='Factor 1',y='Factor 2', ax=ax, c='red',s=10)
W_plus.loc['sub-010098'].plot.scatter(x='Factor 1',y='Factor 2', color='w', edgecolor='k',s=60, ax=ax, alpha=.5) # Subject with two scans (one on each cluster)

#W_plus.loc['sub-010015'].plot.scatter(x='Factor 1',y='Factor 2', color='w', edgecolor='k',s=60, ax=ax, alpha=.5) #Subject with 4 scans
#W_plus.loc['sub-010094'].plot.scatter(x='Factor 1',y='Factor 2', color='w', edgecolor='y',s=60, ax=ax, alpha=.5)
#W_plus.loc['sub-010017'].plot.scatter(x='Factor 1',y='Factor 2', color='w', edgecolor='m',s=60, ax=ax, alpha=.5)
#W_plus.loc['sub-010060'].plot.scatter(x='Factor 1',y='Factor 2', color='w', edgecolor='c',s=60, ax=ax, alpha=.5)
#W_plus.loc['sub-010034'].plot.scatter(x='Factor 1',y='Factor 2', color='w', edgecolor='g',s=60, ax=ax, alpha=.5)
final_scan_list[FINAL_SELECTION].plot.scatter(x='Factor 1',y='Factor 2', ax=ax, c='k', s=1)
# -

# # Create Design Matrix
# Transform the same information into a dictionary

print(FINAL_SELECTION)
final_scan_list[FINAL_SELECTION]

scans_per_cluster={cl_label:final_scan_list[FINAL_SELECTION][final_scan_list[FINAL_SELECTION]['Cluster Label']==cl_label].index for cl_label in ['Image-Pos-Others','Surr-Neg-Self']}

# Forth, we print again the number of scans per cluster, as a sanity check

[(k,i.shape) for k,i in scans_per_cluster.items()]

# Fifth, we generate the design matrix taking into accoun only the scans that we use in this part of the analysis (namely those in clusters ```Large F1``` and ```Large F2```)

DESIGN_MATRIX = np.vstack([np.tile(np.array([1,0]),(len(scans_per_cluster['Image-Pos-Others']),1)),
                           np.tile(np.array([0,1]),(len(scans_per_cluster['Surr-Neg-Self']),1))])
DESIGN_MATRIX_PATH = osp.join(SUBJECT_LEVEL_DIR,'NBS_CL02_DesingMatrix.txt')
np.savetxt(DESIGN_MATRIX_PATH,DESIGN_MATRIX,delimiter=' ',fmt='%d')
print('++ INFO: Design Matrix for 2 Cluster solution saved in [%s]' % DESIGN_MATRIX_PATH)
print('++ INFO: Design Matrix for 2 Cluster solution has shape [%s]' % str(DESIGN_MATRIX.shape))

SUBJECT_LEVEL_DIR

len(final_scan_list[FINAL_SELECTION].index.get_level_values('Subject').unique())

# ***
# # 3. Create Copies of Scan-wise FC Matrices in NBS folders
#
# Count how many scans we have in total across the two clusters of interest

#_,scans_list = get_sbj_scan_list(when='post_motion', return_snycq=False)
Nscans       = final_scan_list[FINAL_SELECTION].set_index('Cluster Label').loc[['Image-Pos-Others','Surr-Neg-Self']].shape[0]
print(Nscans)

# ## 3.1. Load all FC matrices into a XR.DataArray 

# %%time
sfc_Z_xr={}
for ATLAS_NAME in [FB_200ROI_ATLAS_NAME]:
    # Create Empty Numpy Array where to hold all FC matrices. At the end we will move this into an Xarray
    # ===================================================================================================
    sfc_Z_arr = np.empty((Nscans,Nrois[ATLAS_NAME],Nrois[ATLAS_NAME])) * np.nan
    scan_idx      = 0
    scan_name_idx = []
    # For all clusters of interest
    # ============================
    for cluster_id in ['Image-Pos-Others','Surr-Neg-Self']:
        # For each scan in a given cluster
        # ================================
        for sbj,run in scans_per_cluster[cluster_id]:
            # Load FC matrix from disk
            # ========================
            scan_name_idx.append('.'.join([sbj,run]))
            _,_,sesID,_,runID,_,acqID = run.split('-')
            sfc_path = osp.join(DATA_DIR,'PrcsData',sbj,'preprocessed','func','pb06_staticFC','{acqID}_run-{runID}.{ATLAS_NAME}_000.netcc'.format(acqID=acqID,runID=runID, ATLAS_NAME=ATLAS_NAME))
            aux_cc_r = load_netcc(sfc_path)
            # Apply Fisher's transformation
            # =============================
            aux_cc_Z = aux_cc_r.apply(np.arctanh)
            np.fill_diagonal(aux_cc_Z.values,1)
            sfc_Z_arr[scan_idx,:,:] = aux_cc_Z
            # Update counter
            # ==============
            scan_idx = scan_idx + 1
            del aux_cc_r, aux_cc_Z
    # Save all FC matrixes for a given atlas in XR.Array Form
    # =======================================================
    sfc_Z_xr[ATLAS_NAME] = xr.DataArray(sfc_Z_arr,
                        dims=['scan','roi_x','roi_y'],
                        coords={'scan':scan_name_idx,
                                'roi_x':roi_info[ATLAS_NAME]['ROI_ID'],
                                'roi_y':roi_info[ATLAS_NAME]['ROI_ID']})
    del sfc_Z_arr

# ## 3.2. Create Data Folder for this solution
#
# Once more, to make sure all things are consistent, we will create empty folders before we start copying the scan-wise FC matrices

for ATLAS_NAME in [FB_200ROI_ATLAS_NAME]:
    NBS_CL02_matrices_folder = osp.join(SUBJECT_LEVEL_DIR,'NBS_CL02_Data')
    if osp.exists(NBS_CL02_matrices_folder):
        print("++ WARNING: Removing pre-existing folder [%s]" % NBS_CL02_matrices_folder)
        rmtree(NBS_CL02_matrices_folder)
    os.mkdir(NBS_CL02_matrices_folder)
    print("++ INFO: Creating empty folder [%s]" % NBS_CL02_matrices_folder)

# ## 3.3. Make copies of matrices into the folder

# %%time
for ATLAS_NAME in tqdm([FB_200ROI_ATLAS_NAME], desc='Atlas'):
    NBS_CL02_matrices_folder = osp.join(SUBJECT_LEVEL_DIR,'NBS_CL02_Data')
    for i,item in enumerate(list(sfc_Z_xr[ATLAS_NAME].indexes['scan'])):
        dest_path = osp.join(NBS_CL02_matrices_folder,'subject{id}.txt'.format(id=str(i+1).zfill(3)))
        np.savetxt(dest_path,sfc_Z_xr[ATLAS_NAME].loc[item,:,:],delimiter=' ',fmt='%f')

# # 4. Run NBS Analysis in Matlab
#
# 1. Open Matlab
#
# 2. Load the Path to NBS
#
# ```addpath(genpath('/data/SFIMJGC_HCP7T/hcp7t_fv_sleep_extraSW/NBS1.2/'))```
#
# 3. Configure NBS appropriately for each of the contrasts
#
#     a. Image-Pos-Others
#     
#     * ```Design Matrix = /data/SFIMJGC_Introspec/2023_fc_introspection/code/fc_introspection/resources/nbs/NBS_CL02_DesingMatrix.txt```
#     * ```Contrast = [1,-1]```
#     * ```Statistical Test = T-test```
#     * ```Threshold = 3.1``` equivalent to p<0.001
#     * ```Connectivity Matrices = /data/SFIMJGC_Introspec/2023_fc_introspection/code/fc_introspection/resources/nbs/Schaefer2018_400Parcels_7Networks_AAL2/NBS_CL02_Data/subject0001.txt```
#     * ```Node Coordinates = /data/SFIMJGC_Introspec/2023_fc_introspection/code/fc_introspection/resources/nbs/Schaefer2018_400Parcels_7Networks_AAL2/Schaefer2018_400Parcels_7Networks_AAL2_NBS_Node_Coordinates.txt```
#     * ```Node Labels = /data/SFIMJGC_Introspec/2023_fc_introspection/code/fc_introspection/resources/nbs/Schaefer2018_400Parcels_7Networks_AAL2/Schaefer2018_400Parcels_7Networks_AAL2_NBS_Node_Labels.txt```
#     * ```Permutations = 5000```
#     * ```Significance = 0.05```
#     * ```Method = Network-Based Statistics (NBS)```
#     * ```Component Size = Extent```
#     
#     Once the program finish, please save the results as: ```/data/SFIMJGC_Introspec/2023_fc_introspection/code/fc_introspection/resources/nbs/Schaefer2018_400Parcels_7Networks_AAL2/NBS_CL02_Results/NBS_CL02_Image-Pos-Others_gt_Surr-Neg-Self.mat```
#
#     b. Surr-Neg-Self
#     
#     Same as above, except for contast, please use ```Contrast = [-1,1]```
#     
#     And save results as ```/data/SFIMJGC_Introspec/2023_fc_introspection/code/fc_introspection/resources/nbs/Schaefer2018_400Parcels_7Networks_AAL2/NBS_CL02_Results/NBS_CL02_Surr-Neg-Self_gt_Image-Pos-Others.mat```
#

display.Image('./figures/S13_NBS_Configuration.png', width=1200)


