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

# # Description
#
# This notebook prepares the data for the NBS analysis between the two extreme sets of scans. As of now this code can be used to prepare two different sets of analysis:
#
# * Basic: Does not take into account the fact that there are repeated scans per subject. This is what I presented in Padova. It is not necessarily correct.
# * SbjAware: Does take into account that there are repeated scans per subjects. Less connections are singificant, but this is more correct. This is what we report in the initial manuscript submission.

# +
import pandas as pd
import numpy as np
import xarray as xr
import os.path as osp
import os
from tqdm import tqdm
from shutil import rmtree
import seaborn as sns
import matplotlib.pyplot as plt
from utils.plotting import hvplot_fc
from scipy.spatial.distance import squareform
from utils.basics import SNYCQ_W_PATH, SNYCQ_CLUSTERS_INFO_PATH, RESOURCES_DINFO_DIR,RESOURCES_NBS_DIR, ATLASES_DIR, DATA_DIR, PRJ_DIR
from utils.basics import FB_400ROI_ATLAS_NAME as ATLAS_NAME
from utils.basics import FB_400ROI_BRAINNET_NODES_PATH as BRAINNET_NODES_PATH
from sfim_lib.io.afni import load_netcc
import panel as pn
from matplotlib import colors
FINAL_SCAN_LISTS = {}

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
# -

from sklearn.preprocessing import OneHotEncoder
import hvplot.pandas

import os
port_tunnel = int(os.environ['PORT2'])
print('++ INFO: Second Port available: %d' % port_tunnel)

# # 1. Load Information about scans

# Load x,y on low dimensional space for each scan

W = pd.read_csv(SNYCQ_W_PATH,index_col=['Subject','Run'])

# Load cluster membershipt for each scan

clusters_info = pd.read_csv(SNYCQ_CLUSTERS_INFO_PATH, index_col=['Subject','Run'])

# Load motion information for each scan

mot_info = pd.read_csv(osp.join(RESOURCES_DINFO_DIR,'motion_confounds.csv'),index_col=['Subject','Run'])

# Combine all info we have about the scans into a single dataframe

SCAN_INFO_df = pd.concat([W,clusters_info,mot_info],axis=1)
SCAN_INFO_df.head(2)

plt.subplots(1,1,figsize=(3,3))
sns.scatterplot(data=SCAN_INFO_df,x='Factor 1',y='Factor 2', hue='Cluster Label')

# # 2. Remove scans in intermediate cluster
#
# For NBS we are only interested in contrasting the two sets on the corners of the low dimensional space. For that reason, we will next remove scans in the intermediate set from the ```SELECTED_SCANS``` data structure

SELECTED_SCANS = SCAN_INFO_df[SCAN_INFO_df['Cluster Label'] != 'Intermediate']
plt.subplots(1,1,figsize=(3,3))
sns.scatterplot(data=SELECTED_SCANS,x='Factor 1',y='Factor 2', hue='Cluster Label')

# We remove from memory intermediate variables that we no longer need, and check the final contents of the ```SELECTED_SCANS``` dataframe

del mot_info, clusters_info, W

SELECTED_SCANS.head(3)

# In addition, we print some basic information for the selected list of scans

FINAL_N_scans = SELECTED_SCANS.index.get_level_values('Subject').shape[0]
FINAL_N_sbjs  = SELECTED_SCANS.index.get_level_values('Subject').unique().shape[0]
print('++ Number of scans across both clustes: %d scans' % FINAL_N_scans)
print('++ Number of subjects across both clusters: %d subjects' % FINAL_N_sbjs)
print('++ Number of scans per cluster')
SELECTED_SCANS['Cluster Label'].value_counts()

# ***
# # 3. Load the FC matrices for selected scans in an Xarray
#
# This xarray will be indexed by ```<SBJ>.<SCAN>``` on the scan dimension, and ROI names in the other two dimensions

ATLASINFO_PATH = osp.join(ATLASES_DIR,ATLAS_NAME,'{ATLAS_NAME}.roi_info.csv'.format(ATLAS_NAME=ATLAS_NAME))
roi_info       = pd.read_csv(ATLASINFO_PATH)
Nrois          = roi_info.shape[0]
print("++ INFO: Selected Atlas = %s" % ATLAS_NAME)
print("++ INFO: Number of ROIs = %d" % Nrois)

# +
# %%time
# Create Empty Numpy Array where to hold all FC matrices. At the end we will move this into an Xarray
# ===================================================================================================
sfc_Z_arr      = np.empty((FINAL_N_scans,Nrois,Nrois)) * np.nan
print('++ INFO: Shape of final Xarray: %s' % str(sfc_Z_arr.shape))
i              = 0  # Index to move through the scan dimension numerically
xr_coords_scan = [] # List of scan IDs to later use as the coordinates for the scan dimension

# For each scan in a given cluster
# ================================
for (sbj,run),_ in tqdm(SELECTED_SCANS.iterrows()):
    # Load FC matrix from disk
    # ========================
    xr_coords_scan.append('.'.join([sbj,run])) 
    _,_,sesID,_,runID,_,acqID = run.split('-')
    sfc_path = osp.join(DATA_DIR,'PrcsData',sbj,'preprocessed','func','pb06_staticFC','{acqID}_run-{runID}.{ATLAS_NAME}_000.netcc'.format(acqID=acqID,runID=runID, ATLAS_NAME=ATLAS_NAME))
    aux_cc_r = load_netcc(sfc_path)
    # Apply Fisher's transformation
    # =============================
    aux_cc_Z = aux_cc_r.apply(np.arctanh)
    np.fill_diagonal(aux_cc_Z.values,1)
    sfc_Z_arr[i,:,:] = aux_cc_Z
    # Update counter
    # ==============
    i = i + 1
    del aux_cc_r, aux_cc_Z

# Save all FC matrixes for a given atlas in XR.Array Form
# =======================================================
sfc_Z_xr = xr.DataArray(sfc_Z_arr,
                    dims=['scan','roi_x','roi_y'],
                    coords={'scan':xr_coords_scan ,
                            'roi_x':roi_info['ROI_ID'],
                            'roi_y':roi_info['ROI_ID']})
del sfc_Z_arr
# -

# ***
# # 4. Create Materials to run NBS
#
# a. Create folder for this particular Atlas / Final Selection in NBS resource folder. If folder already exists, a warning will be given.

MY_NBS_FOLDER = osp.join(RESOURCES_NBS_DIR, ATLAS_NAME)
print(MY_NBS_FOLDER)

if osp.exists(MY_NBS_FOLDER):
    print('++ WARNING: %s already exists... \n   you may want to delete prior results to avoid confussion.' % MY_NBS_FOLDER)
    rmtree(MY_NBS_FOLDER)
os.makedirs(MY_NBS_FOLDER)
print('++ INFO: New folder created %s' % MY_NBS_FOLDER)

# b. Create subfolder where the FC matrices will be saved in a way that NBS can understand.

FC_MATRIX_DATA_FOLDER = osp.join(MY_NBS_FOLDER,'NBS_CL02_Data')
if osp.exists(FC_MATRIX_DATA_FOLDER):
    print('++ WARNING: Removing prior data folder [%s]' % FC_MATRIX_DATA_FOLDER)
    rmtree(FC_MATRIX_DATA_FOLDER)
print('++ INFO: Creating new data folder [%s]' % FC_MATRIX_DATA_FOLDER)
os.makedirs(FC_MATRIX_DATA_FOLDER)

# c. Define path to both NBS design matrix: 
# * Basic: only two columns
# * SbjAware: with the extra columns per subject

DESING_MATRIX_BASIC_PATH       = osp.join(MY_NBS_FOLDER,'NBS_CL02_DesingMatrix.txt')
DESING_MATRIX_SBJAWARE_PATH    = osp.join(MY_NBS_FOLDER,'NBS_CL02_DesingMatrix_SubjectAware.txt')

# # 5. Create node, coordinate and label files for NBS and BrainNet packages
#
# These two software requires a few additional files with information about ROI names, centroids and labels. We generate those next for the two atlases of interest:
#
# * ```<ATLAS_NAME>_BrainNet_Nodes.node```: Information about ROI names and centroids for BrainNet.
# * ```<ATLAS_NAME>_NBS_Node_Coordinates.txt```: ROI centroids in NBS format.
# * ```<ATLAS_NAME>_NBS_Node_Labels.txt```: ROI names in NBS format.

Nw2Id = {'Vis':1,'SomMot':2,'DorsAttn':3,'SalVentAttn':4,'Limbic':5,'Cont':6,'Default':7,'Subcortical':8}

# Create Brainnet Nodes data structure
# ====================================
BRAINNET_NODES_df               = roi_info[['pos_R','pos_A','pos_S','ROI_Name']].copy()
BRAINNET_NODES_df['Node Size']  = 1
BRAINNET_NODES_df['Node Color'] = [Nw2Id[n.split('_')[1]] for n in BRAINNET_NODES_df['ROI_Name']]
BRAINNET_NODES_df = BRAINNET_NODES_df[['pos_R','pos_A','pos_S','Node Color','Node Size','ROI_Name']]
# Save to disk
# ============
BRAINNET_NODES_df.to_csv(BRAINNET_NODES_PATH, sep=' ', index=None, header=None)
print('++ INFO: BrainNet_Node file written to disk:   [%s]' % BRAINNET_NODES_PATH)
# Save coordinate file to disk for NBS
# ====================================
coor_file_path = osp.join(RESOURCES_NBS_DIR,ATLAS_NAME,'{ATLAS_NAME}_NBS_Node_Coordinates.txt'.format(ATLAS_NAME=ATLAS_NAME))
BRAINNET_NODES_df[['pos_R','pos_A','pos_S']].to_csv(coor_file_path, sep=' ', index=None, header=None)
print("++ INFO: NBS Coordinate file written to disk: [%s]" % coor_file_path)
# Save label file to disk for NBS
# ===============================
label_file_path = osp.join(RESOURCES_NBS_DIR,ATLAS_NAME,'{ATLAS_NAME}_NBS_Node_Labels.txt'.format(ATLAS_NAME=ATLAS_NAME))
BRAINNET_NODES_df['ROI_Name'].to_csv(label_file_path, sep=' ', index=None, header=None)
print("++ INFO: NBS Label file written to disk:      [%s]" % label_file_path)
print("++ =====================================")

# ### 5.1. Basic Scenario
#
# #### 5.1.1. Create basic design matrix
#
# This next cell will create a basic design matrix, while at the same time it copies the Z-scored FC matrices in the appropriate 
# folder (and order) for NBS

FINAL_SCAN_LIST = list(sfc_Z_xr.scan.values)
with open(DESING_MATRIX_BASIC_PATH, 'w') as the_file:
    for i,idx in tqdm(enumerate(FINAL_SCAN_LIST)):
        sbj,scan = idx.split('.')
        assert tuple(SELECTED_SCANS.reset_index().iloc[i][['Subject','Run']].values) == (sbj,scan), "ERROR --> Something is wrong with the index ordering"
        dest_path = osp.join(FC_MATRIX_DATA_FOLDER,'subject{id}.txt'.format(id=str(i+1).zfill(3)))
        np.savetxt(dest_path,sfc_Z_xr.loc[idx,:,:],delimiter=' ',fmt='%f')
        cl = SELECTED_SCANS.loc[sbj,scan]['Cluster Label']
        if cl == 'Image-Pos-Others':
            the_file.write('1 1\n')
        else:
            the_file.write('1 0\n')
the_file.close()
print('++ INFO[%s]: Basic design matrix saved to disk %s' % (ATLAS_NAME, DESING_MATRIX_BASIC_PATH))

DESIGN_MATRIX_BASIC = np.loadtxt(DESING_MATRIX_BASIC_PATH)

# +
fig,ax=plt.subplots(1,1,figsize=(2,7))
sns.heatmap(DESIGN_MATRIX_BASIC,ax=ax, cmap=['lightgray','white'], cbar=True)
ax.set_ylabel('Scan ID');
ax.set_xticklabels(['Intercept','Image-Pos-Others']);
plt.xticks(rotation=90);

# Set the colorbar labels
colorbar = ax.collections[0].colorbar
colorbar.set_ticks([0.25,0.75])
colorbar.set_ticklabels(['0', '1'])
ax.axhline(y=0, color='k',linewidth=1)
ax.axhline(y=158, color='k',linewidth=1)
ax.axvline(x=0, color='k',linewidth=1)
ax.axvline(x=1, color='k',linewidth=1)
ax.axvline(x=1.99, color='k',linewidth=1)
# -

# #### 5.1.2. Create contasts for Basic

CONTRASTS_BASIC={'Image-Pos-Others_gt_Surr-Neg-Self':[0,1],
                'Surr-Neg-Self_gt_Image-Pos-Others':[0,-1]}
for CONTRAST in CONTRASTS_BASIC.keys():
    CONTRAST_PATH = osp.join(RESOURCES_NBS_DIR,ATLAS_NAME,f'NBS_CL02_Contrast_Basic_{CONTRAST}.txt') 
    np.savetxt(CONTRAST_PATH,np.array(CONTRASTS_BASIC[CONTRAST]).reshape(1,-1),delimiter=' ',fmt='%d')

# ### 5.2. Create SbjAware NBS Configuration Files
#
# #### 5.2.1. Design Matrix
#
# Matrix with 1 col for group (-1,1) and then one column per subject
#
# Get list of unique subjects in the order the appear in the Xarray

FINAL_SBJ_LIST        = [item.split('.')[0] for item in FINAL_SCAN_LIST]
FINAL_SBJ_LIST_NOREPS = []
for sbj in FINAL_SBJ_LIST:
    if sbj not in FINAL_SBJ_LIST_NOREPS:
        FINAL_SBJ_LIST_NOREPS.append(sbj)
print('%d scans --> %d unique subjects' %(len(FINAL_SBJ_LIST),len(FINAL_SBJ_LIST_NOREPS)))

# Generate the extra columns with scans per subject information

ONE_HOT_ENCODER_SBJ      = OneHotEncoder(sparse_output=False, dtype=int)
DESIGN_MATRIX_EXTRA_COLS = ONE_HOT_ENCODER_SBJ.fit_transform(np.array(FINAL_SBJ_LIST).reshape(-1,1))
DESIGN_MATRIX_EXTRA_COLS = pd.DataFrame(DESIGN_MATRIX_EXTRA_COLS, columns=ONE_HOT_ENCODER_SBJ.categories_[0])[FINAL_SBJ_LIST_NOREPS].values

fig, ax          = plt.subplots(1,1,figsize=(20,25))
sns.heatmap(DESIGN_MATRIX_EXTRA_COLS, xticklabels=FINAL_SBJ_LIST_NOREPS, yticklabels=FINAL_SCAN_LIST, ax=ax)
plt.close()
dashboard        = pn.Row(pn.pane.Matplotlib(fig));
dashboard_server = dashboard.show(port=port_tunnel,open=False)
print('++ CLICK ON LINK TO SEE MATRIX')

# Once you are done looking at matrices, you can stop the server running this cell
dashboard_server.stop()

# +
ONE_COLUMN_GROUP_ENCODING = np.array([1 if SELECTED_SCANS.loc[i.split('.')[0],i.split('.')[1]]['Cluster Label'] == 'Image-Pos-Others' else -1 for i in FINAL_SCAN_LIST]).reshape(-1,1)
DESING_MATRIX_SBJAWARE   = np.concatenate([ONE_COLUMN_GROUP_ENCODING,DESIGN_MATRIX_EXTRA_COLS], axis=1)
np.savetxt(DESING_MATRIX_SBJAWARE_PATH,DESING_MATRIX_SBJAWARE,delimiter=' ',fmt='%d')

print('++ INFO[%s]: Augmented design matrix saved to disk %s' % (ATLAS_NAME,DESING_MATRIX_SBJAWARE_PATH))
print('++ INFO[%s]: Augmented design matrix shape %s' % (ATLAS_NAME, str(DESING_MATRIX_SBJAWARE.shape)))
# -

fig, ax          = plt.subplots(1,1,figsize=(20,25))
sns.heatmap(DESING_MATRIX_SBJAWARE, xticklabels=['Set Membership']+FINAL_SBJ_LIST_NOREPS, yticklabels=FINAL_SCAN_LIST, ax=ax, cmap=['black','gray','white'])
plt.close()
dashboard        = pn.Row(pn.pane.Matplotlib(fig));
dashboard_server = dashboard.show(port=port_tunnel,open=False)
print('++ CLICK ON LINK TO SEE MATRIX')

# Once you are done looking at matrices, you can stop the server running this cell
dashboard_server.stop()

# #### 5.2.2. Contrast Vectors

CONTRASTS_SBJAWARE={'Image-Pos-Others_gt_Surr-Neg-Self':[1]+list(np.zeros(FINAL_N_sbjs).astype(int)),
                    'Surr-Neg-Self_gt_Image-Pos-Others':[-1]+list(np.zeros(FINAL_N_sbjs).astype(int))}
for CONTRAST in CONTRASTS_SBJAWARE.keys():
    CONTRAST_PATH = osp.join(RESOURCES_NBS_DIR,ATLAS_NAME,f'NBS_CL02_Contrast_SbjAware_{CONTRAST}.txt') 
    np.savetxt(CONTRAST_PATH,np.array(CONTRASTS_SBJAWARE[CONTRAST]).reshape(1,-1),delimiter=' ',fmt='%d')
    

dashboard = pn.Column(pn.pane.Markdown('# DESING_MATRIX_SBJAWARE'),
                   pd.DataFrame(DESING_MATRIX_SBJAWARE, 
                   columns=['Group']+FINAL_SBJ_LIST_NOREPS, 
                   index=['.'.join([i,SELECTED_SCANS.loc[i.split('.')[0],i.split('.')[1]]['Cluster Label']]) for i in FINAL_SCAN_LIST]).hvplot.heatmap(width=1500,height=2000).opts(xrotation=90))
dashboard_server = dashboard.show(port=port_tunnel,open=False)
print('++ CLICK ON LINK TO SEE MATRIX')

# Once you are done looking at matrices, you can stop the server running this cell
dashboard_server.stop()

# # 6. Run NBS Analysis in Matlab
#
# Example of how to run for the ```SbjAware_Surr-Neg-Self > Image-Pos-Others``` contrast.
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
#     * ```Design Matrix = /data/SFIMJGC_Introspec/2023_fc_introspection/code/fc_introspection/resources/nbs/Schaefer2018_400Parcels_7Networks_AAL2/NBS_CL02_DesingMatrix_SubjectAware.txt```
#     * ```Contrast = /data/SFIMJGC_Introspec/2023_fc_introspection/code/fc_introspection/resources/nbs/Schaefer2018_400Parcels_7Networks_AAL2/NBS_CL02_Contrast_SbjAware_Surr-Neg-Self_gt_Image-Pos-Others.txt```
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
#     Once the program finish, please save the results as: ```/data/SFIMJGC_Introspec/2023_fc_introspection/code/fc_introspection/resources/nbs/Schaefer2018_400Parcels_7Networks_AAL2/NBS_CL02_Results/NBS_3p1/NBS_CL02_Surr-Neg-Self_gt_Image-Pos-Others.mat```
#
#

display.Image('./figures/S13_NBS_Configuration.png', width=1200)

# Once you are done looking at matrices, you can stop the server running this cell
dashboard_server.stop()
