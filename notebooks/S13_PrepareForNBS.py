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

import os.path as osp
import os
import pandas as pd
import numpy as np
import xarray as xr
from sfim_lib.io.afni import load_netcc
from shutil import rmtree

from utils.basics import RESOURCES_NBS_DIR, CORTICAL_ATLAS_NAME, CORTICAL_ATLAS_PATH
from utils.basics import ATLASES_DIR, CORTICAL_ATLAS_PATH, SNYCQ_CLUSTERS_INFO_PATH, BRAINNET_NODES_PATH, DATA_DIR
from utils.basics import get_sbj_scan_list

if not osp.exists(RESOURCES_NBS_DIR):
    os.makedirs(RESOURCES_NBS_DIR)

# # 1. Create the BrainNet_Nodes.node, NBS Coordinates and Labels files

ATLASINFO_PATH = osp.join(ATLASES_DIR,ATLAS_NAME,'{ATLAS_NAME}_order_FSLMNI152_2mm.ranked.roi_info.csv'.format(ATLAS_NAME=ATLAS_NAME))
roi_info       = pd.read_csv(ATLASINFO_PATH)

Nrois = roi_info.shape[0]
print(Nrois)

Nw2Id                           = {'Vis':1,'SomMot':2,'DorsAttn':3,'SalVentAttn':4,'Limbic':5,'Cont':6,'Default':7}
BRAINNET_NODES_df               = roi_info[['pos_R','pos_A','pos_S','ROI_Name']].copy()
BRAINNET_NODES_df['Node Size']  = 1
BRAINNET_NODES_df['Node Color'] = [Nw2Id[n.split('_')[1]] for n in BRAINNET_NODES_df['ROI_Name']]
BRAINNET_NODES_df = BRAINNET_NODES_df[['pos_R','pos_A','pos_S','Node Color','Node Size','ROI_Name']]

BRAINNET_NODES_df['Node Color'].unique()

BRAINNET_NODES_df.to_csv(BRAINNET_NODES_PATH, sep=' ', index=None, header=None)
print('++ INFO: BrainNet_Node file written to disk [%s]' % BRAINNET_NODES_PATH)

BRAINNET_NODES_df[['pos_R','pos_A','pos_S']].to_csv(osp.join(RESOURCES_NBS_DIR,'NBS_Node_Coordinates.txt'), sep=' ', index=None, header=None)
BRAINNET_NODES_df['ROI_Name'].to_csv(osp.join(RESOURCES_NBS_DIR,'NBS_Node_Labels.txt'), sep=' ', index=None, header=None)

# ***
# # 2. Create Design Matrix for NBS

clusters_info = pd.read_csv(SNYCQ_CLUSTERS_INFO_PATH, index_col=['Subject','Run'])

clusters_info[clusters_info['Cluster Label']=='Large F1']

N_clusters = len(clusters_info['Cluster ID'].unique())
print("++ INFO: Number of Clusters = %d clusters" % N_clusters)

scans_per_cluster={cl_label:clusters_info[clusters_info['Cluster Label']==cl_label].index for cl_label in ['Large F1','Large F2','Middle']}

DESIGN_MATRIX = np.vstack([np.tile(np.array([1,0,0]),(len(scans_per_cluster['Large F1']),1)),
                           np.tile(np.array([0,1,0]),(len(scans_per_cluster['Large F2']),1)),
                           np.tile(np.array([0,0,1]),(len(scans_per_cluster['Middle']),  1))])
DESIGN_MATRIX_PATH = osp.join(RESOURCES_NBS_DIR,'NBS_CL03_DesingMatrix.txt')
np.savetxt(DESIGN_MATRIX_PATH,DESIGN_MATRIX,delimiter=' ',fmt='%d')
print('++ INFO: Design Matrix for 3 Cluster solution saved in [%s]' % DESIGN_MATRIX_PATH)
print('++ INFO: Design Matrix for 3 Cluster solution has shape [%s]' % str(DESIGN_MATRIX.shape))

DESIGN_MATRIX = np.vstack([np.tile(np.array([1,0]),(len(scans_per_cluster['Large F1']),1)),
                           np.tile(np.array([0,1]),(len(scans_per_cluster['Large F2']),1))])
DESIGN_MATRIX_PATH = osp.join(RESOURCES_NBS_DIR,'NBS_CL02_DesingMatrix.txt')
np.savetxt(DESIGN_MATRIX_PATH,DESIGN_MATRIX,delimiter=' ',fmt='%d')
print('++ INFO: Design Matrix for 3 Cluster solution saved in [%s]' % DESIGN_MATRIX_PATH)
print('++ INFO: Design Matrix for 3 Cluster solution has shape [%s]' % str(DESIGN_MATRIX.shape))

# ***
# # 3. Create Copies of Matrices in NBS folder

_,scans_list = get_sbj_scan_list(when='post_motion', return_snycq=False)
Nscans = len(scans_list)

# %%time
sfc_Z_arr = np.empty((Nscans,Nrois,Nrois)) * np.nan
scan_idx      = 0
scan_name_idx = []
for cluster_id in ['Large F1', 'Large F2','Middle']:
    for sbj,run in scans_per_cluster[cluster_id]:
        scan_name_idx.append('.'.join([sbj,run]))
        _,_,sesID,_,runID,_,acqID = run.split('-')
        sfc_path = osp.join(DATA_DIR,'PrcsData',sbj,'preprocessed','func','pb06_staticFC','{acqID}_run-{runID}.Schaefer2018_200Parcels_7Networks_000.netcc'.format(acqID=acqID,runID=runID))
        aux_cc_r = load_netcc(sfc_path)
        aux_cc_Z = aux_cc_r.apply(np.arctanh)
        np.fill_diagonal(aux_cc_Z.values,1)
        sfc_Z_arr[scan_idx,:,:] = aux_cc_Z
        scan_idx = scan_idx + 1

sfc_Z_xr = xr.DataArray(sfc_Z_arr,
                        dims=['scan','roi_x','roi_y'],
                        coords={'scan':scan_name_idx,
                                'roi_x':roi_info['ROI_ID'],
                                'roi_y':roi_info['ROI_ID']})

# #### Create Data Folder for this solution

NBS_CL03_matrices_folder = osp.join(RESOURCES_NBS_DIR,'NBS_CL03_Data')
if osp.exists(NBS_CL03_matrices_folder):
    rmtree(NBS_CL03_matrices_folder)
os.mkdir(NBS_CL03_matrices_folder)

# #### Make copies of matrices into the folder

# %%time
for i,item in enumerate(list(sfc_Z_xr.indexes['scan'])):
    dest_path = osp.join(NBS_CL03_matrices_folder,'subject{id}.txt'.format(id=str(i+1).zfill(3)))
    np.savetxt(dest_path,sfc_Z_xr.loc[item,:,:],delimiter=' ',fmt='%f')

# ***
# # 3. Create Copies of Matrices in NBS folder

Nscans = clusters_info.set_index('Cluster Label').loc[['Large F1','Large F2']].shape[0]

# %%time
sfc_Z_arr = np.empty((Nscans,Nrois,Nrois)) * np.nan
scan_idx      = 0
scan_name_idx = []
for cluster_id in ['Large F1', 'Large F2']:
    for sbj,run in scans_per_cluster[cluster_id]:
        scan_name_idx.append('.'.join([sbj,run]))
        _,_,sesID,_,runID,_,acqID = run.split('-')
        sfc_path = osp.join(DATA_DIR,'PrcsData',sbj,'preprocessed','func','pb06_staticFC','{acqID}_run-{runID}.Schaefer2018_200Parcels_7Networks_000.netcc'.format(acqID=acqID,runID=runID))
        aux_cc_r = load_netcc(sfc_path)
        aux_cc_Z = aux_cc_r.apply(np.arctanh)
        np.fill_diagonal(aux_cc_Z.values,1)
        sfc_Z_arr[scan_idx,:,:] = aux_cc_Z
        scan_idx = scan_idx + 1

sfc_Z_xr = xr.DataArray(sfc_Z_arr,
                        dims=['scan','roi_x','roi_y'],
                        coords={'scan':scan_name_idx,
                                'roi_x':roi_info['ROI_ID'],
                                'roi_y':roi_info['ROI_ID']})

# #### Create Data Folder for this solution

NBS_CL02_matrices_folder = osp.join(RESOURCES_NBS_DIR,'NBS_CL02_Data')
if osp.exists(NBS_CL02_matrices_folder):
    rmtree(NBS_CL02_matrices_folder)
os.mkdir(NBS_CL02_matrices_folder)

# #### Make copies of matrices into the folder

# %%time
for i,item in enumerate(list(sfc_Z_xr.indexes['scan'])):
    dest_path = osp.join(NBS_CL02_matrices_folder,'subject{id}.txt'.format(id=str(i+1).zfill(3)))
    np.savetxt(dest_path,sfc_Z_xr.loc[item,:,:],delimiter=' ',fmt='%f')


