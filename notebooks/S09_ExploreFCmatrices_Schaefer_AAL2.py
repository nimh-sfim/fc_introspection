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
# This notebook provides a first look at the FC matrices that form this data sample:
#
# * Load all connectivity matrices
#
# * Compute the average connectivity matrix across the whole sample
#
# * Plot the average FC for the whole sample
#
# * Create dashboard to fastly explore all the individual scan FC matrices

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import xarray as xr
import numpy as np
import os.path as osp
import hvplot.pandas
from tqdm import tqdm
import matplotlib.pyplot as plt
#import holoviews as hv
from utils.basics import get_sbj_scan_list
from scipy.spatial.distance import squareform
from utils.basics import DATA_DIR, CORTICAL_400ROI_ATLAS_NAME, FB_400ROI_ATLAS_NAME, ATLASES_DIR, FB_200ROI_ATLAS_NAME
from utils.plotting import hvplot_fc, plot_fc
from sfim_lib.io.afni import load_netcc
from scipy.spatial.distance import cosine      as cosine_distance
from scipy.spatial.distance import correlation as correlation_distance
from scipy.spatial.distance import euclidean   as euclidean_distance
import seaborn as sns
import panel as pn
from sklearn.utils.validation import check_symmetric

import os
port_tunnel = int(os.environ['PORT2'])
print('++ INFO: Second Port available: %d' % port_tunnel)

ATLAS_NAME = FB_400ROI_ATLAS_NAME

# # 1. Load the final list of scans used in this project

sbj_list, scan_list = get_sbj_scan_list(when='post_motion', return_snycq=False)

# # 2. Load information about the Atlas and ROI needed for plotting

ATLASINFO_PATH = osp.join(ATLASES_DIR,ATLAS_NAME,f'{ATLAS_NAME}.roi_info.csv')
roi_info       = pd.read_csv(ATLASINFO_PATH)
roi_info

# # 3. Load into memory all individual scan FC matrices
#
# We will place the matrices in two separate xr.DataArray data structures. One will hold the matrices in terms of Pearson's correlation (```all_sfc_R```) and the other one in terms of their Fisher's transform (```all_sfc_Z```). In the first cell below, we create empty versions of these two data structures. These empty data structures will get populated in the subsequent cell.

unique_sbj_ids = list(pd.Series([sbj for sbj,_ in scan_list]).unique())
unique_run_ids = list(pd.Series([run for _,run in scan_list]).unique())

# Create empty Xr Data Array to hold all FC matrices
all_sfc_R = xr.DataArray(dims=['Subject','Run','ROI1','ROI2'], 
                         coords={'Subject':unique_sbj_ids,
                                 'Run': unique_run_ids,
                                 'ROI1':roi_info['ROI_Name'].values,
                                 'ROI2':roi_info['ROI_Name'].values})
all_sfc_Z = xr.DataArray(dims=['Subject','Run','ROI1','ROI2'], 
                         coords={'Subject':unique_sbj_ids,
                                 'Run': unique_run_ids,
                                 'ROI1':roi_info['ROI_Name'].values,
                                 'ROI2':roi_info['ROI_Name'].values})

# %%time
all_rois = list(roi_info['ROI_Name'].values)
# Load all matrices
for sbj,run in tqdm(scan_list):
    _,_,_,_,run_num,_,run_acq = run.split('-')
    netcc_path = osp.join(DATA_DIR,'PrcsData',sbj,'preprocessed','func','pb06_staticFC',f'{run_acq}_run-{run_num}.{ATLAS_NAME}_000.netcc')
    netcc      = load_netcc(netcc_path)
    this_scan_rois = [ item.strip().strip('7Networks_') for item in list(netcc.columns)]
    all_sfc_R.loc[sbj,run,:,:] = netcc
    all_sfc_Z.loc[sbj,run,:,:] = netcc.apply(np.arctanh)

# + active=""
# # Create empty Xr Data Array to hold all FC matrices
# all_sfc_R = xr.DataArray(dims=['Scan','ROI1','ROI2'], 
#                          coords={'Scan':[sbj+'|'+run for sbj,run in scan_list],
#                                'ROI1':roi_info['ROI_Name'].values,
#                                'ROI2':roi_info['ROI_Name'].values})
# all_sfc_Z = xr.DataArray(dims=['Scan','ROI1','ROI2'], 
#                          coords={'Scan':[sbj+'|'+run for sbj,run in scan_list],
#                                'ROI1':roi_info['ROI_Name'].values,
#                                'ROI2':roi_info['ROI_Name'].values})

# + active=""
# %%time
# all_rois = list(roi_info['ROI_Name'].values)
# # Load all matrices
# for sbj,run in tqdm(scan_list):
#     _,_,_,_,run_num,_,run_acq = run.split('-')
#     netcc_path = osp.join(DATA_DIR,'PrcsData',sbj,'preprocessed','func','pb06_staticFC','{run_acq}_run-{run_num}.{ATLAS_NAME}_000.netcc'.format(run_acq = run_acq, run_num = run_num, ATLAS_NAME = FB_400ROI_ATLAS_NAME))
#     netcc      = load_netcc(netcc_path)
#     this_scan_rois = [ item.strip().strip('7Networks_') for item in list(netcc.columns)]
#     all_sfc_R.loc[sbj+'|'+run,:,:] = netcc
#     all_sfc_Z.loc[sbj+'|'+run,:,:] = netcc.apply(np.arctanh)
# -

# # 4. Compute the average matrix for the whole sample
#
# To compute the sample mean, we first Fisher's transform each individual matrix, average those, and do the inverse transform of the average

REFERENCE_fc = np.tanh(all_sfc_Z.mean(dim=['Subject','Run'])).values

# Put the matrix into a properly annotated DataFrame structure

REFERENCE_fc = pd.DataFrame(REFERENCE_fc, columns=list(roi_info['ROI_Name']),index=list(roi_info['ROI_Name']))
REFERENCE_fc.index.name   = 'ROI1'
REFERENCE_fc.columns.name = 'ROI2'

# Plot the sample mean (or Reference) FC matrix

hvplot_fc(REFERENCE_fc, ATLASINFO_PATH, cbar_title='Average FC for the whole sample', cmap='RdBu_r', major_label_overrides = 'regular_grid')

# # 5. Explore individual subject matrices

sbj_level_sfc_R = np.arctan(all_sfc_Z.mean(dim='Run'))

sbj_select = pn.widgets.Select(name='Subject', options=unique_sbj_ids)
@pn.depends(sbj_select)
def plot_subject_fc(sbj):
    this_subject_mat = pd.DataFrame(sbj_level_sfc_R.loc[sbj].values, index=list(sbj_level_sfc_R.coords['ROI1'].values), columns=list(sbj_level_sfc_R.coords['ROI2'].values))
    return hvplot_fc(this_subject_mat, ATLASINFO_PATH, cbar_title='FC '+sbj, cmap='RdBu_r', major_label_overrides = 'regular_grid')
sbj_mat_dashboard = pn.Row(sbj_select, plot_subject_fc)

sbj_mat_dashboard_server = sbj_mat_dashboard.show(port=port_tunnel,open=False)

sbj_mat_dashboard_server.stop()
