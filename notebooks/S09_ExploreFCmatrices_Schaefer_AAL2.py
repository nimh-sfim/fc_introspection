#!/usr/bin/env python
# coding: utf-8

# # Description: Dashboard to inspect FC matrices for all scans
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
# 
# > NOTE: This notebook is intended for QC. It does not generate any results presented in the manuscript

# In[2]:


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# In[3]:


import pandas as pd
import xarray as xr
import numpy as np
import os.path as osp
import hvplot.pandas
from tqdm import tqdm
from utils.basics import get_sbj_scan_list
from utils.basics import DATA_DIR, FB_400ROI_ATLAS_NAME, ATLASES_DIR
from utils.plotting import hvplot_fc
from sfim_lib.io.afni import load_netcc
import panel as pn


# In[4]:


ATLAS_NAME = FB_400ROI_ATLAS_NAME
print(ATLAS_NAME)


# # 1. Load the final list of scans used in this project

# In[5]:


sbj_list, scan_list = get_sbj_scan_list(when='post_motion', return_snycq=False)


# # 2. Load information about the Atlas and ROI needed for plotting

# In[6]:


ATLASINFO_PATH = osp.join(ATLASES_DIR,ATLAS_NAME,f'{ATLAS_NAME}.roi_info.csv')
roi_info       = pd.read_csv(ATLASINFO_PATH)
roi_info


# # 3. Load into memory all individual scan FC matrices
# 
# We will place the matrices in two separate xr.DataArray data structures. One will hold the matrices in terms of Pearson's correlation (```all_sfc_R```) and the other one in terms of their Fisher's transform (```all_sfc_Z```). In the first cell below, we create empty versions of these two data structures. These empty data structures will get populated in the subsequent cell.

# In[7]:


unique_sbj_ids = list(pd.Series([sbj for sbj,_ in scan_list]).unique())
unique_run_ids = list(pd.Series([run for _,run in scan_list]).unique())


# In[8]:


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


# In[9]:


get_ipython().run_cell_magic('time', '', "all_rois = list(roi_info['ROI_Name'].values)\n# Load all matrices\nfor sbj,run in tqdm(scan_list):\n    _,_,_,_,run_num,_,run_acq = run.split('-')\n    netcc_path = osp.join(DATA_DIR,'PrcsData',sbj,'preprocessed','func','pb06_staticFC',f'{run_acq}_run-{run_num}.{ATLAS_NAME}_000.netcc')\n    netcc      = load_netcc(netcc_path)\n    this_scan_rois = [ item.strip().strip('7Networks_') for item in list(netcc.columns)]\n    all_sfc_R.loc[sbj,run,:,:] = netcc\n    all_sfc_Z.loc[sbj,run,:,:] = netcc.apply(np.arctanh)\n")


# # 4. Compute the average matrix for the whole sample
# 
# To compute the sample mean, we first Fisher's transform each individual matrix, average those, and do the inverse transform of the average

# In[10]:


REFERENCE_fc = np.tanh(all_sfc_Z.mean(dim=['Subject','Run'])).values


# Put the matrix into a properly annotated DataFrame structure

# In[11]:


REFERENCE_fc = pd.DataFrame(REFERENCE_fc, columns=list(roi_info['ROI_Name']),index=list(roi_info['ROI_Name']))
REFERENCE_fc.index.name   = 'ROI1'
REFERENCE_fc.columns.name = 'ROI2'


# Plot the sample mean (or Reference) FC matrix

# In[12]:


hvplot_fc(REFERENCE_fc, ATLASINFO_PATH, cbar_title='Average FC for the whole sample', cmap='RdBu_r', major_label_overrides = 'regular_grid')


# # 5. Explore individual subject matrices

# In[13]:


sbj_level_sfc_R = np.arctan(all_sfc_Z.mean(dim='Run'))


# In[14]:


sbj_select = pn.widgets.Select(name='Subject', options=unique_sbj_ids)
@pn.depends(sbj_select)
def plot_subject_fc(sbj):
    this_subject_mat = pd.DataFrame(sbj_level_sfc_R.loc[sbj].values, index=list(sbj_level_sfc_R.coords['ROI1'].values), columns=list(sbj_level_sfc_R.coords['ROI2'].values))
    return hvplot_fc(this_subject_mat, ATLASINFO_PATH, cbar_title='FC '+sbj, cmap='RdBu_r', major_label_overrides = 'regular_grid')
sbj_mat_dashboard = pn.Row(sbj_select, plot_subject_fc)


# In[ ]:


sbj_mat_dashboard_server = sbj_mat_dashboard.show()


# In[16]:


sbj_mat_dashboard_server.stop()

