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
# This notebook provides a first look at the FC matrices that form this data sample:
#
# * Load all connectivity matrices
#
# * Compute the average connectivity matrix across the whole sample
#
# * Plot the average FC for the whole sample
#
# * Create dashboard to fastly explore all the individual scan FC matrices

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
from utils.basics import DATA_DIR, ATLAS_NAME, ATLASES_DIR
from utils.plotting import hvplot_fc, plot_fc
from sfim_lib.io.afni import load_netcc
from scipy.spatial.distance import cosine, euclidean, correlation
import seaborn as sns
import panel as pn

import os
port_tunnel = int(os.environ['PORT2'])
print('++ INFO: Second Port available: %d' % port_tunnel)

# # 1. Load the final list of scans used in this project

sbj_list, scan_list = get_sbj_scan_list(when='post_motion', return_snycq=False)

# # 2. Load information about the Atlas and ROI needed for plotting

ATLASINFO_PATH = osp.join(ATLASES_DIR,ATLAS_NAME,'{ATLAS_NAME}_order_FSLMNI152_2mm.ranked.roi_info.csv'.format(ATLAS_NAME=ATLAS_NAME))
roi_info       = pd.read_csv(ATLASINFO_PATH)

# # 3. Load into memory all individual scan FC matrices
#
# We will place the matrices in two separate xr.DataArray data structures. One will hold the matrices in terms of Pearson's correlation (```all_sfc_R```) and the other one in terms of their Fisher's transform (```all_sfc_Z```). In the first cell below, we create empty versions of these two data structures. These empty data structures will get populated in the subsequent cell.

# Create empty Xr Data Array to hold all FC matrices
all_sfc_R = xr.DataArray(dims=['Scan','ROI1','ROI2'], 
                         coords={'Scan':[sbj+'|'+run for sbj,run in scan_list],
                               'ROI1':roi_info['ROI_Name'].values,
                               'ROI2':roi_info['ROI_Name'].values})
all_sfc_Z = xr.DataArray(dims=['Scan','ROI1','ROI2'], 
                         coords={'Scan':[sbj+'|'+run for sbj,run in scan_list],
                               'ROI1':roi_info['ROI_Name'].values,
                               'ROI2':roi_info['ROI_Name'].values})

# %%time
all_rois = list(roi_info['ROI_Name'].values)
# Load all matrices
for sbj,run in tqdm(scan_list):
    _,_,_,_,run_num,_,run_acq = run.split('-')
    netcc_path = osp.join(DATA_DIR,'PrcsData',sbj,'preprocessed','func','pb06_staticFC','{run_acq}_run-{run_num}.{ATLAS_NAME}_000.netcc'.format(run_acq = run_acq, run_num = run_num, ATLAS_NAME = ATLAS_NAME))
    netcc      = load_netcc(netcc_path)
    this_scan_rois = [ item.strip().strip('7Networks_') for item in list(netcc.columns)]
    all_sfc_R.loc[sbj+'|'+run,:,:] = netcc
    all_sfc_Z.loc[sbj+'|'+run,:,:] = netcc.apply(np.arctanh)

# # 4. Compute the average matrix for the whole sample
#
# To compute the sample mean, we first Fisher's transform each individual matrix, average those, and do the inverse transform of the average

REFERENCE_fc = np.tanh(all_sfc_Z.mean(dim='Scan')).values

hvplot_fc(REFERENCE_fc, ATLASINFO_PATH, cbar_title='Average FC for the whole sample')

# # 5. Compute Vectorized versions of all individual scan FC

Nrois  = REFERENCE_fc.shape[0]
Nconns = int(Nrois * (Nrois-1) / 2)

# Create a vectorized version of the sample mean FC
np.fill_diagonal(REFERENCE_fc,0)
REFERENCE_fc_VECT = squareform(REFERENCE_fc)
assert Nconns == len(REFERENCE_fc_VECT)

all_sfc_R_VECT = pd.DataFrame(index=np.arange(Nconns),columns=[sbj+'|'+run for sbj,run in scan_list])
for sbj,run in tqdm(scan_list):
    scan_label          = sbj+'|'+run 
    this_scan_fc_matrix = all_sfc_R.loc[scan_label].values
    np.fill_diagonal(this_scan_fc_matrix,0)
    this_scan_fc_vector = squareform(this_scan_fc_matrix)
    all_sfc_R_VECT[scan_label] = this_scan_fc_vector

# # 6. Compute Similarity to Mean (corr, cov, cosine) and get sorted list of scans

corr_to_REFERENCE    = all_sfc_R_VECT.corrwith(pd.Series(REFERENCE_fc_VECT))
corr_to_REFERENCE.name = 'R'

cov_to_REFERENCE = pd.Series(dtype=float)
for sbj,run in scan_list:
    scan_label = sbj+'|'+run
    aux        = all_sfc_R_VECT[scan_label]
    cov_to_REFERENCE[scan_label] = np.cov(REFERENCE_fc_VECT,aux)[0,1]
cov_to_REFERENCE.name = 'Covariance'

cos_to_REFERENCE = pd.Series(dtype=float)
for sbj,run in scan_list:
    scan_label = sbj+'|'+run
    aux        = all_sfc_R_VECT[scan_label]
    cos_to_REFERENCE[scan_label] = cosine(REFERENCE_fc_VECT,aux)
cos_to_REFERENCE.name = 'Cosine'

scan_list_sorted={'none':['|'.join([sbj,run]) for sbj,run in scan_list],
                  'cosine':list(cos_to_REFERENCE.sort_values(ascending=False).index),
                  'correlation':list(corr_to_REFERENCE.sort_values(ascending=False).index),
                  'covariance':list(cov_to_REFERENCE.sort_values(ascending=False).index)}

sort_method_select = pn.widgets.Select(name='Sort Method', options=list(scan_list_sorted.keys()), value='none')
@pn.depends(sort_method_select)
def plot_carpet(sorting_method):
    fig, ax = plt.subplots(1,1,figsize=(20,5))
    plot = sns.heatmap(all_sfc_R_VECT[scan_list_sorted[sorting_method]],cmap='RdBu_r', vmin=-.8, vmax=.8, xticklabels=False, yticklabels=False)
    plot.set_xlabel('Scans', fontsize=14)
    plot.set_ylabel('Connections', fontsize=14)
    plt.close()
    return fig
dashboard01 = pn.Column(sort_method_select,plot_carpet)

dashboard01_server = dashboard01.show(port=port_tunnel,open=False)

dashboard01_server.stop()

# # 7. Plot Distribution of correlatations towards the REFERENCE

(corr_to_REFERENCE.hvplot.hist(bins=50, normed=True, title='Corr(Scan,REF)', xlabel='Correlation to Sample Mean', ylabel='Distribution', fontsize=14, xlim=(0,1), width=500) \
* corr_to_REFERENCE.hvplot.kde()) + \
(cov_to_REFERENCE.hvplot.hist(bins=50, normed=True, title='Cov(Scan,REF)', xlabel='Covariance to Sample Mean', ylabel='Distribution', fontsize=14, width=500) \
* cov_to_REFERENCE.hvplot.kde()) + \
(cos_to_REFERENCE.hvplot.hist(bins=50, normed=True, title='Cosine(Scan,REF)', xlabel='Covariance to Sample Mean', ylabel='Distribution', fontsize=14, width=500) \
* cos_to_REFERENCE.hvplot.kde())

# # 8. Generate dashboard to explore all individual scan FC matrices
#
# > **NOTE**: If you run the code within the notebook, the dashboard does not update when you hit play. Make sure to open it by clicking on the URL that the show command below returns

# This is needed to ensure the labels are not cut when serving the matplotlib figures via pn.Column
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

Nscans          = len(scan_list)
scan_selector   = pn.widgets.Player(name='Scan Number', start=0, end=Nscans-1, value=0, width=700)
@pn.depends(scan_selector,sort_method_select)
def plot_one_scan(scan_num,sorting_method):
    aux_scan_list             = scan_list_sorted[sorting_method]
    scan_name                 = aux_scan_list[scan_num]
    sbj, run                  = scan_name.split('|')
    _,_,_,_,run_num,_,run_acq = (run).split('-')
    scan_label = sbj + ',' + run_num+'_'+run_acq + ' | R = %.3f' % corr_to_REFERENCE[scan_name] + ' | Cos = %.3f' % cos_to_REFERENCE[scan_name] + ' | Cov = %.3f' % cov_to_REFERENCE[scan_name]
    data = all_sfc_R.loc[scan_name].values
    fig  = plot_fc(data,ATLASINFO_PATH, figsize=(10,10))
    return pn.Card(pn.pane.Matplotlib(fig),title=scan_label,collapsible=False)


dashboard = pn.Column(sort_method_select,scan_selector,plot_one_scan)

dashboard_server = dashboard.show(port=port_tunnel,open=False)

# Once you are done looking at matrices, you can stop the server running this cell
dashboard_server.stop()

# # 8. Explore extreme cases

# ### Show the matrix most different to the sample mean

most_different_scan      = corr_to_REFERENCE.sort_values().index[0]
most_different_scan_corr = corr_to_REFERENCE.sort_values()[0]
print('INFO: Most dissimilar scan to the sample mean [%s --> R(scan,mean) = %.2f ]' % (most_different_scan,most_different_scan_corr))
most_different_matrix = all_sfc_R.loc[most_different_scan,:,:].values
np.fill_diagonal(most_different_matrix,1)
hvplot_fc(most_different_matrix, ATLASINFO_PATH)

# ### Show the matrix most similar to the sample mean

most_similar_scan      = corr_to_REFERENCE.sort_values(ascending=False).index[0]
most_similar_scan_corr = corr_to_REFERENCE.sort_values(ascending=False)[0]
print('INFO: Most dissimilar scan to the sample mean [%s --> R(scan,mean) = %.2f ]' % (most_similar_scan,most_similar_scan_corr))
most_similar_matrix = all_sfc_R.loc[most_similar_scan,:,:].values
np.fill_diagonal(most_similar_matrix,1)
hvplot_fc(most_similar_matrix, ATLASINFO_PATH)

median_selector          = corr_to_REFERENCE[corr_to_REFERENCE==corr_to_REFERENCE.median()]
median_similar_scan      = median_selector.index[0]
median_similar_scan_corr = median_selector.values
print('INFO: Scan with median similarity to the sample mean [%s --> R(scan,mean) = %.2f ]' % (median_similar_scan,median_similar_scan_corr))
median_similar_matrix = all_sfc_R.loc[median_similar_scan,:,:].values
np.fill_diagonal(median_similar_matrix,1)
hvplot_fc(median_similar_matrix, ATLASINFO_PATH)
