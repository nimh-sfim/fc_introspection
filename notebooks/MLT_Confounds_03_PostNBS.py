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

import pandas as pd
import numpy as np
import os.path as osp
import hvplot.pandas
from utils.basics import FB_ATLAS_NAME, ATLASES_DIR, CORTICAL_ATLAS_NAME
from utils.plotting import hvplot_fc, plot_fc, hvplot_fc_nwlevel, plot_as_circos
import holoviews as hv
from holoviews import opts

# # 2. Load information about the Atlas and ROI needed for plotting

SOLUTION = 'CL02_0p001'
ATLAS_NAME = FB_ATLAS_NAME

ATLASINFO_PATH = osp.join(ATLASES_DIR,ATLAS_NAME,'{ATLAS_NAME}.roi_info.csv'.format(ATLAS_NAME=ATLAS_NAME))
roi_info       = pd.read_csv(ATLASINFO_PATH)

data_f1GTf2 = np.loadtxt('/data/SFIMJGC_Introspec/2023_fc_introspection/code/fc_introspection/resources/mtl_snycq/nbs/{ATLAS_NAME}/NBS_{s}_Results/NBS_{s}_F1gtF2.edge'.format(s=SOLUTION, ATLAS_NAME=ATLAS_NAME))
data_f1GTf2 = pd.DataFrame(data_f1GTf2,
                           index   = roi_info.set_index(['Hemisphere','Network','ROI_Name','ROI_ID']).index, 
                           columns = roi_info.set_index(['Hemisphere','Network','ROI_Name','ROI_ID']).index)

data_f2GTf1 = np.loadtxt('/data/SFIMJGC_Introspec/2023_fc_introspection/code/fc_introspection/resources/mtl_snycq/nbs/{ATLAS_NAME}/NBS_{s}_Results/NBS_{s}_F2gtF1.edge'.format(s=SOLUTION, ATLAS_NAME=ATLAS_NAME))
data_f2GTf1 = pd.DataFrame(data_f2GTf1,
                           index   = roi_info.set_index(['Hemisphere','Network','ROI_Name','ROI_ID']).index, 
                           columns = roi_info.set_index(['Hemisphere','Network','ROI_Name','ROI_ID']).index)

data_f1GTf2

hvplot_fc(data_f1GTf2,ATLASINFO_PATH, cbar_title='F1 > F2', verbose=True) + hvplot_fc(data_f2GTf1,ATLASINFO_PATH, cbar_title='F2 > F1') + hvplot_fc(data_f1GTf2-data_f2GTf1,ATLASINFO_PATH, cbar_title='F1 > F2', verbose=True)

# ***

hvplot_fc_nwlevel(data_f1GTf2, title='F1 > F2', add_net_colors=True) + hvplot_fc_nwlevel(data_f2GTf1, title='F2 > F1', add_net_colors=True) + \
hvplot_fc_nwlevel(data_f1GTf2+data_f2GTf1, title='Between-Group Differences',add_net_colors=True)

hvplot_fc_nwlevel(data_f1GTf2, mode='count', title='F1 > F2') + hvplot_fc_nwlevel(data_f2GTf1, mode='count', title='F2 > F1') + \
hvplot_fc_nwlevel(data_f1GTf2+data_f2GTf1,  mode='count', title='')

# ***

data = data_f1GTf2 - data_f2GTf1

plot_as_circos(data,roi_info, edge_weight=.5, figsize=(7,7))

plot_as_circos(data_f1GTf2, roi_info, edge_weight=.5, figsize=(7,7))

# +
# plot_as_circos?
# -


