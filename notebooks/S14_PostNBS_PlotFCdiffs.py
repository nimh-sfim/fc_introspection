# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: FC Instrospection py 3.10 | 2023b
#     language: python
#     name: fc_introspection_2023b_py310
# ---

# # Description - Exploration of NBS Results
#
# This notebook takes the outputs from running NBS and plots then for interpretation. It was used to generate the Circos plots of interest for NBS models

# +
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import numpy as np
import os.path as osp
import hvplot.pandas
from utils.basics import FB_400ROI_ATLAS_NAME as ATLAS_NAME
from utils.basics import ATLASES_DIR, RESOURCES_NIMARE_DIR, RESOURCES_CONN_DIR, RESOURCES_NBS_DIR
from utils.plotting import hvplot_fc, hvplot_fc_nwlevel, create_graph_from_matrix, plot_as_graph
import holoviews as hv
from holoviews import opts
from IPython import display
import panel as pn
import matplotlib.pyplot as plt
from nilearn.plotting import plot_connectome
from matplotlib.colors import LinearSegmentedColormap
from sklearn.preprocessing import MinMaxScaler
from nilearn.image import load_img
from nilearn import masking
# -

NBS_CONTRASTS = ['SetA_gt_SetB','SetB_gt_SetA']

# + vscode={"languageId": "raw"} active=""
# SOLUTION      = 'CL02'
# THRESHOLD     = 'NBS_3p1'
# DESIGN_MATRIX = 'SbjAware'
# NBS_CONTRASTS = ['Image-Pos-Others_gt_Surr-Neg-Self','Surr-Neg-Self_gt_Image-Pos-Others']
# -

# # 2. Load information about the Atlas and ROI needed for plotting
#
# Load the data structure with information about the ROIs in the atlas

ATLASINFO_PATH = osp.join(ATLASES_DIR,ATLAS_NAME,'{ATLAS_NAME}.roi_info.csv'.format(ATLAS_NAME=ATLAS_NAME))
roi_info       = pd.read_csv(ATLASINFO_PATH)
Nrois          = roi_info.shape[0]
Nedges         = int(Nrois*(Nrois-1)/2)
print(Nrois,Nedges)

roi_info[roi_info['ROI_Name']=='LH_SalVentAttn_Med_5']

# Count the number of networks and get their names

networks = list(roi_info['Network'].unique())
print(networks, len(networks))

# Load the connections that are significantly stronger for the contrast: $$Image-Pos-Others > Surr-Neg-Self$$
# and the contrast: $$Surr-Neg-Self > Image-Pos-Others$$

data = {}
for contrast in NBS_CONTRASTS:
    aux_path = osp.join(RESOURCES_NBS_DIR,ATLAS_NAME,f'NBS_Results',f'NBS_{contrast}.edge')
    if osp.exists(aux_path):
        aux_data = np.loadtxt(aux_path)
        data[contrast]  = pd.DataFrame(aux_data,
                                         index   = roi_info.set_index(['Hemisphere','Network','ROI_Name','ROI_ID','RGB']).index, 
                                         columns = roi_info.set_index(['Hemisphere','Network','ROI_Name','ROI_ID','RGB']).index)
        print('++ INFO: Data available for %s' % contrast)
    else:
        data[contrast] = pd.DataFrame(np.zeros((Nrois,Nrois)),
                                     index   = roi_info.set_index(['Hemisphere','Network','ROI_Name','ROI_ID','RGB']).index, 
                                     columns = roi_info.set_index(['Hemisphere','Network','ROI_Name','ROI_ID','RGB']).index)
        print('++ WARTNING: No results available for %s' % contrast)

N_sig_edges = data['SetA_gt_SetB'].sum().sum()/2
PC_sig_edges = 100 * N_sig_edges / Nedges
N_sig_edges,PC_sig_edges


N_sig_nodes  = (data['SetA_gt_SetB'].sum() > 0).sum()
PC_sig_nodes = 100 * N_sig_nodes / N_sig_nodes
N_sig_nodes,PC_sig_nodes

data['Both'] = data['SetA_gt_SetB'] - data['SetB_gt_SetA']

# We will also write the results of NBS into text format that we can load into CONN to generate the brain views of the results

RESOURCES_CONN_DIR

for contrast in data.keys():
    if data[contrast] is not None:
        aux_path = osp.join(RESOURCES_CONN_DIR,f'NBS_{contrast}.txt')
        np.savetxt(aux_path,data[contrast].values)
        print("++ INFO: Contrast data [%s] saved to disk %s" %(contrast,aux_path))

# # Plot results at the individual connection level

hvplot_fc(data['Both'].loc[:,networks,:].T.loc[:,networks,:].T, by='Network', add_color_segments=True, add_labels=True, cmap=['#ED7D31','#ffffff', '#4472C4'], major_label_overrides={-0.5:'Set B > Set A',0:'',0.5:'Set A > Set B'}, colorbar_position='top').opts(toolbar=None, title='Both')

plot_as_graph(data['SetA_gt_SetB'], edge_weight=.5, show_hemi_labels=False,pos_edges_color='k')

data['SetA_gt_SetB'].sum(axis=1).sort_values(ascending=False)

hvplot_fc_nwlevel(data['SetA_gt_SetB'], title='', add_net_colors=True, add_net_labels='both', mode='count', cmap='Greys', clim_max=100, labels_text_color='Greys_r').opts(toolbar=None)

hvplot_fc_nwlevel(data['Image-Pos-Others_gt_Surr-Neg-Self'], title='', add_net_colors=True, add_net_labels='y', mode='count', cmap='Reds', clim_max=30, labels_text_color='Reds_r').opts(toolbar=None)

# # Laterality Index for each contrast
#

aux         = (data['SetA_gt_SetB']).copy()
aux.index   = data['SetA_gt_SetB'].index.get_level_values('Hemisphere')
aux.columns = data['SetA_gt_SetB'].columns.get_level_values('Hemisphere')
f2GTf1_LL   = (aux.loc['LH','LH'].sum().sum() / 2)
f2GTf1_RR   = (aux.loc['RH','RH'].sum().sum() / 2)
f2GTf1_LR   = aux.loc['LH','RH'].sum().sum()
print('++ INFO [SetA > SetB] L-L Conns: %d' % f2GTf1_LL)
print('++ INFO [SetA > SetB] R-R Conns: %d' % f2GTf1_RR)
print('++ INFO [SetA > SetB] R-L Conns: %d' % f2GTf1_LR)
print('++ --------------------------------------------------------')
f2GTf1_fcLI  = (f2GTf1_LL - f2GTf1_RR) / (f2GTf1_LL + f2GTf1_RR)
print('++ INFO [SetA > SetB] fcLI:      %.2f' % f2GTf1_fcLI)


