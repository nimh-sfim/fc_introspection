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
from utils.basics import ATLASES_DIR, RESOURCES_NIMARE_DIR, RESOURCES_CONN_DIR, FB_200ROI_ATLAS_NAME, RESOURCES_NBS_DIR
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

import os
port_tunnel = int(os.environ['PORT2'])
print('++ INFO: Second Port available: %d' % port_tunnel)

SOLUTION      = 'CL02'
THRESHOLD     = 'NBS_3p1'
DESIGN_MATRIX = 'SbjAware'
NBS_CONTRASTS = ['Image-Pos-Others_gt_Surr-Neg-Self','Surr-Neg-Self_gt_Image-Pos-Others']

# # 2. Load information about the Atlas and ROI needed for plotting
#
# Load the data structure with information about the ROIs in the atlas

ATLASINFO_PATH = osp.join(ATLASES_DIR,ATLAS_NAME,'{ATLAS_NAME}.roi_info.csv'.format(ATLAS_NAME=ATLAS_NAME))
roi_info       = pd.read_csv(ATLASINFO_PATH)
Nrois          = roi_info.shape[0]
print(Nrois)

# Count the number of networks and get their names

networks = list(roi_info['Network'].unique())
print(networks, len(networks))

# Load the connections that are significantly stronger for the contrast: $$Image-Pos-Others > Surr-Neg-Self$$
# and the contrast: $$Surr-Neg-Self > Image-Pos-Others$$

data = {}
for contrast in NBS_CONTRASTS:
    aux_path = osp.join(RESOURCES_NBS_DIR,ATLAS_NAME,f'NBS_{SOLUTION}_Results',THRESHOLD,DESIGN_MATRIX,f'NBS_{SOLUTION}_{contrast}.edge')
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

data['Surr-Neg-Self_gt_Image-Pos-Others'].sum().sum()

data['Both'] = data['Image-Pos-Others_gt_Surr-Neg-Self'] - data['Surr-Neg-Self_gt_Image-Pos-Others']

# We will also write the results of NBS into text format that we can load into CONN to generate the brain views of the results

for contrast in data.keys():
    if data[contrast] is not None:
        aux_path = osp.join(RESOURCES_CONN_DIR,f'NBS_{DESIGN_MATRIX}_{THRESHOLD}_{contrast}.txt')
        np.savetxt(aux_path,data[contrast].values)
        print("++ INFO: Contrast data [%s] saved to disk %s" %(contrast,aux_path))

# # Plot results at the individual connection level

hvplot_fc(data['Both'].loc[:,networks,:].T.loc[:,networks,:].T, by='Network', add_color_segments=True, add_labels=True, cmap=['#4472C4','#ffffff','#ED7D31'], major_label_overrides={-0.5:'Surr-Neg-Self > Images-Pos-Others',0:'',0.5:'Images-Pos-Others > Surr-Neg-Self'}, colorbar_position='top').opts(toolbar=None, title=f'{DESIGN_MATRIX} | {THRESHOLD} | Both')

plot_as_graph(data['Surr-Neg-Self_gt_Image-Pos-Others'], edge_weight=.5, show_hemi_labels=False,pos_edges_color='k')

data['Surr-Neg-Self_gt_Image-Pos-Others'].sum(axis=1).sort_values(ascending=False)

hvplot_fc_nwlevel(data['Surr-Neg-Self_gt_Image-Pos-Others'], title='', add_net_colors=True, add_net_labels='both', mode='count', cmap='Greys', clim_max=100, labels_text_color='Greys_r').opts(toolbar=None)

hvplot_fc_nwlevel(data['Image-Pos-Others_gt_Surr-Neg-Self'], title='', add_net_colors=True, add_net_labels='y', mode='count', cmap='Reds', clim_max=30, labels_text_color='Reds_r').opts(toolbar=None)

# # Laterality Index for each contrast
#

aux         = (data['Surr-Neg-Self_gt_Image-Pos-Others']).copy()
aux.index   = data['Surr-Neg-Self_gt_Image-Pos-Others'].index.get_level_values('Hemisphere')
aux.columns = data['Surr-Neg-Self_gt_Image-Pos-Others'].columns.get_level_values('Hemisphere')
f2GTf1_LL   = (aux.loc['LH','LH'].sum().sum() / 2)
f2GTf1_RR   = (aux.loc['RH','RH'].sum().sum() / 2)
f2GTf1_LR   = aux.loc['LH','RH'].sum().sum()
print('++ INFO [Surr-Neg-Self > Image-Pos-Others] L-L Conns: %d' % f2GTf1_LL)
print('++ INFO [Surr-Neg-Self > Image-Pos-Others] R-R Conns: %d' % f2GTf1_RR)
print('++ INFO [Surr-Neg-Self > Image-Pos-Others] R-L Conns: %d' % f2GTf1_LR)
print('++ --------------------------------------------------------')
f2GTf1_fcLI  = (f2GTf1_LL - f2GTf1_RR) / (f2GTf1_LL + f2GTf1_RR)
print('++ INFO [Surr-Neg-Self > Image-Pos-Others] fcLI:      %.2f' % f2GTf1_fcLI)
