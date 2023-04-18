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
#     display_name: FC Instrospection (2023 | 3.10)
#     language: python
#     name: fc_introspection_2023_py310
# ---

# # Description - Exploration of NBS Results
#
# This notebook takes the outputs from running NBS and plots then for interpretation

import pandas as pd
import numpy as np
import os.path as osp
import hvplot.pandas
from utils.basics import FB_400ROI_ATLAS_NAME, ATLASES_DIR
from utils.plotting import hvplot_fc, hvplot_fc_nwlevel, create_graph_from_matrix, plot_as_graph
import holoviews as hv
from holoviews import opts
from IPython import display
import panel as pn
import matplotlib.pyplot as plt
from nilearn.plotting import plot_connectome
from matplotlib.colors import LinearSegmentedColormap
from sklearn.preprocessing import MinMaxScaler

SOLUTION   = 'CL02_0p001'
ATLAS_NAME = FB_400ROI_ATLAS_NAME

# # 2. Load information about the Atlas and ROI needed for plotting
#
# Load the data structure with information about the ROIs in the atlas

ATLASINFO_PATH = osp.join(ATLASES_DIR,ATLAS_NAME,'{ATLAS_NAME}.roi_info.csv'.format(ATLAS_NAME=ATLAS_NAME))
roi_info       = pd.read_csv(ATLASINFO_PATH)

# Count the number of networks and get their names

networks = list(roi_info['Network'].unique())
print(networks, len(networks))

# Load the connections that are significantly stronger for the contrast: $$Large F1 > Large F2$$

data_f1GTf2 = np.loadtxt(f'/data/SFIMJGC_Introspec/2023_fc_introspection/code/fc_introspection/resources/nbs/{ATLAS_NAME}/NBS_{SOLUTION}_Results/NBS_{SOLUTION}_F1gtF2.edge')
data_f1GTf2 = pd.DataFrame(data_f1GTf2,
                           index   = roi_info.set_index(['Hemisphere','Network','ROI_Name','ROI_ID','RGB']).index, 
                           columns = roi_info.set_index(['Hemisphere','Network','ROI_Name','ROI_ID','RGB']).index)

# Load the connections that are significantly stronger for the contrast: $$Large F2 > Large F1$$

data_f2GTf1 = np.loadtxt(f'/data/SFIMJGC_Introspec/2023_fc_introspection/code/fc_introspection/resources/nbs/{ATLAS_NAME}/NBS_{SOLUTION}_Results/NBS_{SOLUTION}_F2gtF1.edge')
data_f2GTf1 = pd.DataFrame(data_f2GTf1,
                           index   = roi_info.set_index(['Hemisphere','Network','ROI_Name','ROI_ID','RGB']).index, 
                           columns = roi_info.set_index(['Hemisphere','Network','ROI_Name','ROI_ID','RGB']).index)

# ## 2.1. Plot results as regular FC matrices
# Plot the NBS results in matrix form in three different ways: 
# 1) connections for both contrasts
# 2) Connections for $Large F1 > Large F2$ only
# 3) Connections for $Large F2 > Large F1$ only

data = data_f1GTf2 - data_f2GTf1

f = (hvplot_fc(data.loc[:,networks,:].T.loc[:,networks,:].T, by='Network', add_color_segments=True, add_labels=True, cmap=['#4472C4','#ffffff','#ED7D31'], major_label_overrides={-0.5:'Large F2 > Large F1',0:'',0.5:'Large F1 > Large F2'}) + \
hvplot_fc(data_f1GTf2.loc[:,networks,:].T.loc[:,networks,:].T, by='Network', add_color_segments=True, add_labels=True, cmap=['#4472C4','#ffffff','#ED7D31'], major_label_overrides={-0.5:'Large F2 > Large F1',0:'',0.5:'Large F1 > Large F2'}) + \
hvplot_fc(-data_f2GTf1.loc[:,networks,:].T.loc[:,networks,:].T, by='Network', add_color_segments=True, add_labels=True, cmap=['#4472C4','#ffffff','#ED7D31'], major_label_overrides={-0.5:'Large F2 > Large F1',0:'',0.5:'Large F1 > Large F2'})).opts(toolbar=None)
pn.Row(f).save('./figures/S14_NBS_asFCmatrices.png')

display.Image('./figures/S14_NBS_asFCmatrices.png')

# ## 2.2. Plot results as counts of inter- and between- network connections
#
# To get a better feeling of what networks have more/less significant connections, we will plot a summary view of the results with counts of significant connections per contrast at the network level (instead of ROI level)

f=(hvplot_fc_nwlevel(data_f1GTf2, title='Large F1 > Large F2', add_net_colors=True, add_net_labels=True, mode='count', cmap='viridis', clim_max=50, labels_text_color='steelblue') + \
hvplot_fc_nwlevel(data_f2GTf1, title='Large F2 > Large F1', add_net_colors=True, add_net_labels=True, mode='count', cmap='viridis', clim_max=50, labels_text_color='steelblue')).opts(toolbar=None)
pn.Row(f).save('./figures/S14_NBS_net_counts.png')

display.Image('./figures/S14_NBS_net_counts.png')

# ## 2.3. Plot results as percentage of inter- and intra-network connections
#
# Similar view to the one above, but this time instead of reporting the number of significant connecitons, we report the percentage of those relative to the total number of inter- or intra-connections for a particular cell on the matrix

f=(hvplot_fc_nwlevel(data_f1GTf2, title='Large F1 > Large F2', add_net_colors=True, add_net_labels=True, mode='percent', cmap='viridis', clim_max=5, labels_text_color='steelblue') + \
hvplot_fc_nwlevel(data_f2GTf1, title='Large F2 > Large F1', add_net_colors=True, add_net_labels=True, mode='percent', cmap='viridis', clim_max=5, labels_text_color='steelblue')).opts(toolbar=None)
pn.Row(f).save('./figures/S14_NBS_net_percent.png')

display.Image('./figures/S14_NBS_net_percent.png')

# ## 2.4. Plot results in glass brains
#
# We now plot the results of the NBS analysis as glass brains using nilearn's ```plot_connection``` function
#
# ### 2.4.1. All Significant connections (both contrast directions)

G, G_att      = create_graph_from_matrix(data)
N_rois        = G_att.shape[0]
G_att['Size'] = MinMaxScaler(feature_range=(0,200)).fit_transform(G_att['Degree'].values.reshape(-1,1))

fig, ax            = plt.subplots(1,1,figsize=(15,7))
fig_NBS_both_brain = plot_connectome(data,roi_info[['pos_R','pos_A','pos_S']],
                                   node_color=roi_info['RGB'], 
                                   node_size=G_att['Size'], 
                                   edge_kwargs={'linewidth':.5}, 
                                   edge_cmap=LinearSegmentedColormap.from_list('nbs_model',['#4472C4','#ED7D31'],N=2), 
                                   node_kwargs={'linewidth':1, 'edgecolor':'k'}, axes=ax)
fig_NBS_both_brain.savefig('./figures/S14_NBS_GlassBrain_FullModel.png')

f = pn.Row(pn.pane.DataFrame(G_att[['ROI_Name','Degree']].sort_values(by='Degree', ascending=False)[0:10],width=500),
       G_att['Degree'].hvplot.box(width=300, color='gray').opts(toolbar=None))
f.save('./figures/S14_NBS_HighDegree_FullModel.png')

display.Image('./figures/S14_NBS_HighDegree_FullModel.png')

# ### 2.4.2. Significant Connections only for the contrast Large F1 > Large F2

G, G_att      = create_graph_from_matrix(data_f1GTf2)
N_rois        = G_att.shape[0]
G_att['Size'] = MinMaxScaler(feature_range=(0,200)).fit_transform(G_att['Degree'].values.reshape(-1,1))

fig, ax            = plt.subplots(1,1,figsize=(15,7))
fig_NBS_f1GTf2_brain = plot_connectome(data_f1GTf2,roi_info[['pos_R','pos_A','pos_S']],
                                   node_color=roi_info['RGB'], 
                                   node_size=G_att['Size'], 
                                   edge_kwargs={'linewidth':.5, 'color':'#ED7D31'}, 
                                   node_kwargs={'linewidth':1, 'edgecolor':'k'}, axes=ax)
fig_NBS_f1GTf2_brain.savefig('./figures/S14_NBS_GlassBrain_f1GTf2.png')

f = pn.Row(pn.pane.DataFrame(G_att[['ROI_Name','Degree']].sort_values(by='Degree', ascending=False)[0:10],width=500),
       G_att['Degree'].hvplot.box(width=300, color='#ED7D31').opts(toolbar=None))
f.save('./figures/S14_NBS_HighDegree_f1GTf2.png')

display.Image('./figures/S14_NBS_HighDegree_f1GTf2.png')

# ### 2.4.2. Significant Connections only for the contrast Large F2 > Large F1

G, G_att      = create_graph_from_matrix(data_f2GTf1)
N_rois        = G_att.shape[0]
G_att['Size'] = MinMaxScaler(feature_range=(0,200)).fit_transform(G_att['Degree'].values.reshape(-1,1))

fig, ax            = plt.subplots(1,1,figsize=(15,7))
fig_NBS_f2GTf1_brain = plot_connectome(data_f2GTf1,roi_info[['pos_R','pos_A','pos_S']],
                                   node_color=roi_info['RGB'], 
                                   node_size=G_att['Size'], 
                                   edge_kwargs={'linewidth':.5, 'color':'#4472C4'}, 
                                   node_kwargs={'linewidth':1, 'edgecolor':'k'}, axes=ax)
fig_NBS_f2GTf1_brain.savefig('./figures/S14_NBS_GlassBrain_f2GTf1.png')

f = pn.Row(pn.pane.DataFrame(G_att[['ROI_Name','Degree']].sort_values(by='Degree', ascending=False)[0:10],width=500),
       G_att['Degree'].hvplot.box(width=300, color='#4472C4').opts(toolbar=None))
f.save('./figures/S14_NBS_HighDegree_f2GTf1.png')

display.Image('./figures/S14_NBS_HighDegree_f2GTf1.png')

# ## 2.5. Plot NBS results as Circos Plots
#
# ### 2.5.1. Full Model

plot_as_graph(data,layout='circos',edge_weight=.2, show_degree=True)

# ### 2.5.2. Large F1 > Large F2 

plot_as_graph(data_f1GTf2,layout='circos',edge_weight=.2, show_degree=True)

# ### 2.5.2. Large F2 > Large F1

plot_as_graph(-data_f2GTf1,layout='circos',edge_weight=.2, show_degree=True)
