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

# # Description - Exploration of NBS Results
#
# This notebook takes the outputs from running NBS and plots then for interpretation

import pandas as pd
import numpy as np
import os.path as osp
import hvplot.pandas
from utils.basics import FB_400ROI_ATLAS_NAME, ATLASES_DIR, RESOURCES_NIMARE_DIR, RESOURCES_CONN_DIR, FB_200ROI_ATLAS_NAME
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

SOLUTION   = 'CL02'
SCENARIO   = 'All_Scans'
THRESHOLD  = 'NBS_3p1'
ATLAS_NAME = FB_400ROI_ATLAS_NAME

# # 2. Load information about the Atlas and ROI needed for plotting
#
# Load the data structure with information about the ROIs in the atlas

ATLASINFO_PATH = osp.join(ATLASES_DIR,ATLAS_NAME,'{ATLAS_NAME}.roi_info.csv'.format(ATLAS_NAME=ATLAS_NAME))
roi_info       = pd.read_csv(ATLASINFO_PATH)

# Count the number of networks and get their names

networks = list(roi_info['Network'].unique())
print(networks, len(networks))

# Load the connections that are significantly stronger for the contrast: $$Image-Pos-Others > Surr-Neg-Self$$

#data_f1GTf2 = np.loadtxt(f'/data/SFIMJGC_Introspec/2023_fc_introspection/code/fc_introspection/resources/nbs/{ATLAS_NAME}/NBS_{SOLUTION}_Results/NBS_{SOLUTION}_Image-Pos-Others_gt_Surr-Neg-Self.edge')
data_f1GTf2 = np.loadtxt(f'/data/SFIMJGC_Introspec/2023_fc_introspection/code/fc_introspection/resources/nbs/{ATLAS_NAME}/{SCENARIO}/NBS_{SOLUTION}_Results/{THRESHOLD}/NBS_{SOLUTION}_Image-Pos-Others_gt_Surr-Neg-Self.edge')
data_f1GTf2 = pd.DataFrame(data_f1GTf2,
                           index   = roi_info.set_index(['Hemisphere','Network','ROI_Name','ROI_ID','RGB']).index, 
                           columns = roi_info.set_index(['Hemisphere','Network','ROI_Name','ROI_ID','RGB']).index)

# Load the connections that are significantly stronger for the contrast: $$Surr-Neg-Self > Image-Pos-Others$$

data_f2GTf1 = np.loadtxt(f'/data/SFIMJGC_Introspec/2023_fc_introspection/code/fc_introspection/resources/nbs/{ATLAS_NAME}/{SCENARIO}/NBS_{SOLUTION}_Results/{THRESHOLD}/NBS_{SOLUTION}_Surr-Neg-Self_gt_Image-Pos-Others.edge')
data_f2GTf1 = pd.DataFrame(data_f2GTf1,
                           index   = roi_info.set_index(['Hemisphere','Network','ROI_Name','ROI_ID','RGB']).index, 
                           columns = roi_info.set_index(['Hemisphere','Network','ROI_Name','ROI_ID','RGB']).index)

# We will also write the results of NBS into text format that we can load into CONN to generate the brain views of the results

np.savetxt(osp.join(RESOURCES_CONN_DIR,f'GroupDiffs_f1GTf2_{SCENARIO}_{THRESHOLD}_matrix.txt'),data_f1GTf2.values)
np.savetxt(osp.join(RESOURCES_CONN_DIR,f'GroupDiffs_f2GTf1_{SCENARIO}_{THRESHOLD}_matrix.txt'),data_f2GTf1.values)
np.savetxt(osp.join(RESOURCES_CONN_DIR,f'GroupDiffs_Both_{SCENARIO}_{THRESHOLD}_matrix.txt'),data_f1GTf2.values-data_f2GTf1.values)

# ## 2.1. Plot results as regular FC matrices
# Plot the NBS results in matrix form in three different ways: 
# 1) connections for both contrasts
# 2) Connections for ```Image-Pos-Others > Surr-Neg-Self``` only
# 3) Connections for ```Surr-Neg-Self > Image-Pos-Others``` only

data = data_f1GTf2 - data_f2GTf1

((data_f2GTf1).sum() > 0).sum()

((data_f1GTf2).sum() > 0).sum()

f_fullmodel_matrix = hvplot_fc(data.loc[:,networks,:].T.loc[:,networks,:].T, by='Network', add_color_segments=True, add_labels=True, cmap=['#4472C4','#ffffff','#ED7D31'], major_label_overrides={-0.5:'Surr-Neg-Self > Images-Pos-Others',0:'',0.5:'Images-Pos-Others > Surr-Neg-Self'}, colorbar_position='top').opts(toolbar=None)
f_fullmodel_matrix #pn.Row(f_fullmodel_matrix).save('./figures/S14_NBS_FullModel_MatrixView.png')

display.Image('./figures/S14_NBS_FullModel_MatrixView.png')

# We can also explore the two contrast separately in this form of representing the results

f = (hvplot_fc(data_f1GTf2.loc[:,networks,:].T.loc[:,networks,:].T, by='Network', add_color_segments=True, add_labels=True, cmap=['#4472C4','#ffffff','#ED7D31'], major_label_overrides={-0.5:'Surr-Neg-Self > Images-Pos-Others',0:'',0.5:'Images-Pos-Others > Surr-Neg-Self'}) + \
hvplot_fc(-data_f2GTf1.loc[:,networks,:].T.loc[:,networks,:].T, by='Network', add_color_segments=True, add_labels=True, cmap=['#4472C4','#ffffff','#ED7D31'], major_label_overrides={-0.5:'Surr-Neg-Self > Images-Pos-Others',0:'',0.5:'Images-Pos-Others > Surr-Neg-Self'})).opts(toolbar=None)
f #pn.Row(f).save('./figures/S14_NBS_TwoSeparateModels_MatrixView.png')

display.Image('./figures/S14_NBS_TwoSeparateModels_MatrixView.png')

# ## 2.2. Plot results as counts of inter- and between- network connections
#
# To get a better feeling of what networks have more/less significant connections, we will plot a summary view of the results with counts of significant connections per contrast at the network level (instead of ROI level)

f_F1gtF2 = hvplot_fc_nwlevel(data_f1GTf2, title='', add_net_colors=True, add_net_labels=True, mode='count', cmap='oranges', clim_max=50, labels_text_color='black').opts(toolbar=None)
f_F2gtF1 = hvplot_fc_nwlevel(data_f2GTf1, title='', add_net_colors=True, add_net_labels=True, mode='count', cmap='blues', clim_max=50, labels_text_color='gray').opts(toolbar=None)
#pn.Row(f_F1gtF2).save('./figures/S14_NBS_F1gtF2_nw_counts.png')
#pn.Row(f_F2gtF1).save('./figures/S14_NBS_F2gtF1_nw_counts.png')

hvplot_fc_nwlevel(data_f1GTf2, title='')

f_F1gtF2

f_F2gtF1

display.Image('./figures/S14_NBS_F1gtF2_nw_counts.png')

display.Image('./figures/S14_NBS_F2gtF1_nw_counts.png')

f_both = hvplot_fc_nwlevel(data_f1GTf2.abs()+data_f2GTf1.abs(), title='', add_net_colors=True, add_net_labels=True, mode='count', cmap='greys', clim_max=250, labels_text_color='black').opts(toolbar=None)
pn.Row(f_both).save('./figures/S14_NBS_BOTH_nw_counts.png')

display.Image('./figures/S14_NBS_BOTH_nw_counts.png')

data = (data_f1GTf2.abs()+data_f2GTf1.abs()).copy()
networks     = list(data.index.get_level_values('Network').unique())
num_networks = len(networks)
num_sig_cons = pd.DataFrame(index=networks, columns=networks)
pc_sig_cons  = pd.DataFrame(index=networks, columns=networks)
ncons_df     = pd.DataFrame(index=networks, columns=networks)
for n1 in networks:
    for n2 in networks:
        aux = data.loc[data.index.get_level_values('Network')==n1,data.columns.get_level_values('Network')==n2]
        if n1 == n2:
            # Within Network Scenario
            assert np.diagonal(aux).sum() == 0., "WARNING --> Diagonal is not zero for intra-network FC"
            ncons                   = aux.shape[0] * (aux.shape[0] -1 ) / 2
            ncons_df.loc[n1,n2]     = int(ncons)
            num_sig_cons.loc[n1,n2] = int(aux.sum().sum()/2) 
            assert aux.sum().sum()/2 == np.floor(aux.sum().sum()/2), "WARNING --> Matrix was not symmetric"
            pc_sig_cons.loc[n1,n2]  = 100 * num_sig_cons.loc[n1,n2] / ncons
        else:
            # Across Network Scenario
            #assert np.diagonal(aux).sum() == 0., "WARNING --> Diagonal is not zero for intra-network FC"
            ncons = aux.shape[0] * aux.shape[1]
            ncons_df.loc[n1,n2] = int(ncons)
            num_sig_cons.loc[n1,n2] = int(aux.sum().sum())
            pc_sig_cons.loc[n1,n2]  = 100 * num_sig_cons.loc[n1,n2] / ncons
num_sig_cons = num_sig_cons.infer_objects()
pc_sig_cons  = pc_sig_cons.infer_objects()

a = ncons_df.hvplot.heatmap(clim=(0,10000)).opts(toolbar=None)
a*hv.Labels(a)

a = pd.DataFrame(num_sig_cons.sum(axis=1), columns=['Total']).hvplot.heatmap(width=120, cmap='greys', clim=(0,1000)).opts(colorbar=False, toolbar=None)
b = a*hv.Labels(a)
pn.Row(b).save('./figures/S14_NBS_BOTH_nw_total_counts.png')

display.Image('./figures/S14_NBS_BOTH_nw_total_counts.png')

(100*num_sig_cons / ncons_df)

pc_sig_cons.sum(axis=1).sort_values()



a = pd.DataFrame(pc_sig_cons.sum(axis=1), columns=['%']).round(1).hvplot.heatmap(width=120, cmap='greys', clim=(0,30)).opts(colorbar=False, toolbar=None)
b = a*hv.Labels(a)
pn.Row(b).save('./figures/S14_NBS_BOTH_nw_total_pc.png')

display.Image('./figures/S14_NBS_BOTH_nw_total_pc.png')

# ## 2.3. Plot results as percentage of inter- and intra-network connections
#
# Similar view to the one above, but this time instead of reporting the number of significant connecitons, we report the percentage of those relative to the total number of inter- or intra-connections for a particular cell on the matrix

f_F1gtF2 = hvplot_fc_nwlevel(data_f1GTf2, title='', add_net_colors=True, add_net_labels=True, mode='percent', cmap='oranges', clim_max=5, labels_text_color='black').opts(toolbar=None)
f_F2gtF1 = hvplot_fc_nwlevel(data_f2GTf1, title='', add_net_colors=True, add_net_labels=True, mode='percent', cmap='blues', clim_max=5, labels_text_color='gray').opts(toolbar=None)
pn.Row(f_F1gtF2).save('./figures/S14_NBS_F1gtF2_nw_percent.png')
pn.Row(f_F2gtF1).save('./figures/S14_NBS_F2gtF1_nw_percent.png')

display.Image('./figures/S14_NBS_F1gtF2_nw_percent.png')

display.Image('./figures/S14_NBS_F2gtF1_nw_percent.png')

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

f = plot_as_graph(data,layout='circos',edge_weight=.2, show_degree=True)
f

f.savefig('./figures/S14_NBS_Circos_FullModel.png')

# ### 2.5.2. Image-Pos-Others > Surr-Neg-Self

f = plot_as_graph(data_f1GTf2,layout='circos',edge_weight=.2, show_degree=True)
f

f.savefig('./figures/S14_NBS_Circos_f1GTf2.png')

# ### 2.5.2. Surr-Neg-Self > Image-Pos-Others

f = plot_as_graph(-data_f2GTf1,layout='circos',edge_weight=.2, show_degree=True)
f

f.savefig('./figures/S14_NBS_Circos_f2GTf1.png')

# # 10. Laterality Index for each contrast
#
# First, we compute the fcLI for the ```Images-Pos-Others > Surr-Neg-Self``` contrast

aux = (data_f1GTf2).copy()
aux.index = data_f1GTf2.index.get_level_values('Hemisphere')
aux.columns = data_f1GTf2.columns.get_level_values('Hemisphere')
f1GTf2_LL = (aux.loc['LH','LH'].sum().sum() / 2)
f1GTf2_RR = (aux.loc['RH','RH'].sum().sum() / 2)
f1GTf2_LR = aux.loc['LH','RH'].sum().sum()
print('++ INFO [Image-Pos-Others > Surr-Neg-Self] L-L Conns: %d' % f1GTf2_LL)
print('++ INFO [Image-Pos-Others > Surr-Neg-Self] R-R Conns: %d' % f1GTf2_RR)
print('++ INFO [Image-Pos-Others > Surr-Neg-Self] R-L Conns: %d' % f1GTf2_LR)
print('++ --------------------------------------------------------')
f1GTf2_fcLI  = (f1GTf2_LL - f1GTf2_RR) / (f1GTf2_LL + f1GTf2_RR)
print('++ INFO [Image-Pos-Others > Surr-Neg-Self] fcLI:      %.2f' % f1GTf2_fcLI)

# Next, we do the same for the ```Surr-Neg-Self > Images-Pos-Others``` contrast

aux = (data_f2GTf1).copy()
aux.index = data_f2GTf1.index.get_level_values('Hemisphere')
aux.columns = data_f2GTf1.columns.get_level_values('Hemisphere')
f2GTf1_LL = (aux.loc['LH','LH'].sum().sum() / 2)
f2GTf1_RR = (aux.loc['RH','RH'].sum().sum() / 2)
f2GTf1_LR = aux.loc['LH','RH'].sum().sum()
print('++ INFO [Image-Pos-Others > Surr-Neg-Self] L-L Conns: %d' % f2GTf1_LL)
print('++ INFO [Image-Pos-Others > Surr-Neg-Self] R-R Conns: %d' % f2GTf1_RR)
print('++ INFO [Image-Pos-Others > Surr-Neg-Self] R-L Conns: %d' % f2GTf1_LR)
print('++ --------------------------------------------------------')
f2GTf1_fcLI  = (f2GTf1_LL - f2GTf1_RR) / (f2GTf1_LL + f2GTf1_RR)
print('++ INFO [Image-Pos-Others > Surr-Neg-Self] fcLI:      %.2f' % f2GTf1_fcLI)
