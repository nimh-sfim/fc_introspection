#!/usr/bin/env python
# coding: utf-8

# # Description - Exploration of NBS Results
# 
# NBS runs on Matlab (as explained in the previous notebook).
# 
# Prior to running this notebook, you also need to run a matlab script `matlab/NBS2BrainViewer.m` that will take the outputs from NBS and convert them into a format that can be easily read here.
# 
# This notebook generates the Circos plots and the network-summary connection counts. In-brain connectomes were generated with CONN using the matlab script `matlab/CONN_NBS_IndividualContrast_onBrain.m`

# In[1]:


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import numpy as np
import os.path as osp
import hvplot.pandas
import panel as pn
import holoviews as hv
from utils.basics import FB_400ROI_ATLAS_NAME as ATLAS_NAME
from utils.basics import ATLASES_DIR, RESOURCES_CONN_DIR, RESOURCES_NBS_DIR
from utils.plotting import hvplot_fc, hvplot_fc_nwlevel, plot_as_graph
elt_labels = {'T3p1':'Edge-Level Threshold (p<0.001)', 'T2p58':'Edge-Level Threshold (p<0.005)'}


# In[2]:


NBS_CONTRASTS = ['SetA_gt_SetB','SetB_gt_SetA']


# # 2. Load information about the Atlas and ROI needed for plotting
# 
# Load the data structure with information about the ROIs in the atlas

# In[3]:


ATLASINFO_PATH = osp.join(ATLASES_DIR,ATLAS_NAME,'{ATLAS_NAME}.roi_info.csv'.format(ATLAS_NAME=ATLAS_NAME))
roi_info       = pd.read_csv(ATLASINFO_PATH)
Nrois          = roi_info.shape[0]
Nedges         = int(Nrois*(Nrois-1)/2)
print(Nrois,Nedges)


# In[4]:


roi_info[roi_info['ROI_Name']=='LH_SalVentAttn_Med_5']


# Count the number of networks and get their names

# In[5]:


networks = list(roi_info['Network'].unique())
print(networks, len(networks))


# Load significant connections

# In[6]:


data = {}
for contrast in NBS_CONTRASTS:
    for elt in ['T3p1','T2p58']:
        aux_path = osp.join(RESOURCES_NBS_DIR,ATLAS_NAME,f'NBS_Results',f'NBS_{elt}_s0.05_{contrast}.edge')
        if osp.exists(aux_path):
            aux_data = np.loadtxt(aux_path)
            data[elt, contrast]  = pd.DataFrame(aux_data,
                                            index   = roi_info.set_index(['Hemisphere','Network','ROI_Name','ROI_ID','RGB']).index, 
                                            columns = roi_info.set_index(['Hemisphere','Network','ROI_Name','ROI_ID','RGB']).index)
            print('++ INFO: Data available for %s' % contrast)
        else:
            data[elt,contrast] = pd.DataFrame(np.zeros((Nrois,Nrois)),
                                        index   = roi_info.set_index(['Hemisphere','Network','ROI_Name','ROI_ID','RGB']).index, 
                                        columns = roi_info.set_index(['Hemisphere','Network','ROI_Name','ROI_ID','RGB']).index)
            print('++ WARTNING: No results available for %s,%s' % (elt, contrast))


# Print number (and percentage) of significant edges per contrast

# In[10]:


for contrast in NBS_CONTRASTS:
    for elt in ['T3p1','T2p58']:
            N_sig_edges = data[elt, contrast].sum().sum()/2
            PC_sig_edges = 100 * N_sig_edges / Nedges
            print(f"Significant edges for {elt}, {contrast}: {N_sig_edges:.0f} ({PC_sig_edges:.2f}%)")


# Print number (and percentage) of significnat nodes per constrast

# In[11]:


for contrast in NBS_CONTRASTS:
    for elt in ['T3p1','T2p58']:
        N_sig_nodes  = (data[elt, contrast].sum() > 0).sum()
        PC_sig_nodes = 100 * N_sig_nodes / Nrois
        print(f"Significant nodes for {elt}, {contrast}: {N_sig_nodes} ({PC_sig_nodes:.2f}%)")


# In[12]:


for elt in ['T3p1','T2p58']:
    data[elt, 'Both'] = data[elt, 'SetA_gt_SetB'] - data[elt, 'SetB_gt_SetA']


# We will also write the results of NBS into text format that we can load into CONN to generate the brain views of the results

# In[13]:


RESOURCES_CONN_DIR


# In[14]:


for elt,contrast in data.keys():
    if data[(elt,contrast)] is not None:
        aux_path = osp.join(RESOURCES_CONN_DIR,f'NBS_{elt}_s0.05_{contrast}.txt')
        np.savetxt(aux_path,data[(elt,contrast)].values)
        print("++ INFO: Contrast data [%s,%s] saved to disk %s" %(elt,contrast,aux_path))


# # Plot results at the individual connection level

# In[15]:


layout = pn.Row()
for elt in ['T3p1','T2p58']:
    plot = hvplot_fc(data[elt,'Both'].loc[:,networks,:].T.loc[:,networks,:].T, 
                     by='Network', 
                     add_color_segments=True, 
                     add_labels=True, 
                     cmap=['#ED7D31','#ffffff', '#4472C4'], 
                     major_label_overrides={-0.5:'Set B > Set A',0:'',0.5:'Set A > Set B'}, colorbar_position='top').opts(toolbar=None, title=elt_labels[elt])
    layout.append(plot)
    
layout.save(osp.join('figures','Notebook_Image_NBS_FullMatrices.html'), progress=True)
print('Saved Notebook_Image_NBS_FullMatrices.html to figures/')


# ![Matrices](./figures/Notebook_Image_NBS_FullMatrices.png)

# Same information in the form of circos plots

# In[16]:


circos_plots = {}
for elt in ['T3p1','T2p58']:
    plot = plot_as_graph(data[elt,'SetA_gt_SetB'], edge_weight=.5, show_hemi_labels=False,pos_edges_color='k')
    circos_plots[elt] = plot
    if elt == 'T3p1':
        plot.savefig(osp.join('figures','Figure03_B.png'), bbox_inches='tight')
    else:
        plot.savefig(osp.join('figures','Supplementary_Figure07_A.png'), bbox_inches='tight')


# | Edge Threshold (p<0.001) | Edge Threshold (p<0.005) |
# |--------------------------|--------------------------|
# |![ELT1](./figures/Figure03_B.png) | ![ELT2](./figures/Supplementary_Figure07_A.png) 

# Top degree regions

# In[17]:


data['T3p1','SetA_gt_SetB'].sum(axis=1).sort_values(ascending=False), data['T2p58','SetA_gt_SetB'].sum(axis=1).sort_values(ascending=False)


# Network-level Connection Counts

# In[18]:


layout = pn.Row()
for elt in ['T3p1','T2p58']:
    if elt == 'T3p1':
        clim_max = 100
        file_name = 'Figure03_D.html'
    else:
        clim_max = 300
        file_name = 'Supplementary_Figure07_C.html'
    plot = hvplot_fc_nwlevel(data[elt,'SetA_gt_SetB'], title='', add_net_colors=True, add_net_labels='both', mode='count', cmap='Greys', clim_max=clim_max, labels_text_color='Greys_r').opts(toolbar=None)
    hv.save(plot,osp.join('figures',file_name))
    layout.append(plot)
    
layout.save(osp.join('figures','Notebook_Image_NBS_ConnectionCounts.html'), progress=True)


# In[20]:


layout


# | Edge Threshold (p<0.001) | Edge Threshold (p<0.005) |
# |--------------------------|--------------------------|
# | ![NC1](./figures/Figure03_D.png) | ![NC2](./figures/Supplementary_Figure07_C.png)

# # Laterality Index for each contrast
# 

# In[21]:


for elt in ['T3p1','T2p58']:
    aux         = (data[elt,'SetA_gt_SetB']).copy()
    aux.index   = data[elt,'SetA_gt_SetB'].index.get_level_values('Hemisphere')
    aux.columns = data[elt,'SetA_gt_SetB'].columns.get_level_values('Hemisphere')
    f2GTf1_LL   = (aux.loc['LH','LH'].sum().sum() / 2)
    f2GTf1_RR   = (aux.loc['RH','RH'].sum().sum() / 2)
    f2GTf1_LR   = aux.loc['LH','RH'].sum().sum()
    print('++ INFO [SetA > SetB] L-L Conns: %d' % f2GTf1_LL)
    print('++ INFO [SetA > SetB] R-R Conns: %d' % f2GTf1_RR)
    print('++ INFO [SetA > SetB] R-L Conns: %d' % f2GTf1_LR)
    print('++ --------------------------------------------------------')
    f2GTf1_fcLI  = (f2GTf1_LL - f2GTf1_RR) / (f2GTf1_LL + f2GTf1_RR)
    print('++ INFO [%s | SetA > SetB] fcLI:      %.2f' % (elt, f2GTf1_fcLI))

