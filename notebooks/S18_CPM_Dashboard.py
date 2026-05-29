#!/usr/bin/env python
# coding: utf-8

# # Description
# 
# Dashboard to access results for the CPM portion of the analyses.

# In[1]:


import pandas as pd
import os.path as osp
from utils.basics import RESOURCES_CPM_DIR, RESOURCES_CONN_DIR
import hvplot.pandas
from tqdm import tqdm
import numpy as np
import xarray as xr
import pickle
from utils.basics import FB_400ROI_ATLAS_NAME, ATLASES_DIR
from cpm.plotting import plot_predictions
import seaborn as sns
import panel as pn
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import squareform
from utils.plotting import  plot_as_graph, create_graph_from_matrix, hvplot_fc_nwlevel

from nilearn.plotting import plot_connectome
from nxviz.utils import node_table
from sklearn.preprocessing import MinMaxScaler
from IPython import display

import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=FutureWarning)


# In[2]:


print('++ Packages versions:')
print('hvplot version: %s' % str(hvplot.__version__))
print('xr version: %s' % str(xr.__version__))
print('pandas version: %s' % str(pd.__version__))


# In[3]:


from nxviz.utils import node_table


# In[4]:


import os
port_tunnel = 35707
#port_tunnel = int(os.environ['PORT2'])
print('++ INFO: Second Port available: %d' % port_tunnel)


# In[5]:


ACCURACY_METRIC      = 'pearson'
CORR_TYPE            = 'pearson'
E_SUMMARY_METRIC     = 'sum'
CONFOUNDS            = 'conf_residualized'
BEHAVIOR_LIST        = ['Factor1','Factor2','Vigilance','Images','Words','People','Myself','Positive','Negative','Surroundings','Intrusive','Future','Past','Specific']
BEHAVIOR_LIST_LABELS = {'Factor1':'Thought Pattern 1','Factor2':'Thought Pattern 2','Vigilance':'Wakefulness','Images':'Images',
                        'Words':'Words','People':'People','Myself':'Myself','Positive':'Positive','Negative':'Negative',
                       'Surroundings':'Surroundings','Intrusive':'Intrusive','Future':'Future','Past':'Past','Specific':'Specific'}
SPLIT_MODE           = 'subject_aware'
ATLAS                = FB_400ROI_ATLAS_NAME
CPM_NITERATIONS      = 100
CPM_NULL_NITERATIONS = 10000


# # 1. Load CPM Predictions
# 
# Load summary of CPM results as created in ```S17_CPM_View_Prediction_Results```
# 

# In[6]:


results_path = osp.join(RESOURCES_CPM_DIR,'cpm_predictions_summary-subject_aware-conf_residualized-pearson.pkl')
cpm_results_dict = pd.read_pickle(results_path)


# In[7]:


null_df       = cpm_results_dict['null_df']
real_df       = cpm_results_dict['real_df']
accuracy_null = cpm_results_dict['accuracy_null']
accuracy_real = cpm_results_dict['accuracy_real']
p_values      = cpm_results_dict['p_values']
null_predictions_xr = cpm_results_dict['null_predictions_xr']
real_predictions_xr = cpm_results_dict['real_predictions_xr']

real_df.head()


# ## 1.1. Create Dashboard Functions for showing predictions as boxenplots

# In[8]:


def get_boxen_plot(behavior):
    median_width = 0.4
    sns.set(style='whitegrid')
    fig,ax = plt.subplots(1,1,figsize=(1,5))
    sns.boxenplot(data=null_df[null_df['Question']==behavior],x='Question',y='R', color='lightgray', ax=ax) 
    sns.stripplot(data=real_df[real_df['Question']==behavior],x='Question', y='R', alpha=.5, ax=ax)
    plt.xticks(rotation=0);
    for tick, text in zip(ax.get_xticks(), ax.get_xticklabels()):
        # Add Black Line Signaling Median
        question   = text.get_text()
        median_val = accuracy_real[question].median().values[0]
        ax.plot([tick-median_width/2, tick+median_width/2],[median_val,median_val], lw=4, color='k')
        # Statistical Significant Information
        p = p_values.loc[question,'Non Parametric']
        if 5.00e-02 < p <= 1.00e+00:
            annot = '' 
        elif 1.00e-02 < p <= 5.00e-02:
            annot = '*'
        elif 1.00e-03 < p <= 1.00e-02:
            annot = '**'
        elif 1.00e-04 < p <= 1.00e-03:
            annot = '***'
        elif p <= 1.00e-04:
            annot = '****'
        max_val = real_df.set_index('Question').max()['R']
        ax.annotate(annot, xy=(tick, max_val+0.02), ha='center', fontsize=15)
    ax.set_ylim(-.3,.4)
    ax.set_ylabel('R (Observed,Predicted)');
    ax.set_xlabel('')
    #ax.yaxis.get_label().set_visible(True)
    plt.close()
    plt.tight_layout()
    return fig


# In[9]:


def get_obs_vs_pred(behavior):
    behav_obs_pred = pd.DataFrame(real_predictions_xr.median(dim='Iteration').loc[behavior,:,['observed','predicted (glm)']], 
                                  columns=['observed','predicted (glm)'])
    fig,ax = plt.subplots(1,1,figsize=(5,5))
    r,p = plot_predictions(behav_obs_pred, ax=ax,
                       xlabel='Observed [%s]' % behavior, 
                       ylabel='Predicted [%s]' % behavior, 
                       font_scale=1,p_value=p_values.loc[behavior,'Non Parametric'])
    plt.close()
    return fig


# # 2. Load CPM Network Models
# 
# First, we just load one model as a reference to infer the number of edges. We need this to create empty datastructures that will subsequently populate

# In[10]:


ref_path = osp.join(RESOURCES_CPM_DIR,'swarm_outputs','real',ATLAS,SPLIT_MODE, CONFOUNDS,CORR_TYPE+'_'+E_SUMMARY_METRIC,'Images','cpm_Images_rep-{r}.pkl'.format(r=str(1).zfill(5)))
print(ref_path)
ref_data = pd.read_pickle(ref_path)
n_edges = ref_data['models']['pos'].shape[1]


# ## 2.1. Load ROI Information

# Next, we load the dtaframe with information about the different ROIs: labels, network membership, centroid, color

# In[11]:


ATLASINFO_PATH = osp.join(ATLASES_DIR,ATLAS,f'{ATLAS}.roi_info.csv')
roi_info       = pd.read_csv(ATLASINFO_PATH)


# And get a list of available networks

# In[12]:


nw_list = list(roi_info['Network'].unique())
print(nw_list)


# ## 2.2. Load models for all prediction targets
# 
# > **NOTE:** Run only one of the two cells in this subsection. See below
# 
# If new results are available run the following cell, which takes time, but will load all results into memory. It will also save a pickle file with the new results. That way on successive runs of the notebook you won't have to wait for this cell to complete. Alternatively, you could run the cell below, which looks for the fila and loads it into memory

# In[13]:


get_ipython().run_cell_magic('time', '', "models = {}\nmodels_to_vis = {}\nfor BEHAVIOR in BEHAVIOR_LIST:\n    models = {(BEHAVIOR_LIST_LABELS[BEHAVIOR],'pos'):pd.DataFrame(index=range(CPM_NITERATIONS), columns=range(n_edges)),\n              (BEHAVIOR_LIST_LABELS[BEHAVIOR],'neg'):pd.DataFrame(index=range(CPM_NITERATIONS), columns=range(n_edges))}\n    df = pd.DataFrame(index=range(CPM_NITERATIONS),columns=['pos','neg','glm'])\n    for r in tqdm(range(CPM_NITERATIONS), desc='Iteration [%s]' % BEHAVIOR_LIST_LABELS[BEHAVIOR]):\n        path = osp.join(RESOURCES_CPM_DIR,'swarm_outputs','real',ATLAS,SPLIT_MODE, CONFOUNDS,CORR_TYPE+'_'+E_SUMMARY_METRIC,BEHAVIOR,'cpm_{b}_rep-{r}.pkl'.format(b=BEHAVIOR,r=str(r+1).zfill(5)))\n        data = pd.read_pickle(path)\n        # We first averaged the number of times an edge was selected within each 10-fold run (resulting in a number between 0 and 1 for each edge)\n        for tail in ['pos','neg']:\n            models[BEHAVIOR_LIST_LABELS[BEHAVIOR],tail].loc[r,:] = data['models'][tail].mean(axis=0)\n    # and then averaged those fractions across all 100 train-test split iterations\n    models_to_vis[BEHAVIOR_LIST_LABELS[BEHAVIOR],'pos'] = models[BEHAVIOR_LIST_LABELS[BEHAVIOR],'pos'].mean()   \n    models_to_vis[BEHAVIOR_LIST_LABELS[BEHAVIOR],'neg'] = models[BEHAVIOR_LIST_LABELS[BEHAVIOR],'neg'].mean()   \n")


# In[14]:


if not osp.exists('../resources/cpm/plot_tmp/'):
    os.makedirs('../resources/cpm/plot_tmp/')
data_to_disk = {'models':models, 'models_to_vis':models_to_vis}
out_path     = '../resources/cpm/plot_tmp/models.pkl'
with open(out_path,'wb') as f:
    pickle.dump(data_to_disk,f)
print('++ Models saved to disk at: %s' % out_path)


# ## 2.3. Compute consensus models for plotting

# In[14]:


thresh           = 0.9
model_consensus,num_edges_toshow,model_consensus_to_plot  = {},{},{}
for BEHAVIOR in BEHAVIOR_LIST_LABELS.values():
    for tail in ['pos','neg']:
        edge_frac                       = models_to_vis[BEHAVIOR,tail]
        model_consensus[BEHAVIOR,tail]  = (edge_frac>=thresh).astype(int)
        num_edges_toshow[BEHAVIOR,tail] = model_consensus[BEHAVIOR,tail].sum()
        print("For the [{behav},{tail}], {edges} edges were selected in at least {pct}% of folds".format(behav=BEHAVIOR,tail=tail, edges=num_edges_toshow[BEHAVIOR,tail], pct=thresh*100))
    model_consensus_to_plot[BEHAVIOR] = pd.DataFrame(squareform(model_consensus[BEHAVIOR,'pos'])-squareform(model_consensus[BEHAVIOR,'neg']),
                          index = roi_info.set_index(['ROI_ID','ROI_Name','Hemisphere','Network','RGB']).index,
                          columns= roi_info.set_index(['ROI_ID','ROI_Name','Hemisphere','Network','RGB']).index)


# In[15]:


num_edges_toshow_DF = pd.Series(num_edges_toshow).reset_index()
num_edges_toshow_DF.columns = ['Target','Network','# Edges']
num_edges_toshow_DF.set_index(['Target','Network'],inplace=True)
num_edges_toshow_DF.groupby('Target').sum()


# # Saving Results for CONN visualizations
# We also write the models to disk in a form that we can later load in CONN

# In[16]:


for BEHAVIOR in BEHAVIOR_LIST_LABELS.values():
    aux_fc = model_consensus_to_plot[BEHAVIOR]
    aux_fc_path = osp.join(RESOURCES_CONN_DIR,f'CPM_{BEHAVIOR}_matrix.txt')
    np.savetxt(aux_fc_path,aux_fc.values)
    print("++ INFO [CONN OUTPUTS] Saving matrix model to %s" % aux_fc_path)


# Create extra files that are ATLAS specific so that we can plot results in CONN

# In[17]:


roi_info['ROI_Name'].to_csv(osp.join(RESOURCES_CONN_DIR,'roi_labels.txt'),header=None, index=None)


# In[18]:


roi_info[['pos_R','pos_A','pos_S']].to_csv(osp.join(RESOURCES_CONN_DIR,'roi_coords.txt'),header=None, index=None)


# In[19]:


(roi_info[['color_R','color_G','color_B']]/256).round(2).to_csv(osp.join(RESOURCES_CONN_DIR,'roi_colors.txt'),header=None, index=None)


# ***
# # 3. Create Dashboard
# 
# 1. Estimate the limits for the colorbar in the NW summary view (connection count mode)

# In[20]:


max_counts = []
for BEHAVIOR in BEHAVIOR_LIST_LABELS.values():
    a = model_consensus_to_plot[BEHAVIOR].abs().groupby('Network').sum().T.groupby('Network').sum()
    for n in a.index:
        a.loc[n,n] = int(a.loc[n,n]/2)
    max_counts.append(a.max().max())
max_counts = np.array(max_counts)
nw_count_max = int(np.quantile(max_counts,.9))


# 2. Create a drop box with all Questions

# In[21]:


behav_select     = pn.widgets.Select(name='Questions',options=list(BEHAVIOR_LIST_LABELS.values()),value='Images')
cmap_pos_select  = pn.widgets.Select(name='Colormap for Positive Matrix', options=['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
                      'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
                      'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn','viridis', 'plasma', 'inferno', 'magma', 'cividis',
                      'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone',
                      'pink', 'spring', 'summer', 'autumn', 'winter', 'cool',
                      'Wistia', 'hot', 'afmhot', 'gist_heat', 'copper'], value='Reds') 
cmap_neg_select  = pn.widgets.Select(name='Colormap for Negative Matrix', options=['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
                      'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
                      'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn','viridis', 'plasma', 'inferno', 'magma', 'cividis',
                      'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone',
                      'pink', 'spring', 'summer', 'autumn', 'winter', 'cool',
                      'Wistia', 'hot', 'afmhot', 'gist_heat', 'copper'], value='Blues') 
matrix_max_count = pn.widgets.IntSlider(name='Max Num Conns:',start=10, end=300, step=5, value=100)
menu_tab         = pn.Column(behav_select,cmap_pos_select,cmap_neg_select, matrix_max_count)


# 3. Create all elements of the dashboard

# In[22]:


circos_show_pos_cb   = pn.widgets.Checkbox(name='Show postively correlated edges', value=True)
circos_show_neg_cb   = pn.widgets.Checkbox(name='Show negatively correlated edges', value=True)
circos_show_degree   = pn.widgets.Checkbox(name='Node Size as a function of degree', value=True)
circos_layout        = pn.widgets.Select(name='Layout', options=['circos','spring','spectral','kamada_kawai'], value='circos')
@pn.depends(behav_select,circos_show_pos_cb,circos_show_neg_cb,circos_layout,circos_show_degree)
def gather_circos_plot(behavior, show_pos, show_neg, layout,show_degree, show_hemi_labels=True):
    return plot_as_graph(model_consensus_to_plot[behavior],figsize=(12,12),edge_weight=.5, title=behavior, show_pos=show_pos, show_neg=show_neg, 
                         pos_edges_color='#640900', neg_edges_color='#090064', layout=layout, show_degree=show_degree, show_hemi_labels=show_hemi_labels)
circos_tab = pn.Column(circos_show_pos_cb,circos_show_neg_cb,gather_circos_plot, circos_show_degree,circos_layout)


# In[23]:


@pn.depends(behav_select)
def gather_interactive_brain_view(behavior):
    G = nx.from_pandas_adjacency(model_consensus_to_plot[behavior].abs())
    d = [val for node,val in G.degree()]
    fig, ax = plt.subplots(1,1,figsize=(20,10))
    plot = plot_connectome(model_consensus_to_plot[behavior],roi_info[['pos_R','pos_A','pos_S']], node_color=roi_info['RGB'], node_size=d, axes=ax) #, linewidth=1, colorbar_fontsize=10, node_size=d)
    return plot


# In[24]:


@pn.depends(behav_select, cmap_pos_select, cmap_neg_select, matrix_max_count)
def gather_nw_matrix(behavior, pos_cmap, neg_cmap, clim_max_count, clim_min_count=0, add_net_labels=True):
    pos_count = hvplot_fc_nwlevel(model_consensus_to_plot[behavior]>0,title='Positive Correlation',mode='count', add_net_colors=True, add_net_labels=add_net_labels, cmap=pos_cmap, labels_text_color='red', clim_min=clim_min_count, clim_max=clim_max_count).opts(toolbar=None) #cmap='Reds'
    neg_count = hvplot_fc_nwlevel(model_consensus_to_plot[behavior]<0,title='Negative Correlation',mode='count', add_net_colors=True, add_net_labels=add_net_labels, cmap=neg_cmap, labels_text_color='blue', clim_min=clim_min_count, clim_max=clim_max_count).opts(toolbar=None)
    all_count = hvplot_fc_nwlevel(model_consensus_to_plot[behavior].abs(),title='Full Model',mode='count', add_net_colors=True).opts(toolbar=None)
    count_card = pn.Card(pn.Row(pos_count,neg_count,all_count), title='Number of Edges', width=2200)
    
    pos_pcent = hvplot_fc_nwlevel(model_consensus_to_plot[behavior]>0,title='Positive Correlation',add_net_colors=True, cmap=pos_cmap, add_net_labels=add_net_labels, clim_max=15, labels_text_color='red').opts(toolbar=None)
    neg_pcent = hvplot_fc_nwlevel(model_consensus_to_plot[behavior]<0,title='Negative Correlation',add_net_colors=True, cmap=neg_cmap, add_net_labels=add_net_labels, clim_max=15, labels_text_color='blue').opts(toolbar=None)
    all_pcent = hvplot_fc_nwlevel(model_consensus_to_plot[behavior].abs(),title='Full Model',add_net_colors=True, clim_max=15).opts(toolbar=None)
    pcent_card = pn.Card(pn.Row(pos_pcent,neg_pcent,all_pcent), title='Percentage of Edges', width=2200)
    return pn.Column(count_card, pcent_card)


# In[25]:


@pn.depends(behav_select)
def get_pred_plots(behavior):
    return pn.Card(pn.Row(pn.pane.Matplotlib(get_boxen_plot(behavior), width=150, height=420,tight=True), 
        pn.pane.Matplotlib(get_obs_vs_pred(behavior), width=420, height=420,tight=True)),
        title='Prediction Power')


# In[26]:


nws_group_from = pn.widgets.CheckBoxGroup(name='Networks', value=nw_list, options=nw_list, inline=True)
nws_group_to   = pn.widgets.CheckBoxGroup(name='Networks', value=nw_list, options=nw_list, inline=True)
only_sel_nw    = pn.widgets.Checkbox(name='Show nodes for selected networks only', value=False)


# In[27]:


@pn.depends(behav_select,nws_group_from,nws_group_to)
def plot_brain_model(behavior,sel_nws_from,sel_nws_to):
    fig, ax = plt.subplots(1,1,figsize=(20,10))
    ax.grid(False)
    ax.axis(False)
    sel_nws_union = list(set(sel_nws_from+sel_nws_to)) 
    sel_rois_info = roi_info.copy()
    full_model         = model_consensus_to_plot[behavior].copy()
    plot_model        = pd.DataFrame(0, index=full_model.index.get_level_values('ROI_ID'), columns=full_model.columns.get_level_values('ROI_ID'))
    for nwf in sel_nws_from:
        for nwt in sel_nws_to:
            index_ = full_model.loc[:,:,:,nwf,:].index
            col_   = full_model.T.loc[:,:,:,nwt,:].index
            plot_model.loc[index_.get_level_values('ROI_ID'),col_.get_level_values('ROI_ID')] = full_model.loc[:,:,:,nwf,:].T.loc[:,:,:,nwt,:].T.values
            plot_model.loc[col_.get_level_values('ROI_ID'),index_.get_level_values('ROI_ID')] = full_model.loc[:,:,:,nwt,:].T.loc[:,:,:,nwf,:].T.values
    plot_model.index = full_model.index
    plot_model.columns = full_model.columns
        
    # ==============
    _,Gnt = create_graph_from_matrix(plot_model)
    # ==============
    _ = plot_connectome(adjacency_matrix=plot_model, 
                                     node_coords=sel_rois_info[['pos_R', 'pos_A','pos_S']],
                                     node_color=sel_rois_info['RGB'],node_size=5*Gnt['Degree'],
                                     edge_kwargs={'linewidth':0.5},
                                     node_kwargs={'edgecolor':'k', 'linewidth':0.5},
                                     figure=fig)
    plt.close()
    return pn.pane.Matplotlib(fig)


# In[28]:


@pn.depends(behav_select)
def get_conn_counts(behavior):
    posconns = (model_consensus_to_plot[behavior]>0).groupby('Network').sum().T.groupby('Network').sum().loc[['Vis','SomMot','DorsAttn','SalVentAttn','Limbic','Cont','Default','Subcortical'],['Vis','SomMot','DorsAttn','SalVentAttn','Limbic','Cont','Default','Subcortical']]
    for nw in nw_list:
        posconns.loc[nw,nw] = posconns.loc[nw,nw] / 2
    posconns_final = posconns.sum()
    posconns_final.name = '# Conns'
    posconns_final['Total'] = int((model_consensus_to_plot[behavior]>0).sum().sum()/2)
    negconns = (model_consensus_to_plot[behavior]<0).groupby('Network').sum().T.groupby('Network').sum().loc[['Vis','SomMot','DorsAttn','SalVentAttn','Limbic','Cont','Default','Subcortical'],['Vis','SomMot','DorsAttn','SalVentAttn','Limbic','Cont','Default','Subcortical']]
    for nw in nw_list:
        negconns.loc[nw,nw] = negconns.loc[nw,nw] / 2
    negconns_final = negconns.sum()
    negconns_final.name = '# Conns'
    negconns_final['Total'] = int((model_consensus_to_plot[behavior]<0).sum().sum()/2)
    return pn.Card(pn.Row(pn.Column(pn.pane.Markdown('### Positive Connections'),pn.pane.DataFrame(posconns_final)),
       pn.Column(pn.pane.Markdown('### Negative Connections'),pn.pane.DataFrame(negconns_final))),title='CPM Model | Connection Counts')


# In[29]:


@pn.depends(behav_select)
def get_top10degree_counts(behavior):
    aux_pos      = (model_consensus_to_plot[behavior]>0).sum().sort_values(ascending=False)[0:10]
    aux_pos.name = 'Degree'
    aux_pos      = pd.DataFrame(aux_pos).reset_index()
    nodes_pos    = aux_pos['ROI_ID'].values.astype(int)
    aux_pos      = aux_pos.drop(['RGB','ROI_ID'],axis=1)
    aux_pos.index = aux_pos.index + 1
    aux_pos.index.name = 'Ranking'

    aux_pos['ROI_Name'] = ['-'.join(i.split('_')[2:]) for i in aux_pos['ROI_Name']]
    
    aux_neg      = (model_consensus_to_plot[behavior]<0).sum().sort_values(ascending=False)[0:10]
    aux_neg.name = 'Degree'
    aux_neg      = pd.DataFrame(aux_neg).reset_index()
    nodes_neg    = aux_neg['ROI_ID'].values.astype(int)
    aux_neg      = aux_neg.drop(['RGB','ROI_ID'],axis=1)
    aux_neg.index = aux_neg.index + 1
    aux_neg.index.name = 'Ranking'
    
    aux_neg['ROI_Name'] = ['-'.join(i.split('_')[2:]) for i in aux_neg['ROI_Name']]
    
    common_nodes = list(np.intersect1d(nodes_pos,nodes_neg))
    print(len(common_nodes))
    if len(common_nodes) > 0:
        output = pn.Card(pn.Row(pn.Column(pn.pane.Markdown('### Positive Connections'),pn.pane.DataFrame(aux_pos, width=550)),
                                pn.Column(pn.pane.Markdown('### Negative Connections'),pn.pane.DataFrame(aux_neg, width=550)), pn.pane.Markdown('> **NOTE: Overlaping nodes %s' % common_nodes)),title='CPM Model | Top 10 Degree Nodes')
    else:
        output = pn.Card(pn.Row(pn.Column(pn.pane.Markdown('### Positive Connections'),pn.pane.DataFrame(aux_pos, width=550)),
       pn.Column(pn.pane.Markdown('### Negative Connections'),pn.pane.DataFrame(aux_neg, width=550))),title='CPM Model | Top 10 Degree Nodes')
    return output


# In[30]:


brain_view_tab=pn.Column(pn.Row('From:',nws_group_from),pn.Row('To. :',nws_group_to),only_sel_nw,plot_brain_model)


# 4. Create the dashboard

# In[31]:


template = pn.template.BootstrapTemplate(title='Project Dashboard',
                                          sidebar=[menu_tab],
                                          main=pn.Tabs(('Connection Counts',get_conn_counts),
                                                       ('Network-Level Matrix',gather_nw_matrix),
                                                       ('Prediction Plots',get_pred_plots),
                                                       ('Top 10 Degree ROIs',get_top10degree_counts), 
                                                       ('Circos Plot',circos_tab),
                                                       ('Brain View',brain_view_tab)))


# In[ ]:


dashboard = template.show()


# Here is a few screenshots of how the dashboard looks:
# 
# ![Img1](./figures/Notebook_Image_Dashboard_NwLevelMatrices.png)
# 
# ![Img2](./figures/Notebook_Image_Dashboard_Circos.png)
# 
# ![Img3](./figures/Notebook_Image_Dashboard_Prediction.png)

# Also here are the connection counts reported int he manuscript

# In[33]:


get_conn_counts('Wakefulness')


# In[34]:


get_conn_counts('Thought Pattern 1')


# In[35]:


get_conn_counts('Thought Pattern 2')


# 

# ### Graphical Elements shown in figures are saved here
# 
# 1. Circos plots and the models they represent

# In[ ]:


behaviors_to_save = ['Wakefulness','Thought Pattern 1','Thought Pattern 2']
figure_names = ['Figure06A','Figure07A','Figure07F']
source_data_filenames = ['figure_06_abd','figure_07_ace','figure_07_fhj']
for behavior,figure_name, source_data_filename in zip(behaviors_to_save,figure_names,source_data_filenames):
    print(f'++ Working on [{behavior}]')
    plot = plot_as_graph(model_consensus_to_plot[behavior],figsize=(12,12),edge_weight=.25, title=behavior, show_pos=True, show_neg=True, 
                         pos_edges_color='#640900', neg_edges_color='#090064', layout='circos', show_degree=True, show_hemi_labels=False)
    plot.savefig(osp.join('figures',f'{figure_name}.png'), bbox_inches='tight', dpi=300)
    plot.savefig(osp.join('figures',f'{figure_name}.svg'), format='svg', bbox_inches='tight')
    model_consensus_to_plot[behavior].to_csv(f'./source_data_files/{source_data_filename}.csv')


# In[38]:


behaviors_to_save = ['Images','Surroundings','Past']
source_data_filenames = ['figure_08_ab','figure_08_cd','figure_08_ef']
for behavior,figure_name, source_data_filename in zip(behaviors_to_save,figure_names,source_data_filenames):
    print(f'++ Working on [{behavior}]')
    model_consensus_to_plot[behavior].to_csv(f'./source_data_files/{source_data_filename}.csv')


# 2. Nw-Level summaries and the data they show

# In[ ]:


import holoviews as hv

from bokeh.io import save
from bokeh.models.plots import Plot
from bokeh.resources import INLINE

hv.extension("bokeh")

def svg_backend(hv_plot, element):
    hv_plot.state.output_backend = "svg"

nwlevel_plots_to_save, nwlevel_data_to_save = {},{}
source_data_filenames = {('Wakefulness','pos'):'./source_data_files/figure_06_c.csv',('Wakefulness','neg'):'./source_data_files/figure_06_e.csv',
                         ('Thought Pattern 1','pos'):'./source_data_files/figure_07_b.csv', ('Thought Pattern 1','neg'):'./source_data_files/figure_07_d.csv',
                         ('Thought Pattern 2','pos'):'./source_data_files/figure_07_g.csv', ('Thought Pattern 2','neg'):'./source_data_files/figure_07_i.csv'}

nwlevel_plot_filenames = {('Wakefulness','pos'):'Figure06C',('Wakefulness','neg'):'Figure06E',
                         ('Thought Pattern 1','pos'):'Figure07B', ('Thought Pattern 1','neg'):'Figure07D',
                         ('Thought Pattern 2','pos'):'Figure07G', ('Thought Pattern 2','neg'):'Figure07I'}

for behavior in behaviors_to_save:
    # Matrix plots
    nwlevel_plots_to_save[(behavior,'pos')] = hvplot_fc_nwlevel(model_consensus_to_plot[behavior]>0,title='',mode='count', add_net_colors=True, add_net_labels='both', cmap='Reds', labels_text_color='red', clim_min=0, clim_max=100).opts(toolbar=None)
    nwlevel_plots_to_save[(behavior,'neg')] = hvplot_fc_nwlevel(model_consensus_to_plot[behavior]<0,title='',mode='count', add_net_colors=True, add_net_labels='both', cmap='Blues', labels_text_color='blue', clim_min=0, clim_max=100).opts(toolbar=None)
    for model_sign in ['pos','neg']:
        svg_plot = nwlevel_plots_to_save[(behavior,model_sign)].opts(hooks=[svg_backend])
        bokeh_obj = hv.render(svg_plot, backend="bokeh")
        # Extra safety for layouts / overlays
        if isinstance(bokeh_obj, Plot):
            bokeh_obj.output_backend = "svg"
        for p in bokeh_obj.select({"type": Plot}):
            p.output_backend = "svg"
        print(f'++ Saving [{behavior},{model_sign}] Nw-Level matrix to ./figures/{nwlevel_plot_filenames[(behavior,model_sign)]}.html')
        save(bokeh_obj, filename=f"./figures/{nwlevel_plot_filenames[(behavior,model_sign)]}.html",resources=INLINE, title=nwlevel_plot_filenames[(behavior,model_sign)])
    # Source Data Files
    nwlevel_data_to_save[(behavior,'pos')] = hvplot_fc_nwlevel(model_consensus_to_plot[behavior]>0,title='',mode='count', add_net_colors=True, add_net_labels='both', cmap='Reds', labels_text_color='red', clim_min=0, clim_max=100, return_data_only=True)
    nwlevel_data_to_save[(behavior,'neg')] = hvplot_fc_nwlevel(model_consensus_to_plot[behavior]<0,title='',mode='count', add_net_colors=True, add_net_labels='both', cmap='Reds', labels_text_color='red', clim_min=0, clim_max=100, return_data_only=True)
    nwlevel_data_to_save[(behavior,'pos')].to_csv(source_data_filenames[(behavior,'pos')])
    nwlevel_data_to_save[(behavior,'neg')].to_csv(source_data_filenames[(behavior,'neg')])


# In[ ]:




