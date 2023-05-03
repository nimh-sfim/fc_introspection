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
# Dashboard to access results for the CPM portion of the analyses.

import pandas as pd
import os.path as osp
from utils.basics import RESOURCES_CPM_DIR
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
from utils.plotting import hvplot_fc_nwlevel, plot_as_graph, create_graph_from_matrix
from nilearn.plotting import plot_connectome
from nxviz.utils import node_table
from sklearn.preprocessing import MinMaxScaler
from IPython import display

import os
port_tunnel = int(os.environ['PORT2'])
print('++ INFO: Second Port available: %d' % port_tunnel)

ACCURACY_METRIC      = 'pearson'
CORR_TYPE            = 'pearson'
E_SUMMARY_METRIC     = 'sum'
CONFOUNDS            = 'conf_residualized'
BEHAVIOR_LIST        = ['Factor1','Factor2','Vigilance','Images','Words','People','Myself','Positive','Negative','Surroundings','Intrusive','Future','Past','Specific']
SPLIT_MODE           = 'subject_aware'
ATLAS                = FB_400ROI_ATLAS_NAME
CPM_NITERATIONS      = 100
CPM_NULL_NITERATIONS = 10000

# # 1. Load CPM Predictions
#
# Load summary of CPM results as created in ```S16_CPM_View_Prediction_Results```
#

# +
results_path = osp.join(RESOURCES_CPM_DIR,'cpm_predictions_summary.pkl')
with open(results_path ,'rb') as f:
    cpm_results_dict = pickle.load(f)
# Extract the different variables
null_df       = cpm_results_dict['null_df']
real_df       = cpm_results_dict['real_df']
accuracy_null = cpm_results_dict['accuracy_null']
accuracy_real = cpm_results_dict['accuracy_real']
p_values      = cpm_results_dict['p_values']
null_predictions_xr = cpm_results_dict['null_predictions_xr']
real_predictions_xr = cpm_results_dict['real_predictions_xr']

real_df.head()


# -

# ## 1.1. Create Dashboard Functions for showing predictions as boxenplots

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

ref_path = osp.join(RESOURCES_CPM_DIR,'swarm_outputs','real',ATLAS,SPLIT_MODE, CONFOUNDS,CORR_TYPE+'_'+E_SUMMARY_METRIC,'Images','cpm_Images_rep-{r}.pkl'.format(r=str(1).zfill(5)))
with open(ref_path,'rb') as f:
    ref_data = pickle.load(f)
n_edges = ref_data['models']['pos'].shape[1]

# ## 2.1. Load ROI Information

# Next, we load the dtaframe with information about the different ROIs: labels, network membership, centroid, color

ATLASINFO_PATH = osp.join(ATLASES_DIR,ATLAS,f'{ATLAS}.roi_info.csv')
roi_info       = pd.read_csv(ATLASINFO_PATH)

# And get a list of available networks

nw_list = list(roi_info['Network'].unique())
print(nw_list)

# ## 2.2. Load models for all prediction targets
#
# > **NOTE:** Run only one of the two cells in this subsection. See below
#
# If new results are available run the following cell, which takes time, but will load all results into memory. It will also save a pickle file with the new results. That way on successive runs of the notebook you won't have to wait for this cell to complete. Alternatively, you could run the cell below, which looks for the fila and loads it into memory

# +
# %%time
models = {}
models_to_vis = {}
for BEHAVIOR in BEHAVIOR_LIST:
    models = {(BEHAVIOR,'pos'):pd.DataFrame(index=range(CPM_NITERATIONS), columns=range(n_edges)),
              (BEHAVIOR,'neg'):pd.DataFrame(index=range(CPM_NITERATIONS), columns=range(n_edges))}
    df = pd.DataFrame(index=range(CPM_NITERATIONS),columns=['pos','neg','glm'])
    for r in tqdm(range(CPM_NITERATIONS), desc='Iteration [%s]' % BEHAVIOR):
        path = osp.join(RESOURCES_CPM_DIR,'swarm_outputs','real',ATLAS,SPLIT_MODE, CONFOUNDS,CORR_TYPE+'_'+E_SUMMARY_METRIC,BEHAVIOR,'cpm_{b}_rep-{r}.pkl'.format(b=BEHAVIOR,r=str(r+1).zfill(5)))
        with open(path,'rb') as f:
            data = pickle.load(f)
        # We first averaged the number of times an edge was selected within each 10-fold run (resulting in a number between 0 and 1 for each edge)
        for tail in ['pos','neg']:
            models[BEHAVIOR,tail].loc[r,:] = data['models'][tail].mean(axis=0)
    # and then averaged those fractions across all 100 train-test split iterations
    models_to_vis[BEHAVIOR,'pos'] = models[BEHAVIOR,'pos'].mean()   
    models_to_vis[BEHAVIOR,'neg'] = models[BEHAVIOR,'neg'].mean()   

data_to_disk = {'models':models, 'models_to_vis':models_to_vis}
out_path     = '../resources/cpm/plot_tmp/models.pkl'
with open(out_path,'wb') as f:
    pickle.dump(data_to_disk,f)
# -

# Alternative cell that loads previous results. Much faster, but will not take into account potential new results.

out_path     = '../resources/cpm/plot_tmp/models.pkl'
with open(out_path,'rb') as f:
    data_to_disk = pickle.load(f)
models = data_to_disk['models']
models_to_vis = data_to_disk['models_to_vis']
del data_to_disk

# ## 2.3. Compute consensus models for plotting

thresh           = 0.9
model_consensus,num_edges_toshow,model_consensus_to_plot  = {},{},{}
for BEHAVIOR in BEHAVIOR_LIST:
    for tail in ['pos','neg']:
        edge_frac                       = models_to_vis[BEHAVIOR,tail]
        model_consensus[BEHAVIOR,tail]  = (edge_frac>=thresh).astype(int)
        num_edges_toshow[BEHAVIOR,tail] = model_consensus[BEHAVIOR,tail].sum()
        print("For the [{behav},{tail}], {edges} edges were selected in at least {pct}% of folds".format(behav=BEHAVIOR,tail=tail, edges=num_edges_toshow[BEHAVIOR,tail], pct=thresh*100))
    model_consensus_to_plot[BEHAVIOR] = pd.DataFrame(squareform(model_consensus[BEHAVIOR,'pos'])-squareform(model_consensus[BEHAVIOR,'neg']),
                          index = roi_info.set_index(['ROI_ID','ROI_Name','Hemisphere','Network','RGB']).index,
                          columns= roi_info.set_index(['ROI_ID','ROI_Name','Hemisphere','Network','RGB']).index)

# ***
# # 3. Create Dashboard
#
# 1. Estimate the limits for the colorbar in the NW summary view (connection count mode)

max_counts = []
for BEHAVIOR in BEHAVIOR_LIST:
    a = model_consensus_to_plot[BEHAVIOR].abs().groupby('Network').sum().T.groupby('Network').sum()
    for n in a.index:
        a.loc[n,n] = int(a.loc[n,n]/2)
    max_counts.append(a.max().max())
max_counts = np.array(max_counts)
nw_count_max = int(np.quantile(max_counts,.9))

# 2. Create a drop box with all Questions

behav_select = pn.widgets.Select(name='Questions',options=BEHAVIOR_LIST,value='Images')

# 3. Create all elements of the dashboard

circos_show_pos_cb   = pn.widgets.Checkbox(name='Show postively correlated edges', value=True)
circos_show_neg_cb   = pn.widgets.Checkbox(name='Show negatively correlated edges', value=True)
circos_show_degree   = pn.widgets.Checkbox(name='Node Size as a function of degree', value=True)
circos_layout        = pn.widgets.Select(name='Layout', options=['circos','spring','spectral','kamada_kawai'], value='circos')
@pn.depends(behav_select,circos_show_pos_cb,circos_show_neg_cb,circos_layout,circos_show_degree)
def gather_circos_plot(behavior, show_pos, show_neg, layout,show_degree):
    #return plot_as_circos(model_consensus_to_plot[behavior],roi_info,figsize=(8,8),edge_weight=1, title=behavior, show_pos=show_pos, show_neg=show_neg)
    return plot_as_graph(model_consensus_to_plot[behavior],figsize=(20,20),edge_weight=1, title=behavior, show_pos=show_pos, show_neg=show_neg, layout=layout, show_degree=show_degree)
circos_tab = pn.Column(circos_show_pos_cb,circos_show_neg_cb,gather_circos_plot, circos_show_degree,circos_layout)


@pn.depends(behav_select)
def gather_interactive_brain_view(behavior):
    G = nx.from_pandas_adjacency(model_consensus_to_plot[behavior].abs())
    d = [val for node,val in G.degree()]
    fig, ax = plt.subplots(1,1,figsize=(20,10))
    plot = plot_connectome(model_consensus_to_plot[behavior],roi_info[['pos_R','pos_A','pos_S']], node_color=roi_info['RGB'], node_size=d, axes=ax) #, linewidth=1, colorbar_fontsize=10, node_size=d)
    return plot


@pn.depends(behav_select)
def gather_nw_matrix(behavior):
    pos_count = hvplot_fc_nwlevel(model_consensus_to_plot[behavior]>0,title='Positive Correlation',mode='count', add_net_colors=True, add_net_labels=True, cmap='Reds').opts(toolbar=None)
    neg_count = hvplot_fc_nwlevel(model_consensus_to_plot[behavior]<0,title='Negative Correlation',mode='count', add_net_colors=True, add_net_labels=True, cmap='Blues').opts(toolbar=None)
    all_count = hvplot_fc_nwlevel(model_consensus_to_plot[behavior].abs(),title='Full Model',mode='count', add_net_colors=True).opts(toolbar=None)
    count_card = pn.Card(pn.Row(pos_count,neg_count,all_count), title='Number of Edges', width=1500)
    
    pos_pcent = hvplot_fc_nwlevel(model_consensus_to_plot[behavior]>0,title='Positive Correlation',add_net_colors=True, cmap='Reds', add_net_labels=True, clim_max=15).opts(toolbar=None)
    neg_pcent = hvplot_fc_nwlevel(model_consensus_to_plot[behavior]<0,title='Negative Correlation',add_net_colors=True, cmap='Blues', add_net_labels=True, clim_max=15).opts(toolbar=None)
    all_pcent = hvplot_fc_nwlevel(model_consensus_to_plot[behavior].abs(),title='Full Model',add_net_colors=True, clim_max=15).opts(toolbar=None)
    pcent_card = pn.Card(pn.Row(pos_pcent,neg_pcent,all_pcent), title='Percentage of Edges', width=1500)
    return pn.Column(count_card, pcent_card)


@pn.depends(behav_select)
def get_pred_plots(behavior):
    return pn.Card(pn.Row(pn.pane.Matplotlib(get_boxen_plot(behavior), width=150, height=420,tight=True), 
        pn.pane.Matplotlib(get_obs_vs_pred(behavior), width=420, height=420,tight=True)),
        title='Prediction Power')


nws_group_from = pn.widgets.CheckBoxGroup(name='Networks', value=nw_list, options=nw_list, inline=True)
nws_group_to   = pn.widgets.CheckBoxGroup(name='Networks', value=nw_list, options=nw_list, inline=True)
only_sel_nw    = pn.widgets.Checkbox(name='Show nodes for selected networks only', value=False)


@pn.depends(behav_select,nws_group_from,nws_group_to,only_sel_nw)
def plot_brain_model(behavior,sel_nws_from,sel_nws_to,sel_nws_only):
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
    
    #if sel_nws_only is True:
    #    plot_model    = plot_model.loc[:,:,:,sel_nws_union,:].T.loc[:,:,:,sel_nws_union,:]
    #    sel_rois_info = sel_rois_info.set_index('Network').loc[sel_nws_union]
        
        
    # ==============
    G,Gnt = create_graph_from_matrix(plot_model)
    #print(G)
    #Gnt = node_table(G).sort_index() #[::-1]
    # ==============
    brain_view = plot_connectome(adjacency_matrix=plot_model, 
                                     node_coords=sel_rois_info[['pos_R', 'pos_A','pos_S']],
                                     node_color=sel_rois_info['RGB'],node_size=5*Gnt['Degree'],
                                     edge_kwargs={'linewidth':0.5},
                                     node_kwargs={'edgecolor':'k', 'linewidth':0.5},
                                     figure=fig)
    plt.close()
    return pn.pane.Matplotlib(fig)

brain_view_tab=pn.Column(pn.Row('From:',nws_group_from, background='whitesmoke'),pn.Row('To. :',nws_group_to, background='whitesmoke'),only_sel_nw,plot_brain_model)

# 4. Create the dashboard

dashboard = pn.Row(pn.Column(behav_select, get_pred_plots, pn.Tabs(('Circos Plot',circos_tab),('Brain View',brain_view_tab))), 
                   pn.Column(gather_nw_matrix))

dashboard_server = dashboard.show(port=port_tunnel,open=False)

display.Image('./figures/S17_CPM_Dashboard_screenshot.png')

# Once you are done looking at matrices, you can stop the server running this cell
dashboard_server.stop()
