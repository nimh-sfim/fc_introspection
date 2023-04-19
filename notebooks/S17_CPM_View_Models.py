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
import os.path as osp
from utils.basics import RESOURCES_CPM_DIR
from scipy.spatial.distance import squareform
import hvplot.pandas
from tqdm import tqdm
import holoviews as hv
import pickle
from utils.plotting import plot_as_circos, hvplot_fc_nwlevel
from nilearn.plotting import view_connectome, plot_connectome
import panel as pn
import networkx as nx
import numpy as np

import os
port_tunnel = int(os.environ['PORT2'])
print('++ INFO: Second Port available: %d' % port_tunnel)

from utils.basics import ATLASES_DIR, FB_400ROI_ATLAS_NAME
CORR_TYPE='pearson'
E_SUMMARY_METRIC='sum'
CPM_NITERATIONS      = 100
CPM_NULL_NITERATIONS = 10000
SPLIT_MODE           = 'subject_aware'
ATLAS_NAME           = FB_400ROI_ATLAS_NAME
CONFOUNDS            = 'conf_residualized'
BEHAVIORS            = ['Factor1','Factor2','Images','Words','People','Myself','Positive','Negative','Surroundings','Intrusive','Vigilance','Future','Past','Specific']

# Load one case, simply to obtain the number of edges

ref_path = osp.join(RESOURCES_CPM_DIR,'swarm_outputs','real',ATLAS_NAME,SPLIT_MODE, CONFOUNDS,CORR_TYPE+'_'+E_SUMMARY_METRIC,'Images','cpm_Images_rep-{r}.pkl'.format(r=str(1).zfill(5)))
with open(ref_path,'rb') as f:
    ref_data = pickle.load(f)
n_edges = ref_data['models']['pos'].shape[1]

ATLASINFO_PATH = osp.join(ATLASES_DIR,ATLAS_NAME,f'{ATLAS_NAME}.roi_info.csv')
roi_info       = pd.read_csv(ATLASINFO_PATH)

# ## Load all real models

# +
# %%time
models = {}
models_to_vis = {}
for BEHAVIOR in BEHAVIORS:
    models = {(BEHAVIOR,'pos'):pd.DataFrame(index=range(CPM_NITERATIONS), columns=range(n_edges)),
              (BEHAVIOR,'neg'):pd.DataFrame(index=range(CPM_NITERATIONS), columns=range(n_edges))}
    df = pd.DataFrame(index=range(CPM_NITERATIONS),columns=['pos','neg','glm'])
    for r in tqdm(range(CPM_NITERATIONS), desc='Iteration [%s]' % BEHAVIOR):
        path = osp.join(RESOURCES_CPM_DIR,'swarm_outputs','real',ATLAS_NAME,SPLIT_MODE, CONFOUNDS,CORR_TYPE+'_'+E_SUMMARY_METRIC,BEHAVIOR,'cpm_{b}_rep-{r}.pkl'.format(b=BEHAVIOR,r=str(r+1).zfill(5)))
        with open(path,'rb') as f:
            data = pickle.load(f)
        sadad
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

out_path     = '../resources/cpm/plot_tmp/models.pkl'
with open(out_path,'rb') as f:
    data_to_disk = pickle.load(f)
models = data_to_disk['models']
models_to_vis = data_to_disk['models_to_vis']
del data_to_disk

pd.DataFrame(data['models']['pos']).

# ## Compute consensus models for plotting

thresh           = 0.9
model_consensus,num_edges_toshow,model_consensus_to_plot  = {},{},{}
for BEHAVIOR in BEHAVIORS:
    for tail in ['pos','neg']:
        edge_frac                       = models_to_vis[BEHAVIOR,tail]
        model_consensus[BEHAVIOR,tail]  = (edge_frac>=thresh).astype(int)
        num_edges_toshow[BEHAVIOR,tail] = model_consensus[BEHAVIOR,tail].sum()
        print("For the [{behav},{tail}], {edges} edges were selected in at least {pct}% of folds".format(behav=BEHAVIOR,tail=tail, edges=num_edges_toshow[BEHAVIOR,tail], pct=thresh*100))
    model_consensus_to_plot[BEHAVIOR] = pd.DataFrame(squareform(model_consensus[BEHAVIOR,'pos'])-squareform(model_consensus[BEHAVIOR,'neg']),
                          index = roi_info.set_index(['ROI_ID','ROI_Name','Hemisphere','Network']).index,
                          columns= roi_info.set_index(['ROI_ID','ROI_Name','Hemisphere','Network']).index)

# ***

# ## Create Dashboard
#
# 1. Estimate the limits for the colorbar in the NW summary view (connection count mode)

max_counts = []
for BEHAVIOR in BEHAVIORS:
    a = model_consensus_to_plot[BEHAVIOR].abs().groupby('Network').sum().T.groupby('Network').sum()
    for n in a.index:
        a.loc[n,n] = int(a.loc[n,n]/2)
    max_counts.append(a.max().max())
max_counts = np.array(max_counts)
nw_count_max = int(np.quantile(max_counts,.9))

# 2. Create a drop box with all Questions

behav_select = pn.widgets.Select(name='Questions',options=BEHAVIORS,value='Images')


# 3. Create all elements of the dashboard

@pn.depends(behav_select)
def gather_circos_plot(behavior):
    return plot_as_circos(model_consensus_to_plot[behavior],roi_info,figsize=(8,8),edge_weight=1, title=behavior)


@pn.depends(behav_select)
def gather_interactive_brain_view(behavior):
    G = nx.from_pandas_adjacency(model_consensus_to_plot[behavior].abs())
    d = [val for node,val in G.degree()]
    plot = plot_connectome(model_consensus_to_plot[behavior],roi_info[['pos_R','pos_A','pos_S']], node_color=roi_info['RGB'], node_size=d) #, linewidth=1, colorbar_fontsize=10, node_size=d)
    return plot


@pn.depends(behav_select)
def gather_nw_matrix(behavior):
    pos_count = hvplot_fc_nwlevel(model_consensus_to_plot[behavior]>0,title='Positive Correlation',mode='count', add_net_colors=True).opts(toolbar=None)
    neg_count = hvplot_fc_nwlevel(model_consensus_to_plot[behavior]<0,title='Negative Correlation',mode='count', add_net_colors=True).opts(toolbar=None)
    all_count = hvplot_fc_nwlevel(model_consensus_to_plot[behavior].abs(),title='Full Model',mode='count', add_net_colors=True).opts(toolbar=None)
    count_card = pn.Card(pn.Row(pos_count,neg_count,all_count), title='Number of Edges', width=1500)
    pos_pcent = hvplot_fc_nwlevel(model_consensus_to_plot[behavior]>0,title='Positive Correlation',add_net_colors=True).opts(toolbar=None)
    neg_pcent = hvplot_fc_nwlevel(model_consensus_to_plot[behavior]<0,title='Negative Correlation',add_net_colors=True).opts(toolbar=None)
    all_pcent = hvplot_fc_nwlevel(model_consensus_to_plot[behavior].abs(),title='Full Model',add_net_colors=True).opts(toolbar=None)
    pcent_card = pn.Card(pn.Row(pos_pcent,neg_pcent,all_pcent), title='Percentage of Edges', width=1500)
    return pn.Column(count_card, pcent_card)


# 4. Create the dashboard

dashboard = pn.Row(pn.Column(behav_select, gather_circos_plot), 
                   pn.Column(gather_nw_matrix))

dashboard_server = dashboard.show(port=port_tunnel,open=False)

# Once you are done looking at matrices, you can stop the server running this cell
dashboard_server.stop()

hvplot_fc_nwlevel(model_consensus_to_plot['Images']>0, mode='count', title='Positive Correlation', add_net_colors=True) + \
hvplot_fc_nwlevel(model_consensus_to_plot['Images']<0, mode='count', title='Negative Correlation') + \
hvplot_fc_nwlevel(model_consensus_to_plot['Images'].abs(), mode='count', title='Full Model') 


