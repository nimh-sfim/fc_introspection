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

import os.path as osp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import getpass
import glob
import holoviews as hv
import pickle
hv.extension('bokeh')
from matplotlib import pyplot

from utils.SNYCQ_NMF_Extra import plot_Q_bars, group_questions, plot_W

import hvplot.pandas
import panel as pn

from sklearn.cluster import AgglomerativeClustering

from utils.basics import SNYCQ_Questions, SNYCQ_Question_type, get_sbj_scan_list

from mlt.NMF_algorithm import NMF_model
from utils.basics import PRJ_DIR, DATA_DIR, ORIG_SNYCQ_PATH

CLUSTERS_TO_EXPLORE = [2,3,4,5,6,7,8]

# ***
# ### 1. Load Final SNYCQ Data

sbj_list, scan_list, SNYCQ_data = get_sbj_scan_list(when='post_motion', return_snycq=True)
SNYCQ_data_wVigilance = SNYCQ_data.copy()
SNYCQ_data = SNYCQ_data.drop(['Vigilance'], axis=1)
print(SNYCQ_data.shape)

# ### 2. Convert SNYCQ to Numpy Matrix

# Load data matrix from SYNCQ data
P = SNYCQ_data.values.astype('float')
print('++ INFO Range of data matrix P: [{}, {}]'.format(np.min(P), np.max(P)))
print('++ INFO Shape of data matrix P: %s' % str(P.shape))
assert np.sum(np.isnan(P)) == 0

fig, ax = pyplot.subplots(figsize=(16,4))
sns.heatmap(P.T, ax=ax, cmap='Blues', cbar=True, cbar_kws = dict(pad=0.1))
bottom, top = ax.get_ylim()
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel('Subjects', fontsize=16)
ax.set_ylabel('Questions', fontsize=16)
ax.set_title('Data Matrix $P^T$', fontsize=18)
ax.set_ylim(bottom + 0.5, top - 0.5)
pyplot.show()

# ***
# ### 3. Compute Embedding

SPARSITY = 0.05 #0.001 #1.5 # 0.001
DIMENSIONS = 2
Factor_cols = ['Factor'+str(i+1) for i in np.arange(DIMENSIONS)]
model = NMF_model(data_matrix=P, data_mask=np.ones_like(P), dimension=DIMENSIONS, method='admm', Wbound=(True, 1.0), sparsity_parameter=SPARSITY)

# saves the output to a datafile.npz
model.output_datafile()

W, Q = model.decomposition()
print('++ INFO: W.shape=%s' % str(W.shape))
print('++ INFO: Q.shape=%s' % str(Q.shape))
print('++ INFO: P.shape=%s' % str(P.shape))

import pickle

a = pd.read_pickle('./results/result.pickle')

pd.DataFrame(a['obj_history']).hvplot(logy=True)

Q_df                = pd.DataFrame(Q,columns=Factor_cols)
Q_df['Question ID'] = SNYCQ_data.columns
Q_df['Question']    = [SNYCQ_Questions[k] for k in SNYCQ_data.columns]
Q_df                = Q_df.set_index(['Question','Question ID'])
Q_df.head(11).round(2)

W_df = pd.DataFrame(W, columns=Factor_cols, index=SNYCQ_data.index)
W_df.head(3).round(2)

# + [markdown] tags=[]
# ***
# ### 4. Explore Axes of the new space (Questions)
# -

Q_weights_plot = Q_df.round(1).droplevel('Question').sort_values(by=Factor_cols, ascending=True).hvplot.heatmap(cmap='RdBu_r', clabel='Factor Loadings:', height=400, width=320, fontsize={'ticks':12, 'clabel':12}).opts(toolbar=None, xrotation=45, yrotation=0)

Q_bars_plot = plot_Q_bars(Q, SNYCQ_data, SNYCQ_Questions, SNYCQ_Question_type)

pn.Row(Q_weights_plot * hv.Labels(Q_weights_plot),plot_Q_bars(Q, SNYCQ_data, SNYCQ_Questions, SNYCQ_Question_type))

W_df.hvplot.scatter(x='Factor1',y='Factor2', aspect='square', hover_cols=['Subject','Run'], frame_width=800, c='k', s=100, xlim=(-.01,1.01), ylim=(-.01,1.01),xlabel='Factor 1', ylabel='Factor 2', fontsize={'xlabel':40, 'ylabel':40, 'xticks':40, 'yticks':40}, grid=True).opts(toolbar=None)

a = pd.DataFrame(SNYCQ_data.loc[('sub-010015','post-ses-02-run-02-acq-PA')])
a.columns=['SNYCQ']
p = a.hvplot.heatmap(cmap='viridis',clim=(0,100),width=270, height=400, fontsize={'yticks':16}).opts(toolbar=None)
p * (hv.Labels(p))

mot_path = osp.join(DATA_DIR,'PrcsData','sub-010135','preprocessed','func','pb01_moco','_scan_id_ses-02_task-rest_acq-PA_run-02_bold','rest_realigned.par')
mot_data = np.loadtxt(mot_path)
mot_df = pd.DataFrame(mot_data)
a = mot_df.hvplot()
type(a)

mot_df.hvplot(width=1000, legend=False)

mot_path = osp.join(DATA_DIR,'PrcsData','sub-010135','preprocessed','func','pb01_moco','_scan_id_ses-02_task-rest_acq-PA_run-02_bold','rest_realigned_rel.rms')
mot_data = np.loadtxt(mot_path)
mot_df = pd.DataFrame(mot_data)
mot_df.hvplot()

SNYCQ_data['Specific'].hvplot.hist(normed=True) * SNYCQ_data['Specific'].hvplot.kde()

# ***
# ### Manual Groups (Factor 1 extremes)

print('++ INFO:          Factor 1 groups:')
F1_Low  = W_df[W_df['Factor1']<.4]
F1_High = W_df[W_df['Factor1']>.6]
print(' +       [F1 <.4] has %d scans' % F1_Low.shape[0])
print(' +       [F1 >.6] has %d scans' % F1_High.shape[0])

F1_Groups = {'Low':F1_Low.index,'High':F1_High.index}

pickle.dump(F1_Groups, open( "/data/SFIMJGC_Introspec/prj2021_dyneusr/Resources.sFC/F1_groups.pkl", "wb" ) )

a = SNYCQ_data_wVigilance.loc[F1_Groups['Low']].melt()
a['variable'] = a['variable'].astype('category')
a['F1_Group'] = 'Low'
a['F1_Group'] = a['F1_Group'].astype('category')
b = SNYCQ_data_wVigilance.loc[F1_Groups['High']].melt()
b['variable'] = b['variable'].astype('category')
b['F1_Group'] = 'High'
b['F1_Group'] = b['F1_Group'].astype('category')
c=pd.concat([a,b],axis=0)
c.columns = ['Question','Answer','F1_Group']

sns.set(font_scale=1.5)
#fig, ax = plt.subplots(1,1,figsize=(7,10))
#sns.boxplot(data=c,y='Question',x='Answer', hue='F1_Group', orient='h', order=list(Q_df.sort_values(by='Factor1', ascending=False).index.get_level_values(1)))
fig, ax = plt.subplots(1,1,figsize=(10,7))
sns.boxplot(data=c,x='Question',y='Answer', hue='F1_Group', orient='v', order=list(Q_df.sort_values(by='Factor1', ascending=False).index.get_level_values(1))+['Vigilance'])
ax.set_xticklabels(ax.get_xticklabels(),rotation = 90);
plt.legend(loc='lower center')

W_df.hvplot.scatter(x='Factor1',y='Factor2', aspect='square',c='k',fontsize={'xticks':14, 'yticks':14,'cticks':14, 'xlabel':14, 'ylabel':14}) * \
W_df.loc[F1_Groups['Low']].hvplot.scatter(x='Factor1',y='Factor2', aspect='square',c='lightblue') * \
W_df.loc[F1_Groups['High']].hvplot.scatter(x='Factor1',y='Factor2', aspect='square',c='orange') 

# ***
# ### Manual Groups (Factor 2 extremes)

print('++ INFO: Factor 2 groups:')
F2_Low  = W_df[W_df['Factor2']<.4]
F2_High = W_df[W_df['Factor2']>.6]
print(' +       [F2 <.4] has %d scans' % F2_Low.shape[0])
print(' +       [F2 >.6] has %d scans' % F2_High.shape[0])

F2_Groups = {'Low':F2_Low.index,'High':F2_High.index}

pickle.dump(F2_Groups, open( "/data/SFIMJGC_Introspec/prj2021_dyneusr/Resources.sFC/F2_groups.pkl", "wb" ) )

a = SNYCQ_data_wVigilance.loc[F2_Groups['Low']].melt()
a['variable'] = a['variable'].astype('category')
a['F2_Group'] = 'Low'
a['F2_Group'] = a['F2_Group'].astype('category')
b = SNYCQ_data_wVigilance.loc[F2_Groups['High']].melt()
b['variable'] = b['variable'].astype('category')
b['F2_Group'] = 'High'
b['F2_Group'] = b['F2_Group'].astype('category')
c=pd.concat([a,b],axis=0)
c.columns = ['Question','Answer','F2_Group']

sns.set(font_scale=1.5)
#fig, ax = plt.subplots(1,1,figsize=(7,10))
#sns.boxplot(data=c,y='Question',x='Answer', hue='F2_Group', orient='h', order=list(Q_df.sort_values(by='Factor2', ascending=False).index.get_level_values(1)))
fig, ax = plt.subplots(1,1,figsize=(10,7))
sns.boxplot(data=c,x='Question',y='Answer', hue='F2_Group', orient='v', order=list(Q_df.sort_values(by='Factor2', ascending=False).index.get_level_values(1))+['Vigilance'])
ax.set_xticklabels(ax.get_xticklabels(),rotation = 90);
plt.legend(loc='upper right')

W_df.hvplot.scatter(x='Factor1',y='Factor2', aspect='square',c='k', fontsize={'xticks':14, 'yticks':14,'cticks':14, 'xlabel':14, 'ylabel':14}) * \
W_df.loc[F2_Groups['Low']].hvplot.scatter(x='Factor1',y='Factor2', aspect='square',c='lightblue') * \
W_df.loc[F2_Groups['High']].hvplot.scatter(x='Factor1',y='Factor2', aspect='square',c='orange') 

# ***
# ### Manual Groups (Corners)

print('++ INFO: Corners groups:')
F1 = W_df[(W_df['Factor1']>.6) & (W_df['Factor2']<.4)]
F2 = W_df[(W_df['Factor1']<.4) & (W_df['Factor2']>.6)]
print(' +       F1 has %d scans' % F1.shape[0])
print(' +       F2 has %d scans' % F2.shape[0])

Corner_Groups = {'F1': F1.index,
                 'F2': F2.index}

pickle.dump(Corner_Groups, open( "/data/SFIMJGC_Introspec/prj2021_dyneusr/Resources.sFC/Corner_groups.pkl", "wb" ) )

a = SNYCQ_data_wVigilance.loc[Corner_Groups['F2']].melt()
a['variable'] = a['variable'].astype('category')
a['Corner_Group'] = 'F2'
a['Corner_Group'] = a['Corner_Group'].astype('category')
b = SNYCQ_data_wVigilance.loc[Corner_Groups['F1']].melt()
b['variable'] = b['variable'].astype('category')
b['Corner_Group'] = 'F1'
b['Corner_Group'] = b['Corner_Group'].astype('category')
c=pd.concat([a,b],axis=0)
c.columns = ['Question','Answer','Corner_Group']

sns.set(font_scale=1.5)
#fig, ax = plt.subplots(1,1,figsize=(7,10))
#sns.boxplot(data=c,y='Question',x='Answer', hue='Corner_Group', orient='h', order=list(Q_df.sort_values(by='Factor1', ascending=False).index.get_level_values(1)))
fig, ax = plt.subplots(1,1,figsize=(10,7))
sns.boxplot(data=c,x='Question',y='Answer', hue='Corner_Group', orient='v', order=list(Q_df.sort_values(by='Factor1', ascending=False).index.get_level_values(1))+['Vigilance'])
ax.set_xticklabels(ax.get_xticklabels(),rotation = 90);
plt.legend(loc='upper right')

W_df.hvplot.scatter(x='Factor1',y='Factor2', aspect='square',c='k',fontsize={'xticks':14, 'yticks':14,'cticks':14, 'xlabel':14, 'ylabel':14}) * \
W_df.loc[Corner_Groups['F1']].hvplot.scatter(x='Factor1',y='Factor2', aspect='square',c='orange') * \
W_df.loc[Corner_Groups['F2']].hvplot.scatter(x='Factor1',y='Factor2', aspect='square',c='lightblue') 

print('++ INFO: Corners groups:')
F1 = W_df[(W_df['Factor1']>.8) & (W_df['Factor2']<.2)]
F2 = W_df[(W_df['Factor1']<.4) & (W_df['Factor2']>.6)]
print(' +       F1  has %d scans' % F1.shape[0])
print(' +       F2 has %d scans'  % F2.shape[0])

Corner_Groups = {'F1': F1.index,
                 'F2': F2.index}

pickle.dump(Corner_Groups, open( "/data/SFIMJGC_Introspec/prj2021_dyneusr/Resources.sFC/Corner_groups_equalsize.pkl", "wb" ) )

a = SNYCQ_data_wVigilance.loc[Corner_Groups['F2']].melt()
a['variable'] = a['variable'].astype('category')
a['Corner_Group'] = 'F2'
a['Corner_Group'] = a['Corner_Group'].astype('category')
b = SNYCQ_data_wVigilance.loc[Corner_Groups['F1']].melt()
b['variable'] = b['variable'].astype('category')
b['Corner_Group'] = 'F1'
b['Corner_Group'] = b['Corner_Group'].astype('category')
c=pd.concat([a,b],axis=0)
c.columns = ['Question','Answer','Corner_Group']

sns.set(font_scale=1.5)
#fig, ax = plt.subplots(1,1,figsize=(7,10))
#sns.boxplot(data=c,y='Question',x='Answer', hue='Corner_Group', orient='h', order=list(Q_df.sort_values(by='Factor1', ascending=False).index.get_level_values(1)))
fig, ax = plt.subplots(1,1,figsize=(10,7))
sns.boxplot(data=c,x='Question',y='Answer', hue='Corner_Group', orient='v', order=list(Q_df.sort_values(by='Factor1', ascending=False).index.get_level_values(1))+['Vigilance'])
ax.set_xticklabels(ax.get_xticklabels(),rotation = 90);
plt.legend(loc='upper right')

W_df.hvplot.scatter(x='Factor1',y='Factor2', aspect='square',c='k',fontsize={'xticks':14, 'yticks':14,'cticks':14, 'xlabel':14, 'ylabel':14}) * \
W_df.loc[Corner_Groups['F1']].hvplot.scatter(x='Factor1',y='Factor2', aspect='square',c='orange') * \
W_df.loc[Corner_Groups['F2']].hvplot.scatter(x='Factor1',y='Factor2', aspect='square',c='lightblue') 

# ***
# ### 5. Cluster Scans

# Clustering based on W
W_orig = W_df.copy()

for num_clusters in CLUSTERS_TO_EXPLORE :
    clustering = AgglomerativeClustering(n_clusters=num_clusters).fit(W_orig)
    W_df['clusterID_W_'+str(num_clusters)] = clustering.labels_

# Clustering based on P
for num_clusters in CLUSTERS_TO_EXPLORE :
    clustering = AgglomerativeClustering(n_clusters=num_clusters).fit(P)
    W_df['clusterID_P_'+str(num_clusters)] = clustering.labels_

W_df

# ***
# ### 6. Reconstructed Data

snycq_reconstructed = pd.DataFrame(np.dot(W_df[['Factor1','Factor2']].values,Q_df.values.T), index=SNYCQ_data.index, columns=SNYCQ_data.columns)

P_recon=snycq_reconstructed.to_numpy()

# Clustering based on P reconstructed or P'
for num_clusters in [2,3,4]:
    clustering = AgglomerativeClustering(n_clusters=num_clusters).fit(P_recon)
    W_df['clusterID_P_recon_'+str(num_clusters)] = clustering.labels_

W_df.head()

W_df = pd.concat([W_df,SNYCQ_data],axis=1)

SNYCQ_data[W_df['clusterID_W_3']==0].reset_index(drop=True).hvplot.heatmap(cmap='viridis') + SNYCQ_data[W_df['clusterID_W_3']==1].reset_index(drop=True).hvplot.heatmap(cmap='viridis') + SNYCQ_data[W_df['clusterID_W_3']==2].reset_index(drop=True).hvplot.heatmap(cmap='viridis')

# + [markdown] tags=[]
# ***
# ### 7. Visualize Subject clustering
# -

nclusters_select = pn.widgets.Select(name='Number of Clusters',options=[1] + CLUSTERS_TO_EXPLORE, value=2, width=100)
data_select      = pn.widgets.Select(name='Data Matrix',options=['Orig','Reconstructed'], value='Orig', width=100)
cluster_input_select = pn.widgets.Select(name='Clust Input Matrix',options=['P','W','P_recon'], value='W', width=100)
colorby_select   = pn.widgets.Select(name='Color', options=['Cluster'] + list(SNYCQ_data.columns),width=100)

import matplotlib.colors as mcolors
mcolors.TABLEAU_COLORS
cmaps_scatter = {i:list(mcolors.TABLEAU_COLORS.values())[0:i] for i in np.arange(10) }


# + active=""
# cmaps_scatter   = {1: ['#ffffff'],
#                    2: ['#3377b4', '#f37f11'],
#                    3: ['#3377b4','#f37f11','#269624'],
#                    4: ['#3377b4','#f37f11','#cf2321','#269624'],
#                    5: ['#3377b4','#f37f11','#cf2321','#269624',]}
# -

@pn.depends(nclusters_select,cluster_input_select, colorby_select)
def plot_sbj_embedding(n_clusters,cluster_input_matrix, colorby):
    if colorby == 'Cluster':
        if n_clusters == 1:
            emb_plot = W_df.hvplot.scatter(x='Factor1',y='Factor2', hover_cols=['Subject','Run'], c='k', colorbar=True, aspect='square', fontsize={'xticks':14, 'yticks':14,'cticks':14, 'xlabel':14, 'ylabel':14})
        else:
            emb_plot = W_df.hvplot.scatter(x='Factor1',y='Factor2', color='clusterID_{m}_{k}'.format(m=cluster_input_matrix,k=str(n_clusters)), hover_cols=['Subject','Run'], cmap=cmaps_scatter[n_clusters], colorbar=False, aspect='square', fontsize={'xticks':14, 'yticks':14,'cticks':14})
    else:
        emb_plot = W_df.hvplot.scatter(x='Factor1',y='Factor2', color=colorby, hover_cols=['Subject','Run'], colorbar=True, aspect='square', fontsize={'xticks':14, 'yticks':14,'cticks':14})
    return emb_plot


@pn.depends(nclusters_select,cluster_input_select)
def plot_sorted_W(n_clusters,cluster_input_matrix):
    if n_clusters==1:
        W_plot = W_df[['Factor1','Factor2']].reset_index(drop=True).T.hvplot.heatmap(height=100, width=900, cmap='Blues_r', fontsize={'xticks':14, 'yticks':14,'cticks':14}).opts(toolbar=None)
    else:
        sorted_idx = W_df.sort_values(by='clusterID_{m}_{k}'.format(m=cluster_input_matrix,k=str(n_clusters))).index
        W_plot     = W_df[['Factor1','Factor2']].loc[sorted_idx].reset_index(drop=True).T.hvplot.heatmap(height=100, width=900, cmap='Blues_r', fontsize={'xticks':14, 'yticks':14,'cticks':14}).opts(toolbar=None)
    return W_plot


@pn.depends(nclusters_select,data_select,cluster_input_select)
def plot_sorted_SNYCQ(n_clusters,mat_type,cluster_input_matrix):
    if mat_type == 'Orig':
        aux = SNYCQ_data
    if mat_type == 'Reconstructed':
        aux = snycq_reconstructed
    Q_sorted_index = list(Q_df.sort_values(by=['Factor2']).index.get_level_values('Question ID'))
    if n_clusters==1:
        snycq_plot = aux.loc[:,Q_sorted_index].reset_index(drop=True).T.hvplot.heatmap(width=900,cmap='viridis', shared_axes=False, fontsize={'xticks':14, 'yticks':14,'cticks':14})
    else:
        W_sorted_idx = W_df.sort_values(by='clusterID_{m}_{k}'.format(m=cluster_input_matrix,k=str(n_clusters))).index
        snycq_plot = aux.loc[W_sorted_idx,Q_sorted_index].reset_index(drop=True).T.hvplot.heatmap(width=900, cmap='viridis',shared_axes=False, fontsize={'xticks':14, 'yticks':14,'cticks':14})
    return snycq_plot


Q_df.sort_values(by=['Factor2']).index.get_level_values('Question ID')


@pn.depends(nclusters_select,data_select,cluster_input_select)
def plot_both_sorted(n_clusters,mat_type,cluster_input_matrix):
    return (plot_sorted_W(n_clusters,cluster_input_matrix) + plot_sorted_SNYCQ(n_clusters, mat_type,cluster_input_matrix)).cols(1)


data_dashboard = pn.Column(pn.Row(pn.Column(nclusters_select,data_select, cluster_input_select, colorby_select),plot_sbj_embedding),
plot_both_sorted)

import os
port_tunnel = int(os.environ['PORT2'])
print('++ INFO: Second Port available: %d' % port_tunnel)

data_dashboard_server = data_dashboard.show(port=port_tunnel,open=False)

data_dashboard_server.stop()

# ***
# ### 8. Save everything to disk

W_path = osp.join(PRJ_DIR,'Resources.SNYCQ_NMF','W.csv')
Q_path = osp.join(PRJ_DIR,'Resources.SNYCQ_NMF','Q.csv')

W_df.to_csv(W_path)
Q_df.to_csv(Q_path)


