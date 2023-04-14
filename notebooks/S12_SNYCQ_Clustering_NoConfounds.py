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
import seaborn as sns
import matplotlib.pyplot as plt
import os.path as osp
import hvplot.pandas
import holoviews as hv
from pyclustering.cluster.xmeans import xmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from sklearn.cluster import AgglomerativeClustering
from scipy.stats import kruskal, wilcoxon, ttest_ind, mannwhitneyu

from dataclasses import dataclass


@dataclass
class matrix_class:

    M : np.ndarray # (column)-normalized data matrix
    M_raw : np.ndarray # raw data matrix
    confound : np.ndarray # normalized confounder matrix
    confound_raw : np.ndarray # raw confounder matrix
    nan_mask : np.ndarray # mask matrix for missing entires (0=missing, 1=available)
    row_idx : np.ndarray # global row index (for multiple data matrices)
    col_idx : np.ndarray # global column index (for multiple data matrices)
    mask : np.ndarray # global mask (for multiple data matrices)
    dataname : str # dataname
    subjlist : list # information on subjects (row information)
    itemlist : list # information on items (column information)
    W : np.ndarray # subject embedding (recall M = [W, C]Q^T)
    Q : np.ndarray # item embedding (recall M = [W, C]Q^T)
    C : np.ndarray # confounder matrix
    Qc : np.ndarray # confounders' loadings (recall Q = [RQ, CQ])
    Z : np.ndarray # auxiliary Z=WQ^T (ADMM)
    aZ : np.ndarray # auxiliary variables (ADMM)


import sys
sys.path.append('./mlt/')
from method.ICQF.ICQF import ICQF

from utils.basics import PROC_SNYCQ_DIR, get_sbj_scan_list
from utils.basics import SNYCQ_Questions, SNYCQ_Question_type, SNYCQ_CLUSTERS_INFO_PATH, DATA_DIR
from utils.SNYCQ_NMF_Extra import plot_anwers_distribution_per_question, plot_Q_bars, cluster_scans, plot_W, plot_W_scatter, plot_P,plot_W_heatmap

SBJs, SCANs, SNYCQ = get_sbj_scan_list(when='post_motion', return_snycq=True)

SNYCQ = SNYCQ.drop('Vigilance',axis=1)
Nscans, Nquestions = SNYCQ.shape
print(SNYCQ.shape)

# ***
# # 2. Dimensionality Reduction for SNYCQ

data = np.load('./mlt/output/full_data.npz')
M    = data['M']
nan_mask = data['nan_mask']
confound = data['confound']
age = confound[:,0]
gender = confound[:,1]
MF_data = matrix_class(M, None, None, None, nan_mask, None, None,
                       None, None, None, None, None, None,
                       None, None, None, None)
DIM = 2
clf = ICQF(DIM, rho=3, tau=3, regularizer=1,
                        W_upperbd=(True, 1.0),
                        Q_upperbd=(True, 100),
                        M_upperbd=(True, 100),
                        W_beta=0.0,
                        Q_beta=0.1,
                        max_iter=200,
                        intercept=False)
MF_data, loss_list = clf.fit_transform(MF_data)

P = SNYCQ.values.astype('float')
print('Range of data matrix P:      [min={}, max={}]'.format(np.min(P), np.max(P)))
print('Dimensions of data matrix P: [#scans={}, #questions={}]'.format(P.shape[0],P.shape[1]))
assert np.sum(np.isnan(P)) == 0 # Make sure there are no missing values

W = pd.DataFrame(MF_data.W, index=SNYCQ.index, columns=['Factor {d}'.format(d=d+1) for d in range(DIM)])
Q = pd.DataFrame(MF_data.Q, index=SNYCQ.columns, columns=['Factor {d}'.format(d=d+1) for d in range(DIM)])

Precon = pd.DataFrame(np.dot(W,Q.T),index=SNYCQ.index,columns=SNYCQ.columns)

# ***

plot_Q_bars(Q,SNYCQ,SNYCQ_Questions, SNYCQ_Question_type)

plot_W(W)

plot_W_scatter(W, plot_hist=True, plot_kde=True, figsize=(5,5), marker_size=20)

# ***
# # Automatic Clustering

N_CLUSTERS = 3
cluster_ids = cluster_scans(W, n_clusters=N_CLUSTERS)
cluster_ids = cluster_ids.astype(int)
print('++ INFO: Number of clusters = %d' % int(cluster_ids.max()+1))

clusters_info = pd.DataFrame(index=W.index, columns=['Cluster ID','Cluster Label'])
clusters_info['Cluster ID'] = cluster_ids

cluster_labels_translate = {x:'Middle' for x in np.unique(cluster_ids)}
Waux = W.copy()
Waux['Cluster ID']    = cluster_ids
cluster_labels_translate[float(Waux.groupby('Cluster ID').mean().sort_values(by='Factor 1', ascending=False).iloc[0].name)] = 'Large F1'
cluster_labels_translate[float(Waux.groupby('Cluster ID').mean().sort_values(by='Factor 2', ascending=False).iloc[0].name)] = 'Large F2'
clusters_info['Cluster Label'] = [cluster_labels_translate[c] for c in Waux['Cluster ID']]
clusters_info['Cluster ID'] = 0
clusters_info.loc[clusters_info['Cluster Label']=='Large F1','Cluster ID'] = 1
clusters_info.loc[clusters_info['Cluster Label']=='Large F2','Cluster ID'] = 2
clusters_info.loc[clusters_info['Cluster Label']=='Middle'  ,'Cluster ID'] = 3
del Waux, cluster_ids, cluster_labels_translate

plot_W_scatter(W, clusters_info=clusters_info, plot_kde=True, plot_hist=True, marker_size=20)

clusters_info.value_counts('Cluster Label')

q_order = ['Future', 'Specific', 'Past', 'Positive', 'People', 'Images', 'Words', 'Negative', 'Surroundings', 'Myself', 'Intrusive']

sort_clf1f2_list = []
for cl_label in ['Large F1','Large F2','Middle']:
    aux = pd.concat([W,clusters_info],axis=1)
    aux = aux.reset_index().set_index('Cluster Label').loc[cl_label]
    sort_clf1f2_list = sort_clf1f2_list + list(aux.sort_values(by=['Factor 1','Factor 2']).set_index(['Subject','Run']).index)
sort_clf1f2_idx = pd.Index(sort_clf1f2_list)

sort_clf2f1_list = []
for cl_label in ['Large F1','Large F2','Middle']:
    aux = pd.concat([W,clusters_info],axis=1)
    aux = aux.reset_index().set_index('Cluster Label').loc[cl_label]
    sort_clf2f1_list = sort_clf2f1_list + list(aux.sort_values(by=['Factor 2','Factor 1']).set_index(['Subject','Run']).index)
sort_clf2f1_idx = pd.Index(sort_clf2f1_list)

plot_W_heatmap(W, clusters_info=clusters_info, scan_order=sort_clf2f1_idx)

plot_P(SNYCQ, question_order=q_order, scan_order=sort_clf1f2_idx, clusters_info=clusters_info)

plot_P(Precon, question_order=q_order, scan_order=sort_clf1f2_idx, clusters_info=clusters_info)

aux = pd.concat([W,clusters_info],axis=1)
aux.hvplot.scatter(x='Factor 1', y='Factor 2', c='Cluster Label', aspect='square', cmap=['lightblue','#ff9999','orange']) + aux.hvplot.kde(y='Factor 2', by='Cluster Label', width=500) + aux.hvplot.kde(y='Factor 1', by='Cluster Label', width=500)

clusters_info.to_csv(SNYCQ_CLUSTERS_INFO_PATH)
print('++ INFO: Clustering Membership Info saved to [%s]' % SNYCQ_CLUSTERS_INFO_PATH)

# ***
#
# # Significant Differences in things of interest across extreme clusters

scans_of_interest = clusters_info[clusters_info['Cluster Label']!='Middle'].index

df = pd.DataFrame(index = scans_of_interest,columns=['Cluster Label','Rel. Motion (mean)','Rel. Motion (max)','Vigilance','Age','Gender'])

_,_,aux_snycq = get_sbj_scan_list(when='post_motion')

for scan in df.index:
    df.loc[scan,'Vigilance'] = aux_snycq.loc[scan,'Vigilance']
    df.loc[scan,'Cluster Label'] = clusters_info.loc[scan,'Cluster Label']
df = df.infer_objects()

df.hvplot.kde(y='Vigilance',by='Cluster Label', alpha=.5, title='Vigilance')

ttest_ind(df.set_index('Cluster Label').loc['Large F1','Vigilance'],df.set_index('Cluster Label').loc['Large F2','Vigilance'],alternative='two-sided')

mannwhitneyu(df.set_index('Cluster Label').loc['Large F1','Vigilance'],df.set_index('Cluster Label').loc['Large F2','Vigilance'],alternative='two-sided')

for scan in df.index:
    sbj,run = scan
    _,_,_,_,run_num,_,run_acq = run.split('-')
    path = osp.join(DATA_DIR,'PrcsData',sbj,'preprocessed','func','pb01_moco','_scan_id_ses-02_task-rest_acq-{run_acq}_run-{run_num}_bold'.format(run_num=run_num,run_acq=run_acq),'rest_realigned_rel.rms')
    mot  = np.loadtxt(path)
    df.loc[scan,'Rel. Motion (mean)'] = mot.mean()
    df.loc[scan,'Rel. Motion (max)']  = mot.max()

df = df.infer_objects()

df.hvplot.kde(y='Rel. Motion (mean)',by='Cluster Label', alpha=.5, title='Rel. Motion (mean)') + df.hvplot.kde(y='Rel. Motion (max)',by='Cluster Label', alpha=.5, title='Rel. Motion (max)') 

ttest_ind(df.set_index('Cluster Label').loc['Large F1','Rel. Motion (mean)'],df.set_index('Cluster Label').loc['Large F2','Rel. Motion (mean)'],alternative='two-sided')

ttest_ind(df.set_index('Cluster Label').loc['Large F1','Rel. Motion (max)'],df.set_index('Cluster Label').loc['Large F2','Rel. Motion (max)'],alternative='two-sided')

# ***
#
# # END OF CODE OF INTEREST SO FAR

clusters_info =pd.read_csv(SNYCQ_CLUSTERS_INFO_PATH, index_col=['Subject','Run'])

clusters_info

clusters_info.reset_index().set_index('Cluster Label').loc['Large F1']











# ***
# # Manual Clustering - Option 1

f1_33 = W['Factor 1'].quantile(.33)
f1_66 = W['Factor 1'].quantile(.66)
f2_33 = W['Factor 2'].quantile(.33)
f2_66 = W['Factor 2'].quantile(.66)
G1 = W.loc[(W['Factor 1'] >= f1_66) & (W['Factor 2'] <= f2_33)]
G2 = W.loc[(W['Factor 1'] <= f1_33) & (W['Factor 2'] >= f2_66)]

line_opts = {'line_dash':'dashed', 'line_color':'gray', 'line_width':1}
W.hvplot.scatter(x='Factor 1', y = 'Factor 2', aspect='square', c='k') * \
hv.HLine(f2_33).opts(**line_opts) * hv.HLine(f2_66).opts(**line_opts) * hv.VLine(f1_33).opts(**line_opts) * hv.VLine(f1_66).opts(**line_opts) * \
G1.hvplot.scatter(x='Factor 1',y='Factor 2', c='lightblue', aspect='square') * \
G2.hvplot.scatter(x='Factor 1',y='Factor 2', c='#ff9999', aspect='square')

print('++ INFO: Sizes of manually generated groups: G1=%d | G2=%d' % (G1.shape[0], G2.shape[0]))

G1 = W.loc[(W['Factor 1'] >= 0.6) & (W['Factor 2'] <= 0.4)]
G2 = W.loc[(W['Factor 1'] <= 0.4) & (W['Factor 2'] >= 0.6)]

line_opts = {'line_dash':'dashed', 'line_color':'gray', 'line_width':1}
W.hvplot.scatter(x='Factor 1', y = 'Factor 2', aspect='square', c='k') * \
hv.HLine(0.4).opts(**line_opts) * hv.HLine(0.6).opts(**line_opts) * hv.VLine(0.4).opts(**line_opts) * hv.VLine(0.6).opts(**line_opts) * \
G1.hvplot.scatter(x='Factor 1',y='Factor 2', c='lightblue', aspect='square', xlim=(0,1), ylim=(0,1)) * \
G2.hvplot.scatter(x='Factor 1',y='Factor 2', c='#ff9999', aspect='square', xlim=(0,1), ylim=(0,1))

print('++ INFO: Sizes of manually generated groups: G1=%d | G2=%d' % (G1.shape[0], G2.shape[0]))


