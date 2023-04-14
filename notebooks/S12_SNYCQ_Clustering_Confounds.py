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
# This script will run the Sparse Box-Constrained Non-Negative Matrix Factorization on the 11 self-report questions (excluding wakefulness). This is the version that separately applies sparsity to the Q and W matrices and that also accounts for potential demographics condounds.
#
# Before running this notebook, you need to run Notebook S11 to decide on the best hyper-paramteres (e.g., dimensionality and sparsity values)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os.path as osp
import holoviews as hv
from holoviews import opts
from scipy.stats import ttest_ind, mannwhitneyu
import panel as pn
from wordcloud import WordCloud
import hvplot.pandas
from IPython.display import Markdown as md


from matplotlib import rc
font_dict = {'family' : 'arial',
        'weight' : 'normal',
        'size'   : 14}
rc('font', **font_dict)

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

from utils.basics import get_sbj_scan_list, SNYCQ_Questions, SNYCQ_Question_type, SNYCQ_CLUSTERS_INFO_PATH, DATA_DIR
from utils.SNYCQ_NMF_Extra import plot_Q_bars, cluster_scans, plot_W, plot_W_scatter, plot_P,plot_W_heatmap

# # 1. Load list of Scans, Subjects and the SNYCQ dataframe

# Get list of scans, subjects and the SNYCQ dataframe for the data entering the analyses

SBJs, SCANs, SNYCQ = get_sbj_scan_list(when='post_motion', return_snycq=True)

# Remove the Vigilance/Wakefulness Question from the dataframe

SNYCQ = SNYCQ.drop('Vigilance',axis=1)
Nscans, Nquestions = SNYCQ.shape
print(SNYCQ.shape)

# # 2. Apply Sparse Box-Constrained Non-Negative Factorization to the SNYCQ dataframe
#
# The line below will run the algorithm with the following hyper-parameters:
#
# * Attempt Demographic Confound Modeling is `True`
# * W bounded to range `[0,1]`
# * Q bounded to range `[0,100]`
# * W sparsity is `0.0`
# * Q sparsity is `0.01`
# * Number of maximum iterations is `200`
# * Dimensionality is `2`

data  = np.load('./mlt/output/full_data.npz')
M_arr = data['M']
nan_mask = data['nan_mask']
confound = data['confound']
MF_data = matrix_class(M_arr, None, confound, None, nan_mask, None, None,
                       None, None, None, None, None, None,
                       None, None, None, None)
DIM=2
clf = ICQF(DIM, rho=3, tau=3, regularizer=1,
                        W_upperbd=(True, 1.0),
                        Q_upperbd=(True, 100),
                        M_upperbd=(True, 100),
                        W_beta=0.0,
                        Q_beta=0.01,
                        max_iter=200)
MF_data, loss_list = clf.fit_transform(MF_data)

# Sanity check to ensure that the inputs used in the hyper-parameter exploration and here on the final application of the algorithm are the same

M = SNYCQ.values.astype('float')
print('Range of data matrix M:      [min={}, max={}]'.format(np.min(M), np.max(M)))
print('Dimensions of data matrix M: [#scans={}, #questions={}]'.format(M.shape[0],M.shape[1]))
assert np.sum(np.isnan(M)) == 0 # Make sure there are no missing values
assert np.all(M_arr==M), "Input to algorithm based on data given to MLT and mine are NOT the same"

# # 2.1. Convert outputs to DataFrames with meaningful indexing for plotting

W = pd.DataFrame(MF_data.W, index=SNYCQ.index, columns=['Factor {d}'.format(d=d+1) for d in range(DIM)])
Q = pd.DataFrame(MF_data.Q, index=SNYCQ.columns, columns=['Factor {d}'.format(d=d+1) for d in range(DIM)])
C = pd.DataFrame(MF_data.C, index=SNYCQ.index, columns=['Age (elder)','Gender (M)','Age (younger)','Gender (F)','Intercept'])
C = C[['Intercept','Age (younger)','Age (elder)','Gender (M)','Gender (F)']] # Sorting to make it more interpretable
Qc = pd.DataFrame(MF_data.Qc, index=SNYCQ.columns, columns = ['Age (elder)','Gender (M)','Age (younger)','Gender (F)','Intercept'])
Qc = Qc[['Intercept','Age (younger)','Age (elder)','Gender (M)','Gender (F)']] # Sorting to make it more interpretable


# # 3. Plot the different outputs from the algorithm
#
# First, we will plot the W matrix (low dimensional represenation of interest) and the Qc matrix (encoding of the demographic data)

W_plot_unsroted = W.reset_index(drop=True).hvplot.heatmap(cmap='Greens', width=300, height=500, fontscale=1.2, clim=(0,1)).opts( colorbar_opts={'title':'W Matrix'}, xrotation=90, toolbar=None)
C_plot_unsorted = C.reset_index(drop=True).hvplot.heatmap(cmap='Purples', width=300, height=500, fontscale=1.2).opts( colorbar_opts={'title':'C Matrix'}, xrotation=90, toolbar=None)
f = pn.Row(W_plot_unsroted,C_plot_unsorted)
f.save('./figures/W_and_C_unsorted.png')

text="![](./figures/W_and_C_unsorted.png)"
md("%s"%text)

text="![](figures/W_and_C_unsorted.png)"
md("%s"%text)

f.show('png')

# Next, we will plot the Q matrix (with relationships between questions and low dimensional factors) and the Qc matrix (with information about how responses relate to demographics)

Q_plot_unsorted  = Q.hvplot.heatmap( cmap='Oranges', width=300, height=500, clim=(0,100), fontscale=1.2).opts( colorbar_opts={'title':'Q Matrix'}, xrotation=90, toolbar=None)
Qc_plot_unsorted = Qc.hvplot.heatmap(cmap='Reds', width=300, height=500, clim=(0,100), fontscale=1.2).opts( colorbar_opts={'title':'Qc Matrix'}, xrotation=90, toolbar=None)
f = pn.Row(Q_plot_unsorted, Qc_plot_unsorted)
f.save('./figures/Q_and_Qc_unsorted_same_scale.png')
f

Q_plot_unsorted  = Q.hvplot.heatmap( cmap='Oranges', width=300, height=500, clim=(0,100), fontscale=1.2).opts( colorbar_opts={'title':'Q Matrix'}, xrotation=90, toolbar=None)
Qc_plot_unsorted = Qc.hvplot.heatmap(cmap='Reds', width=300, height=500, clim=(0,10), fontscale=1.2).opts( colorbar_opts={'title':'Qc Matrix'}, xrotation=90, toolbar=None)
f = pn.Row(Q_plot_unsorted, Qc_plot_unsorted)
f.save('./figures/Q_and_Qc_unsorted_diff_scale.png')

sorted_q = Q.sort_values(by=['Factor 1','Factor 2'],ascending=False).index
Q_plot_unsorted  = Q.loc[sorted_q].hvplot.heatmap( cmap='Oranges', width=300, height=500, clim=(0,100), fontscale=1.2).opts( colorbar_opts={'title':'Q Matrix'}, xrotation=90, toolbar=None)
Qc_plot_unsorted = Qc.loc[sorted_q].hvplot.heatmap(cmap='Reds', width=300, height=500, clim=(0,10), fontscale=1.2).opts( colorbar_opts={'title':'Qc Matrix'}, xrotation=90, toolbar=None)
f = pn.Row(Q_plot_unsorted, Qc_plot_unsorted)
f.save('./figures/Q_and_Qc_sorted_diff_scale.png')

sorted_q = Q.sort_values(by=['Factor 1','Factor 2'],ascending=False).index
Q_plot_unsorted  = Q.loc[sorted_q].hvplot.heatmap( cmap='Oranges', width=300, height=500, clim=(0,100), fontscale=1.2).opts( colorbar_opts={'title':'Q Matrix'}, xrotation=90, toolbar=None)
Qc_plot_unsorted = Qc.loc[sorted_q].hvplot.heatmap(cmap='Reds', width=300, height=500, clim=(0,100), fontscale=1.2).opts( colorbar_opts={'title':'Qc Matrix'}, xrotation=90, toolbar=None)
f = pn.Row(Q_plot_unsorted, Qc_plot_unsorted)
f.save('./figures/Q_and_Qc_sorted_same_scale.png')

f = pn.Row(W.reset_index(drop=True).hvplot.heatmap(cmap='Greens', ylabel='Scans', width=250, height=500, fontscale=1.2, clim=(0,1)).opts( colorbar_opts={'title':'W Matrix'}, xrotation=90, toolbar=None))
f.save('./figures/W_unsorted.png')
f


Q.loc[sorted_q].hvplot.heatmap(width=275,height=400,cmap='Oranges', clim=(0,100), fontscale=1.2).opts( colorbar_opts={'title':'Q Matrix'}, xrotation=90)

Qc.hvplot.heatmap(cmap='Reds', width=300, height=500, clim=(0,100), fontscale=1.2).opts( colorbar_opts={'title':'Qc Matrix'}, xrotation=90, toolbar=None)

Q['Factor 2'].to_dict()

x, y = np.ogrid[:300, :300]
mask = (x - 150) ** 2 + (y - 150) ** 2 > 130 ** 2
mask = 255 * mask.astype(int)

wc = WordCloud(max_font_size=50, width=200, height=200,contour_color='black', contour_width=3, colormap='jet',max_words=3, background_color='white', repeat=True, mask=mask).fit_words(Q['Factor 2'].to_dict())
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")

wc = WordCloud(max_font_size=50, width=200, height=200,contour_color='black', contour_width=3, colormap='jet',max_words=4, background_color='white', repeat=True, mask=mask).fit_words(Q['Factor 1'].to_dict())
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")

Precon = pd.DataFrame(np.dot(W,Q.T),index=SNYCQ.index,columns=SNYCQ.columns)

# ***

plot_Q_bars(Q,SNYCQ,SNYCQ_Questions, SNYCQ_Question_type)

plot_W(W)

f = plot_W_scatter(W, plot_hist=True, plot_kde=False, figsize=(5,5), marker_size=20)
f

f.savefig('./figures/bcm_embedding.png')

top_left_scans = W[(W['Factor 1']<0.4) & (W['Factor 2']>0.9)].index
a = SNYCQ.loc[top_left_scans].reset_index(drop=True)
a.index = a.index.astype(str)
a[sorted_q].hvplot.heatmap(width=250, height=250, clim=(0,100), cmap='Viridis', ylabel='Scan', xlabel='Question', fontscale=1.2).opts(colorbar=False, xrotation=90)

top_left_scans = W[(W['Factor 1']>0.75) & (W['Factor 1']<0.94) & (W['Factor 2']<0.2)].index
a = SNYCQ.loc[top_left_scans].reset_index(drop=True)
a.index = a.index.astype(str)
a[sorted_q].hvplot.heatmap(width=250, height=250, clim=(0,100), cmap='Viridis', ylabel='Scan', xlabel='Question', fontscale=1.2).opts(colorbar=False, xrotation=90)

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

clusters_info.value_counts('Cluster Label')

sbjs = clusters_info.index.get_level_values('Subject').unique()
df = pd.DataFrame(index=sbjs, columns=['Num Scans','Large F1','Large F2','Middle'])
for sbj in sbjs:
    aux = clusters_info.loc[sbj,:]
    df.loc[sbj,'Num Scans'] = aux.shape[0]
    aux_counts = aux['Cluster Label'].value_counts()
    for cl in ['Large F1','Large F2','Middle']:
        if cl in aux_counts:
            df.loc[sbj,cl] = aux_counts[cl]
        else:
            df.loc[sbj,cl] = 0

df.head(5)

['%d subjects have %d scans' % (df.set_index('Num Scans').loc[n+1].shape[0], n+1) for n in range(4)]

clustering_summary = pd.Series(dtype=int, name='Clustering Consistency')
clustering_summary['Total in Sample'] = len(sbjs)
df = df[df['Num Scans']>2]
clustering_summary['With multiple scans']              = df.shape[0]
clustering_summary['All scans in single cluster']      = ((df==0).sum(axis=1)==2).sum()
clustering_summary['All but 1 scan in single cluster'] = (df['Num Scans']-df.drop('Num Scans', axis=1).max(axis=1) == 1.0).sum()
clustering_summary['Others'] = (df['Num Scans']-df.drop('Num Scans', axis=1).max(axis=1) > 1.0).sum()
clustering_summary

f = plot_W_scatter(W, clusters_info=clusters_info, plot_kde=False, plot_hist=True, marker_size=20,figsize=(5,5), cluster_palette=[(1.0, 0.4980392156862745, 0.054901960784313725),
                                                                                                                                  (0.12156862745098039, 0.4666666666666667, 0.7058823529411765),(0.17254901960784313, 0.6274509803921569, 0.17254901960784313),])
f

f.savefig('./figures/bcm_embedding_withclusters.png')

a = W.copy()
a = a.reset_index()
a = a.infer_objects()
a['Subject'] = a['Subject'].astype(str)
a['Run'] = a['Run'].astype(str)

a.hvplot.scatter(x='Factor 1', y='Factor 2', hover_cols=['Subject','Run'], aspect='square')

aux = pd.DataFrame(SNYCQ.loc['sub-010211','post-ses-02-run-02-acq-AP'])
aux.columns=['Answers']
a = aux.hvplot.heatmap(width=150, vlim=(0,100), cmap='Viridis').opts(colorbar=False)
(a* hv.Labels(a)).opts(opts.Labels(text_color='white'))

aux = pd.DataFrame(SNYCQ.loc['sub-010096','post-ses-02-run-01-acq-PA'])
aux.columns=['Answers']
a = aux.hvplot.heatmap(width=150, vlim=(0,100), cmap='Viridis').opts(colorbar=False)
(a* hv.Labels(a)).opts(opts.Labels(text_color='white'))

aux = pd.DataFrame(SNYCQ.loc['sub-010141','post-ses-02-run-01-acq-PA'])
aux.columns=['Answers']
a = aux.hvplot.heatmap(width=150, vlim=(0,100), cmap='Viridis').opts(colorbar=False)
(a* hv.Labels(a)).opts(opts.Labels(text_color='white'))

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

plot_W_heatmap(W, clusters_info=clusters_info, scan_order=sort_clf1f2_idx, cmap='Greens')

f=pn.Row(W.loc[sort_clf1f2_idx].reset_index(drop=True).hvplot.heatmap(cmap='Greens', ylabel='Scans', width=250, height=500, fontscale=1.2, clim=(0,1)).opts( colorbar_opts={'title':'W Matrix'}, xrotation=90, toolbar=None))
f.save('./figures/W_sorted.png')
f

plot_P(SNYCQ, question_order=q_order, scan_order=sort_clf1f2_idx, clusters_info=clusters_info)

plot_P(Precon, question_order=q_order, scan_order=sort_clf1f2_idx, clusters_info=clusters_info)

aux = pd.concat([W,clusters_info],axis=1)
aux.hvplot.scatter(x='Factor 1', y='Factor 2', c='Cluster Label', aspect='square', cmap=['lightblue','#ff9999','orange']) + \
aux.hvplot.kde(y='Factor 2', by='Cluster Label', width=500) + aux.hvplot.kde(y='Factor 1', by='Cluster Label', width=500)

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

df.hvplot.kde(y='Vigilance',by='Cluster Label', alpha=.5, title='Vigilance', color=['#4472C4','#ED7D31'], width=500).opts(legend_position='top_left')

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

(df.hvplot.kde(y='Rel. Motion (mean)',by='Cluster Label', alpha=.5, title='Rel. Motion (mean)',color=['#4472C4','#ED7D31'], width=500).opts(legend_position='top_left') + \
df.hvplot.kde(y='Rel. Motion (max)',by='Cluster Label', alpha=.5, title='Rel. Motion (max)',color=['#4472C4','#ED7D31'], width=500).opts(legend_position='top_right')).cols(1)

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


