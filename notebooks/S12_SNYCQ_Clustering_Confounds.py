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
import seaborn as sns
import os.path as osp
import holoviews as hv
from holoviews import opts
from scipy.stats import ttest_ind, mannwhitneyu
import panel as pn
from wordcloud import WordCloud
import hvplot.pandas
from IPython.display import Markdown as md
from IPython import display
from PIL import Image

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

from utils.basics import get_sbj_scan_list, SNYCQ_Questions, SNYCQ_Question_type, SNYCQ_CLUSTERS_INFO_PATH, DATA_DIR, PRJ_DIR
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

W_plot_unsroted = W.reset_index(drop=True).hvplot.heatmap(cmap='Greens', width=200, height=500, fontscale=1.2, clim=(0,1), shared_axes=False).opts( colorbar_opts={'title':'W Matrix'}, xrotation=90, toolbar=None)
C_plot_unsorted = C.reset_index(drop=True).hvplot.heatmap(cmap='Purples', width=250, height=500, fontscale=1.2).opts( colorbar_opts={'title':'C Matrix'}, xrotation=90, toolbar=None)
f = pn.Row(W_plot_unsroted,C_plot_unsorted)
f.save('./figures/W_and_C_unsorted.png')

# Show static version (for Github purposes)

display.Image("./figures/W_and_C_unsorted.png")

# Next, we will plot the Q matrix (with relationships between questions and low dimensional factors) and the Qc matrix (with information about how responses relate to demographics)

Q_plot_unsorted  = Q.hvplot.heatmap( cmap='Oranges', width=240, height=500, clim=(0,100), fontscale=1.2, shared_axes=False).opts( colorbar_opts={'title':'Q Matrix'}, xrotation=90, toolbar=None)
Qc_plot_unsorted = Qc.hvplot.heatmap(cmap='Reds', width=300, height=500, clim=(0,100), fontscale=1.2).opts( colorbar_opts={'title':'Qc Matrix'}, xrotation=90, toolbar=None)
f = pn.Row(Q_plot_unsorted, Qc_plot_unsorted)
f.save('./figures/Q_and_Qc_unsorted_same_scale.png')

display.Image("./figures/Q_and_Qc_unsorted_same_scale.png")

# We also plot Q and Qc after sorting. This helps better understand what the different factors mean in relationship to the original questions

sorted_q = Q.sort_values(by=['Factor 1','Factor 2'],ascending=False).index
Q_plot_sorted  = Q.loc[sorted_q].hvplot.heatmap( cmap='Oranges', width=240, height=500, clim=(0,100), fontscale=1.2, shared_axes=False).opts( colorbar_opts={'title':'Q Matrix'}, xrotation=90, toolbar=None)
Qc_plot_sorted = Qc.loc[sorted_q].hvplot.heatmap(cmap='Reds', width=300, height=500, clim=(0,100), fontscale=1.2).opts( colorbar_opts={'title':'Qc Matrix'}, xrotation=90, toolbar=None)
f = pn.Row(Q_plot_sorted, Qc_plot_sorted)
f.save('./figures/Q_and_Qc_sorted_diff_scale.png')

display.Image("./figures/Q_and_Qc_sorted_diff_scale.png")

# # 4. Create Wordclouds that represent the two factors
#
# Each factor in the low dimensional representation of W is associated with the original questions as described in matrix Q. To help carry these relationships in a graphic way across different figures, we will generate one wordcloud per factor using the infoermation in matrix Q.
#
# 1. We will create a circular mask to be used as the enclosing shape for the wordcloud

x, y = np.ogrid[:300, :300]
circle_mask = (x - 150) ** 2 + (y - 150) ** 2 > 130 ** 2
circle_mask = 255 * circle_mask.astype(int)


# This function will help us create wordcloud where the importance of a term is represented both by the font size and the color. More important words will have darker colors.

def my_tf_color_func(dictionary):
    def my_tf_color_func_inner(word, font_size, position, orientation, random_state=None, **kwargs):
        freq_as_int = int(dictionary[word])
        color_list = sns.color_palette('Oranges',100).as_hex()
        return color_list[freq_as_int]
    return my_tf_color_func_inner


# We also load a basic mask in the form of a sagital brain to use as the confines of the wordcloud

brain_mask = np.array(Image.open(osp.join(PRJ_DIR,'code','fc_introspection','resources','wordclouds','Brain_Sag.png')))
brain_mask = brain_mask > 100
brain_mask = 255 * brain_mask.astype(int)

# 2. We generate the wordcloud for factor 1

wc = WordCloud(max_font_size=50, min_font_size=5, 
               width=200, height=200,contour_color='gray', contour_width=3,
               max_words=6, background_color='white', 
               repeat=True,
               relative_scaling=0.5, mask=brain_mask,
               color_func=my_tf_color_func(Q['Factor 1'].to_dict())).generate_from_frequencies(Q['Factor 1'].to_dict())
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
wc.to_file('./figures/Factor1_WC.png')

wc = WordCloud(max_font_size=50, min_font_size=5, 
               width=200, height=200,contour_color='gray', contour_width=3, colormap='bone',
               max_words=5, background_color='white', mask=brain_mask,
               repeat=True, 
               relative_scaling=0.5,
               color_func=my_tf_color_func(Q['Factor 2'].to_dict())).generate_from_frequencies(Q['Factor 2'].to_dict())
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
wc.to_file('./figures/Factor2_WC.png')

# # 5. Show Original and Reconstructed Matrix
#
# For this, we first recompute the reconstructed matrix by applying the formulation inherent to Sparse Non-Negative Matrix Factorization
#
# $$M_{recon} = [ W, C] \cdot [Q, Q_{c}]^`$$

Mrecon = pd.DataFrame(np.dot(pd.concat([W,C],axis=1).values, pd.concat([Q,Qc],axis=1).values.T),
                      index=SNYCQ.index,columns=SNYCQ.columns)

Morig = pd.DataFrame(M,index=SNYCQ.index,columns=SNYCQ.columns)

f = (Morig.reset_index(drop=True).T.hvplot.heatmap(cmap='viridis', width=1500, height=300, fontscale=1.2, clim=(0,100), shared_axes=False).opts( colorbar_opts={'title':'M Matrix'}, xrotation=90, toolbar=None, xticks=None) + \
Mrecon.reset_index(drop=True).T.hvplot.heatmap(cmap='viridis', width=1500, height=300, fontscale=1.2, clim=(0,100), shared_axes=False).opts( colorbar_opts={'title':'M_recon Matrix'}, xrotation=90, toolbar=None, xticks=None)).cols(1)
pn.Row(f).save('./figures/M_Mrecon.png')

display.Image("./figures/M_Mrecon.png")

question = 'People'
aux = pd.concat([Morig[question],Mrecon[question]],axis=1)
aux.columns = ['Orig','Recon']
aux.hvplot.scatter(x='Orig',y='Recon', aspect='square')

# ***
#
# # 6. Addtional ways to visualize the data
#
# # 6.1. Factor 2 Question relationship as horizontal barplots
#
# Here we represent matrix Q as a series of horizonzal barplots. Questions that are on the top X percentile (dashed line) are marked in red.

plot_Q_bars(Q,SNYCQ,SNYCQ_Questions, SNYCQ_Question_type)

# ## 6.2. W as unsorted heatmap

plot_W(W)

# ## 6.3. W as a scatter plot

f = plot_W_scatter(W, plot_hist=True, plot_kde=True, figsize=(5,5), marker_size=20)
f.savefig('./figures/bcm_embedding.png')

display.Image('./figures/bcm_embedding.png')

# ## 6.4. Plot a few representative subjects on the extremes

top_left_scans = W[(W['Factor 1']<0.4) & (W['Factor 2']>0.9)].index
a = SNYCQ.loc[top_left_scans].reset_index(drop=True)
a.index = a.index.astype(str)
f = a[sorted_q].hvplot.heatmap(width=250, height=250, clim=(0,100), cmap='Viridis', ylabel='Scan', xlabel='Question', fontscale=1.2).opts(colorbar=False, xrotation=90, toolbar=None)
pn.Row(f).save('./figures/bcm_embedding_rep_highF1.png')

display.Image('./figures/bcm_embedding_rep_highF1.png')

top_left_scans = W[(W['Factor 1']>0.75) & (W['Factor 1']<0.94) & (W['Factor 2']<0.2)].index
a = SNYCQ.loc[top_left_scans].reset_index(drop=True)
a.index = a.index.astype(str)
f = a[sorted_q].hvplot.heatmap(width=250, height=250, clim=(0,100), cmap='Viridis', ylabel='Scan', xlabel='Question', fontscale=1.2).opts(colorbar=False, xrotation=90)
pn.Row(f).save('./figures/bcm_embedding_rep_highF2.png')

display.Image('./figures/bcm_embedding_rep_highF2.png')

# ***
# # 7. Clustering: looking for extremes
#
# Even though we do not see clear clusters in the data, we will rely on agglomerative clustering to find scans sitting on the extremes of reported experience. A k=3 should provide the scans on the extremes and then a "separating" middle group.
#
# The inputs to the clustering algorithm are: 1) matrix ```W``` and 2) ```N_CLUSTERS=3```

N_CLUSTERS = 3
cluster_ids = cluster_scans(W, n_clusters=N_CLUSTERS)
cluster_ids = cluster_ids.astype(int)
print('++ INFO: Number of clusters = %d' % int(cluster_ids.max()+1))

# We save the clustering results in a pandas Dataframe with consistent indexing in terms of subject and scan ID

clusters_info = pd.DataFrame(index=W.index, columns=['Cluster ID','Cluster Label'])
clusters_info['Cluster ID'] = cluster_ids

# For creating the figures it is convenient to transform randomly assigned cluster labels into labels that describe the relationship of the clusters to the underlying factors. We will use these three labels: ```Large F1```, ```Large F2``` and ```Middle```. We use the mean F1 and F2 values to translate original numeric labels into these more meaningful categories.

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

# We print the number of scans per group, to ensure the extreme groups have similar sizes.

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

f = plot_W_scatter(W, clusters_info=clusters_info, plot_kde=False, plot_hist=True, marker_size=20,figsize=(5,5), cluster_palette=[(1.0, 0.4980392156862745, 0.054901960784313725),
                                                                                                                                  (0.12156862745098039, 0.4666666666666667, 0.7058823529411765),(0.17254901960784313, 0.6274509803921569, 0.17254901960784313),])
f

f.savefig('./figures/bcm_embedding_withclusters.png')

clusters_info.to_csv(SNYCQ_CLUSTERS_INFO_PATH)
print('++ INFO: Clustering Membership Info saved to [%s]' % SNYCQ_CLUSTERS_INFO_PATH)

# ## 7.1. Plot W sorted by clustering
#
# Now that we have the clusters, we can group scans in M and W using this information to see how the clustering looks in terms of the original and low dimensional represenation of the data as heatmaps.
#
# We will generate two potential clusters depending on whether we sort scans within each cluster using Factor 1 or Factor 2

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

# We will plot W in two ways, horizontal and vertical. One might be more useful for manuscript figures and the otherone for talks, but the information is the same.

plot_W_heatmap(W, clusters_info=clusters_info, scan_order=sort_clf1f2_idx, cmap='Greens')

f=pn.Row(W.loc[sort_clf1f2_idx].reset_index(drop=True).hvplot.heatmap(cmap='Greens', ylabel='Scans', width=250, height=600, fontscale=1.2, clim=(0,1)).opts( colorbar_opts={'title':'W Matrix'}, xrotation=90, toolbar=None))
f.save('./figures/W_sorted.png')

display.Image('./figures/W_sorted.png')

# ## 7.2. Plot M sorted by clusters
#
# Now we will plot the original data. We will sort questions based on the correlation analyses done while originally exploring potential relationships between questions. Sorting does not change results, it just helps with visualization, so that patterns of responses across clusters become more aparent.

q_order = ['Future', 'Specific', 'Past', 'Positive', 'People', 'Images', 'Words', 'Negative', 'Surroundings', 'Myself', 'Intrusive']

f = plot_P(Morig, question_order=q_order, scan_order=sort_clf1f2_idx, clusters_info=clusters_info)
f.savefig('./figures/M_by_clusters.png')

display.Image('./figures/M_by_clusters.png')

f = plot_P(Mrecon, question_order=q_order, scan_order=sort_clf1f2_idx, clusters_info=clusters_info)
f.savefig('./figures/Mrecon_by_clusters.png')

display.Image('./figures/Mrecon_by_clusters.png')

# ***
#
# # 8. Significant Differences in things of interest across extreme clusters
#
# Our next analysis is looking to whether or not there are significant differences in FC across the two extreme groups. Before doing that, we will do a couple of sanity checks to ensure any differences we find we can interpret them with some degree of confidence to being due to the differerences in reported in-scanner experience.
#
# The set of sanity checks include:
#
# 1) Check for significant differences in reported levels of wakefulness
#
# 2) Check for significant differences in head motion in terms of max motion.
#
# 3) Check for significant differneces in head motion in terms of mean motion.
#
# First, we get the list of scans not part of the middle group.

scans_of_interest = clusters_info[clusters_info['Cluster Label']!='Middle'].index

# Second, we create an empty dataset that we will later populate with information about motion and wakefulness

df = pd.DataFrame(index = scans_of_interest,columns=['Cluster Label','Rel. Motion (mean)','Rel. Motion (max)','Vigilance','Age','Gender'])

# ## 8.1. Significant differences in wakefulness levels?
#
# Load again the original SNYCQ data. The reason for this is that earlier on the notebook we removed the answers to the wakefulness question, which is the one we need now.

_,_,aux_snycq = get_sbj_scan_list(when='post_motion')

# Add the wakefulness answer to the temporary dataframe df

for scan in df.index:
    df.loc[scan,'Vigilance'] = aux_snycq.loc[scan,'Vigilance']
    df.loc[scan,'Cluster Label'] = clusters_info.loc[scan,'Cluster Label']
df = df.infer_objects()

f = df.hvplot.kde(y='Vigilance',by='Cluster Label', alpha=.5, title='Vigilance', color=['#4472C4','#ED7D31'], width=500).opts(legend_position='top_left', toolbar=None)
pn.Row(f).save('./figures/bcm_clustering_wakefulness_diffs.png')

display.Image('./figures/bcm_clustering_wakefulness_diffs.png')

ttest_ind(df.set_index('Cluster Label').loc['Large F1','Vigilance'],df.set_index('Cluster Label').loc['Large F2','Vigilance'],alternative='two-sided')

mannwhitneyu(df.set_index('Cluster Label').loc['Large F1','Vigilance'],df.set_index('Cluster Label').loc['Large F2','Vigilance'],alternative='two-sided')

# ## 8.2. Significance differnces in head motion?
#
# We now add the information about motion

for scan in df.index:
    sbj,run = scan
    _,_,_,_,run_num,_,run_acq = run.split('-')
    path = osp.join(DATA_DIR,'PrcsData',sbj,'preprocessed','func','pb01_moco','_scan_id_ses-02_task-rest_acq-{run_acq}_run-{run_num}_bold'.format(run_num=run_num,run_acq=run_acq),'rest_realigned_rel.rms')
    mot  = np.loadtxt(path)
    df.loc[scan,'Rel. Motion (mean)'] = mot.mean()
    df.loc[scan,'Rel. Motion (max)']  = mot.max()
df = df.infer_objects()

f = (df.hvplot.kde(y='Rel. Motion (mean)',by='Cluster Label', alpha=.5, title='Rel. Motion (mean)',color=['#4472C4','#ED7D31'], width=500).opts(legend_position='top_left', toolbar=None) + \
df.hvplot.kde(y='Rel. Motion (max)',by='Cluster Label', alpha=.5, title='Rel. Motion (max)',color=['#4472C4','#ED7D31'], width=500).opts(legend_position='top_right', toolbar=None)).cols(1).opts(toolbar=None)
pn.Row(f).save('./figures/bcm_clustering_motion_diffs.png')

display.Image('./figures/bcm_clustering_motion_diffs.png')

ttest_ind(df.set_index('Cluster Label').loc['Large F1','Rel. Motion (mean)'],df.set_index('Cluster Label').loc['Large F2','Rel. Motion (mean)'],alternative='two-sided')

ttest_ind(df.set_index('Cluster Label').loc['Large F1','Rel. Motion (max)'],df.set_index('Cluster Label').loc['Large F2','Rel. Motion (max)'],alternative='two-sided')


