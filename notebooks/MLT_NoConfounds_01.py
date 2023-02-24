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

import numpy as np
import pandas as pd
import hvplot.pandas
from utils.basics import get_sbj_scan_list
from utils.SNYCQ_NMF_Extra import cluster_scans, plot_W_scatter, plot_P
q_order = ['Future', 'Specific', 'Past', 'Positive', 'People', 'Images', 'Words', 'Negative', 'Surroundings', 'Myself', 'Intrusive']
import panel as pn

# This notebook looks at the scenario where no confounds are being modeled during the NMF of the experiential data.
#
# MLT tells me that the optimal set of parameters are:
#
# * Number of Dimensions: $D = 2$ 
#
# * Regularization for the W matrix: $\beta_{W} = 0$ 
#
# * Regularization for the Q matrix: $\beta_{Q} = 0.1 $

# 1. Load scan and subject list

sbj_list, scan_list, snycq = get_sbj_scan_list(when='post_motion')

snycq = snycq.drop('Vigilance', axis=1)

# 2. Load downloaded demographics information (per subject)

path = '../../../data/snycq/participants_post_motion_QA.csv'
demographics_per_subject = pd.read_csv(path, index_col=['Subject'])

# 3. Populate a new dataframe with demographics information on a scan-by-scan basis

demographics_per_scan = pd.DataFrame(index=pd.MultiIndex.from_tuples(scan_list, names=['Subject','Run']), columns=demographics_per_subject.columns)

for sbj in demographics_per_subject.index:
    demographics_per_scan.loc[sbj,'gender']            = demographics_per_subject.loc[sbj,'gender']
    demographics_per_scan.loc[sbj,'age (5-year bins)'] = demographics_per_subject.loc[sbj,'age (5-year bins)']

demographics_per_scan['age (5-year bins)'].value_counts()

demographics_per_scan['gender'].value_counts()

# ***
# ## Load results from Ka-Chun

path = '../resources/mtl_snycq/factorization_fulldata_no_confounds.npz'
data = np.load(path)

# The original factorization (no confounds approach) was: $$M = W \cdot Q'$$

assert np.equal(snycq.values,data['M']).sum() == snycq.shape[0]* snycq.shape[1],"Shape of data disagrees across datasets"
assert list(data['subjlist']) == list(snycq.index.get_level_values('Subject')),"Subject order disagrees across datasets"
assert list(data['runlist']) == list(snycq.index.get_level_values('Run')), "Scan order disagrees across datasets"

W = pd.DataFrame(data['W'], columns=['Factor 1','Factor 2'], index=demographics_per_scan.index)
print(W.shape)

Q = pd.DataFrame(data['Q'], columns=['Factor 1','Factor 2'], index=snycq.columns)
print(Q.shape)

M = pd.DataFrame(data['M'],index=snycq.index,columns=snycq.columns)
print(M.shape)

W.hvplot.scatter(x='Factor 1', y='Factor 2', aspect='square')

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

# ***

Mrecon = pd.DataFrame(np.dot(W,Q.T), index=M.index, columns=M.columns)

plot_P(M, question_order=q_order, scan_order=sort_clf1f2_idx, clusters_info=clusters_info)

plot_P(Mrecon, question_order=q_order, scan_order=sort_clf1f2_idx, clusters_info=clusters_info)

out_path='../resources/mtl_snycq/no_confounds/SNYCQ_clusters_info.csv'
clusters_info.to_csv(out_path)
print('++ INFO: Clustering Membership Info saved to [%s]' % out_path)


