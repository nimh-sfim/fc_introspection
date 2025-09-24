# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Generic Kernel (2025a)
#     language: python
#     name: generic_2025a
# ---

# # Description
#
# This notebook runs additional analyses regarding whether or not thought patterns are systematically similar across scans, using two methods:
#
# * Differntial Identifiability: as described in [The quest for identifiability in human functional connectomes](https://www.nature.com/articles/s41598-018-25089-1) by Amico & Goñi (Scientific Reports, 2018)
# * Identifiability Rate: as described in [Functional connectome fingerprinting: identifying individuals using patterns of brain activity](https://www.nature.com/articles/nn.4135) by Finn et al. (Nat. Neuro, 2015)
#
#

# +
import pandas as pd
import numpy as np
import hvplot.pandas
from utils.basics import get_sbj_scan_list
from nilearn.connectome import sym_matrix_to_vec
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from utils.io import read_fc_matrices
from utils.basics import DATA_DIR, FB_400ROI_ATLAS_NAME
# -

NUM_MIN_SCANS = 4

sbj_list, scan_list, snycq_df = get_sbj_scan_list(when='post_motion', return_snycq=True)
sbj_list = list(sbj_list)
print(snycq_df.shape)

# +
# Define custom colors for True and False
cmap_colors = ['white', 'blue']  # False will be white, True will be blue

# Create a custom colormap
custom_cmap = colors.LinearSegmentedColormap.from_list('boolean_cmap', cmap_colors, N=2)
# -

# # 1. Load list of scans and select those amenable for identifiability analyses

# Count the number of runs per subject
scan_list_df = pd.DataFrame(scan_list,columns=['Subject','Run'])
run_counts   = scan_list_df.groupby("Subject")["Run"].nunique().to_frame(name="NumRuns")
# Identify the runs with less than 2 scans
subjects_lt2 = run_counts[run_counts["NumRuns"] < NUM_MIN_SCANS].index.tolist()
# Remove these entries from the snyc_q dataframe
mask = snycq_df.index.get_level_values("Subject").isin(subjects_lt2)

# mask is an array that has true for scans that are from subjects where more than one scan is available
# and false otherwise
mask[0:10], print(mask.shape)

# These scans / subjects will not enter the identifiability analyses
scan_list_df[mask]

# # 2. Differential Identifiability
#
# ## 2.1. Differential Identifiability for Introspection data

# Remove from the introspection dataframe the entries for scans that are single entries per subject
snycq_df = snycq_df.loc[~mask]
print(snycq_df.shape)

# Number of subjects entering the analyses
snycq_df.index.get_level_values('Subject').unique().shape

# Remove the Vigilance entry to match all other analyses
snycq_df = snycq_df.drop('Vigilance',axis=1)

# Compute the identifibility matrix
experience_A  = snycq_df.T.corr(method='pearson')

g = sns.heatmap(experience_A.reset_index(drop=True).T.reset_index(drop=True).T, cmap='inferno', vmin=-.1, vmax=1, square=True)
g.set_xlabel('Scans');
g.set_ylabel('Scans');
g.set_title('Identifiability Matrix (Instropection data)');

experience_A.reset_index(drop=True).T.reset_index(drop=True).T.hvplot.heatmap(aspect='square',cmap='inferno', frame_width=1000)

# +
# Create mask matrix that only contains 1 for scans that are from the same subject
row_subj = experience_A.index.get_level_values("Subject").to_numpy()
col_subj = experience_A.columns.get_level_values("Subject").to_numpy()

within_sbj_mask = pd.DataFrame(
                    (row_subj[:, None] == col_subj[None, :]).astype(bool),
                     index=experience_A.index,
                     columns=experience_A.columns)
across_sbj_mask = ~within_sbj_mask
within_sbj_mask.values[range(len(within_sbj_mask)),range(len(within_sbj_mask))] = False 
# -

fig, axs = plt.subplots(1,2,figsize=(13,5))
g_w      = sns.heatmap(within_sbj_mask.reset_index(drop=True).T.reset_index(drop=True).T, ax=axs[0], square=True, cmap=custom_cmap,cbar=False )
g_a      = sns.heatmap(across_sbj_mask.reset_index(drop=True).T.reset_index(drop=True).T, ax=axs[1], square=True, cmap=custom_cmap,cbar_kws={"ticks": [0.25, .75]})
colorbar = axs[1].collections[0].colorbar
colorbar.set_ticklabels(['False', 'True'])
g_a.set_xlabel('Scans'); g_a.set_ylabel('Scans');g_w.set_xlabel('Scans'); g_w.set_ylabel('Scans');
g_w.set_title('Within Subject mask');g_a.set_title('Across Subject mask')

within_sbj_mask.reset_index(drop=True).T.reset_index(drop=True).T.hvplot.heatmap(aspect='square') + across_sbj_mask.reset_index(drop=True).T.reset_index(drop=True).T.hvplot.heatmap(aspect='square')

Iself_values = experience_A[within_sbj_mask].values.flatten()
Iself_values = Iself_values[~np.isnan(Iself_values)]
Iself        = Iself_values.mean()
print('++ Introspection Iself = %.2f ' % Iself)

Iother_values = experience_A[across_sbj_mask].values.flatten()
Iother_values = Iother_values[~np.isnan(Iother_values)]
Iother        = Iother_values.mean() 
print('++ Introspection Iother = %.2f ' % Iother)

Idiff   = 100 * (Iself - Iother)
print ('++ Identifiability based on Introspection data = %.2f %%' % Idiff)

# ## 2.2. Differential Identifiability based on FC

ATLAS_NAME           = FB_400ROI_ATLAS_NAME

fc_data = read_fc_matrices(scan_list,DATA_DIR,ATLAS_NAME,'pb06_staticFC')

print('++ Shape of FC dataframe with all data %s' % str(fc_data.shape))
fc_data = fc_data.loc[~mask]
print('++ Shape of FC dataframe after keeping only scans with at least %d scans -->  %s' % (NUM_MIN_SCANS,str(fc_data.shape)))

# %%time
fc_A = fc_data.T.corr(method='pearson')
print(fc_A.shape)

experience_A.reset_index(drop=True).T.reset_index(drop=True).T.hvplot.heatmap(aspect='square',cmap='inferno', frame_width=600) + fc_A.reset_index(drop=True).T.reset_index(drop=True).T.hvplot.heatmap(aspect='square',cmap='inferno', frame_width=600)

experience_A.reset_index(drop=True).T.reset_index(drop=True).T.hvplot.heatmap(aspect='square',cmap='RdBu_r', frame_width=600, clim=(-1,1)) + fc_A.reset_index(drop=True).T.reset_index(drop=True).T.hvplot.heatmap(aspect='square',cmap='RdBu_r', frame_width=600, clim=(-1,1))

from scipy.stats import pearsonr,spearmanr,zscore
print(pearsonr(sym_matrix_to_vec(experience_A.values,discard_diagonal=True),sym_matrix_to_vec(fc_A.values,discard_diagonal=True)))
print(spearmanr(sym_matrix_to_vec(experience_A.values,discard_diagonal=True),sym_matrix_to_vec(fc_A.values,discard_diagonal=True)))

fc_Iself_values = fc_A[within_sbj_mask].values.flatten()
fc_Iself_values = fc_Iself_values[~np.isnan(fc_Iself_values)]
fc_Iself        = fc_Iself_values.mean()
print('++ Introspection Iself = %.2f ' % fc_Iself)

fc_Iother_values = fc_A[across_sbj_mask].values.flatten()
fc_Iother_values = fc_Iother_values[~np.isnan(fc_Iother_values)]
fc_Iother        = fc_Iother_values.mean() 
print('++ Introspection Iother = %.2f ' % fc_Iother)

fc_Idiff   = 100 * (fc_Iself - fc_Iother)
print ('++ Identifiability based on Introspection data = %.2f %%' % fc_Idiff)

# # 3. Identifiability Rate
#
# ## 3.1 Indentifibility Rate based on introspection data

# +
avail_scans = list(set([s[1] for s in scan_list]))
avail_sbjs  = list(experience_A.index.get_level_values('Subject').unique())

exp_id_rate = pd.DataFrame(columns=['ID_rate'],index=avail_scans)
exp_id_rate.index.name = 'Target Run'

for target_run in tqdm(avail_scans):
    this_experience_A = experience_A.copy()
    this_experience_A = this_experience_A.loc[this_experience_A.columns.get_level_values("Run") == target_run, this_experience_A.columns.get_level_values("Run") != target_run]
    exp_best_match        = this_experience_A.idxmax(axis=1)
    exp_correct           = np.array([t_sbj==w_sbj for ((t_sbj,t_run),(w_sbj,w_run)) in exp_best_match.items()])
    exp_id_rate.loc[target_run,'ID_rate'] = (100*exp_correct.sum()/len(avail_sbjs))
# -

exp_id_rate.round().mean()

# ## 3.2. Identifiability rate based on FC data

# +
avail_scans = list(set([s[1] for s in scan_list]))
avail_sbjs  = list(experience_A.index.get_level_values('Subject').unique())

fc_id_rate = pd.DataFrame(columns=['ID_rate'],index=avail_scans)
fc_id_rate.index.name = 'Target Run'

for target_run in tqdm(avail_scans):
    this_fc_A = fc_A.copy()
    this_fc_A = this_fc_A.loc[this_fc_A.columns.get_level_values("Run") == target_run, this_fc_A.columns.get_level_values("Run") != target_run]
    fc_best_match        = this_fc_A.idxmax(axis=1)
    fc_correct           = np.array([t_sbj==w_sbj for ((t_sbj,t_run),(w_sbj,w_run)) in fc_best_match.items()])
    fc_id_rate.loc[target_run,'ID_rate'] = (100*fc_correct.sum()/len(avail_sbjs))
# -

fc_id_rate.round().mean()

fc_id_rate.round()









































df1 = snycq_df.loc[(snycq_df.index.get_level_values('Subject').isin(sbjs_in_both)) & (snycq_df.index.get_level_values('Run').isin([run1_id]))].reset_index().set_index('Subject').drop('Run',axis=1)
df2 = snycq_df.loc[(snycq_df.index.get_level_values('Subject').isin(sbjs_in_both)) & (snycq_df.index.get_level_values('Run').isin([run2_id]))].reset_index().set_index('Subject').drop('Run',axis=1)

corr_mat = pd.DataFrame( index=df1.index, columns=df2.index, dtype=float)
for sbj_target in df1.index:
    for sbj_db in df2.index:
        corr_mat.loc[sbj_target,sbj_db] = df1.loc[sbj_target].corr(df2.loc[sbj_db])

# 2. For each subject in df1, find the subject in df2 with the highest correlation
best_match = corr_mat.idxmax(axis=1)
best_corr  = corr_mat.max(axis=1)
got_it_right = pd.Series(best_match.index == best_match.values, index=best_match.index)

result = pd.DataFrame({
    "BestMatch": best_match,
    "Correlation": best_corr,
    "Correct": got_it_right
})

result.sum()

# +
result['Top3'] = False
result['Top5'] = False
result['Top10'] = False

for sbj in sbjs_in_both:
    result.loc[sbj,'Top3'] = sbj in corr_mat.loc[sbj].sort_values(ascending=False).iloc[0:3].index
    result.loc[sbj,'Top5'] = sbj in corr_mat.loc[sbj].sort_values(ascending=False).iloc[0:5].index
    result.loc[sbj,'Top10'] = sbj in corr_mat.loc[sbj].sort_values(ascending=False).iloc[0:10].index
# -

# # sbjs_in_both

result.sum()

# ## 3.2. Identifiability Rate based on FC data

run1_id,run2_id = 'post-ses-02-run-01-acq-PA', 'post-ses-02-run-02-acq-PA'
fc_sbjs_with_scan_sess1 = df.set_index('Run').loc[run1_id]['Subject'].unique()
fc_sbjs_with_scan_sess2 = df.set_index('Run').loc[run2_id]['Subject'].unique()

fc_sbjs_in_both = list(set(fc_sbjs_with_scan_sess1).intersection(fc_sbjs_with_scan_sess2))
print(len(fc_sbjs_in_both))

fc_df1 = fc_data.loc[(fc_data.index.get_level_values('Subject').isin(fc_sbjs_in_both)) & (fc_data.index.get_level_values('Run').isin([run1_id]))].reset_index().set_index('Subject').drop('Run',axis=1)
fc_df2 = fc_data.loc[(fc_data.index.get_level_values('Subject').isin(fc_sbjs_in_both)) & (fc_data.index.get_level_values('Run').isin([run2_id]))].reset_index().set_index('Subject').drop('Run',axis=1)

# %%time
fc_corr_mat = pd.DataFrame( index=fc_df1.index, columns=fc_df2.index, dtype=float)
for sbj_target in tqdm(fc_df1.index):
    for sbj_db in fc_df2.index:
        fc_corr_mat.loc[sbj_target,sbj_db] = fc_df1.loc[sbj_target].corr(fc_df2.loc[sbj_db])

# 2. For each subject in df1, find the subject in df2 with the highest correlation
fc_best_match = fc_corr_mat.idxmax(axis=1)
fc_best_corr  = fc_corr_mat.max(axis=1)
fc_got_it_right = pd.Series(fc_best_match.index == fc_best_match.values, index=fc_best_match.index)

fc_result = pd.DataFrame({
    "BestMatch": fc_best_match,
    "Correlation": fc_best_corr,
    "Correct": fc_got_it_right
})

# +
fc_result['Top3'] = False
fc_result['Top5'] = False
fc_result['Top10'] = False

for sbj in sbjs_in_both:
    fc_result.loc[sbj,'Top3'] = sbj in fc_corr_mat.loc[sbj].sort_values(ascending=False).iloc[0:3].index
    fc_result.loc[sbj,'Top5'] = sbj in fc_corr_mat.loc[sbj].sort_values(ascending=False).iloc[0:5].index
    fc_result.loc[sbj,'Top10'] = sbj in fc_corr_mat.loc[sbj].sort_values(ascending=False).iloc[0:10].index
# -

fc_result.sum()

print('done')

# # 2. SNYCQ - Identifiability Rate


