# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: FC Instrospection py 3.10 | 2023b
#     language: python
#     name: fc_introspection_2023b_py310
# ---

# # Description
#
# This notebook runs additional analyses regarding whether or not thought patterns are systematically similar across scans, using two methods:
#
# * Differntial Identifiability: as described in [The quest for identifiability in human functional connectomes](https://www.nature.com/articles/s41598-018-25089-1) by Amico & Goñi (Scientific Reports, 2018)
# * Identifiability Rate: as described in [Functional connectome fingerprinting: identifying individuals using patterns of brain activity](https://www.nature.com/articles/nn.4135) by Finn et al. (Nat. Neuro, 2015)
#
#

import pandas as pd
import numpy as np
from tqdm import tqdm
from textwrap import wrap
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from utils.io import read_fc_matrices
from utils.basics import DATA_DIR, FB_400ROI_ATLAS_NAME, RESOURCES_SNYCQ_DIR
import os.path as osp
import holoviews as hv
from scipy.stats import pearsonr, spearmanr
from random import shuffle
import warnings
import hvplot.pandas
from matplotlib import colors as mcolors
warnings.simplefilter(action='ignore', category=FutureWarning)

# # 1. Percent of scans in the same clsuter several times
#
# The first way we look at the question of whether introspective reports are trait-like is by looking by how often all scans from the same subject fall in the same scan set (as defined in the previous notebook)
#
# 1. Load SNYCQ items and clustering info

emb_plus  = pd.read_csv(osp.join(RESOURCES_SNYCQ_DIR, 'SNYCQ_tsne_embeddings_plus_scaled.csv'), index_col=[0,1])
scan_list = emb_plus.index.tolist() 
sbj_list  = list(emb_plus.index.get_level_values('Subject').unique())
print('++ [post_introspection_outlier] Number of subjects: %d subjects' % len(scan_list))
print('++ [post_introspection_outlier] Number of scans:    %d scans' % len(sbj_list))
emb_plus.head(2)


# 2. Count how many scans we have per subject. Only scans from subjects that were scanned at least three time will be used in these analyses

# +
N_MIN_SCANS = 3

# Count the number of runs per subject
scan_list_df = pd.DataFrame(scan_list,columns=['Subject','Run'])
run_counts   = scan_list_df.groupby("Subject")["Run"].nunique().to_frame(name="NumRuns")

sbjs_sel_scans = run_counts[run_counts["NumRuns"] >= N_MIN_SCANS].index.tolist()
# -

# 3. Print information regarding how many scans, subjects, etc enter these analyses

Nsbjs_total       = len(emb_plus.index.get_level_values('Subject').unique())
Nsbjs_sel_scans   = len(sbjs_sel_scans)
Nselected_scans   = emb_plus.loc[sbjs_sel_scans].shape[0]
print('++ INFO: Number of subjects in these analyses    : %d subjects' % Nsbjs_total)
print('++ INFO: Number of subjects with %d or more scans : %d subjects' % (N_MIN_SCANS,Nsbjs_sel_scans))
print('++ INFO: Number of scans (for subjects with %d or more scans : %d subjects' % (N_MIN_SCANS,Nselected_scans))


# 4. Count the prevalence of scans in the same set

def count_scans_per_group(sbjs,cluster_info):
    # Extract from cluster_info structure the entries for scans with 2 or more scans
    df = pd.DataFrame(0, index=sbjs, columns=['All scans in same set','All except one scan in same set','Other configuration'], dtype=int)
    aux = cluster_info.loc[sbjs,'Set Label']
    n_scans = aux.shape[0]
    for sbj in aux.index.get_level_values('Subject'):
        auxx = aux.loc[sbj,:]
        auxx_max = auxx.value_counts().max()
        if auxx_max == 4:
            df.loc[sbj,'All scans in same set'] = 1
        elif auxx_max == (auxx.shape[0]-1):
            df.loc[sbj,'All except one scan in same set'] = 1
        else:
            df.loc[sbj,'Other configuration'] = 1
    return df.sum()


# 5. Plot the results
#

# +
import matplotlib.pyplot as plt
import seaborn as sns
from textwrap import wrap

# Example data
final_counts = count_scans_per_group(sbjs_sel_scans, emb_plus)
labels = final_counts.index

# Wrap labels to avoid overlap
labels = ['\n'.join(wrap(l, 20)) for l in labels]

# Define a custom label function
def make_autopct(values):
    def my_autopct(pct):
        total = sum(values)
        val = int(round(pct*total/100.0))
        # Return both label and percentage
        return f"[{val} Participants]\n({pct:.0f}%)"
    return my_autopct

fig, ax = plt.subplots(figsize=(7,7))
colors = sns.color_palette("ch:start=.2,rot=-.3")
wedges, texts, autotexts = ax.pie(
    final_counts,
    labels=None,                      # don't use default labels outside
    colors=colors,
    autopct=make_autopct(final_counts),  # include text inside
    textprops={'color':"black", 'fontsize':10, 'weight':'bold'}
)

# Manually add labels to the center of each wedge
for i, a in enumerate(autotexts):
    a.set_text(f"{labels[i]}\n{a.get_text()}")  # add label + percentage

plt.tight_layout()
plt.show()

# -

# # 2. Differential Identifibiability
#
# Here, we compute Idiff (Differential Identifiability) following the procedure described by [Amico et al. (2018)](https://www.nature.com/articles/s41598-018-25089-1)
#
# 1. Identify scans from subjects that were scanned at least twice.

# Define custom colors for True and False
cmap_colors = ['white', 'blue']  # False will be white, True will be blue
# Create a custom colormap
custom_cmap = mcolors.LinearSegmentedColormap.from_list('boolean_cmap', cmap_colors, N=2)

# Identify the runs with less than 2 scans
subjects_lt2 = run_counts[run_counts["NumRuns"] < 2].index.tolist()
# Create mask with True for scans we want to keep and False for those to remove
mask = emb_plus.index.get_level_values("Subject").isin(subjects_lt2)

# These scans / subjects will not enter the identifiability analyses
scan_list_df[mask]

# 2. Create an index with only the scans from subjects with at least two visits

# Remove from the introspection dataframe the entries for scans that are single entries per subject
scans_for_identifiability = emb_plus.copy().loc[~mask].index
#scans_by_cluster          = emb_plus.copy().loc[~mask].sort_values(by=['Set Label']).index
print('++ Number of scans used in identifiability analyses: %d scans' % len(scans_for_identifiability))

# 3. Create a list of the subjects that enter the anlyses

# Number of subjects entering the analyses
sbjs_for_identifiability = scans_for_identifiability.get_level_values('Subject').unique().tolist()
print('++ Number of subjects used in identifiability analyses: %d subjects' % len(sbjs_for_identifiability))

# ## 2.1. Differential Identifiability for Introspection data
#
# 1. Create a new dataframe that only contains the SNYCQ items of interest (all but vigilance) for the subjects and scans with 2+ visits.
#
#

# Remove the Vigilance entry to match all other analyses
SNYCQ_for_identifiability = emb_plus.loc[scans_for_identifiability].drop(['TSNE1','TSNE2','TSNE3','Group Probability','Set Label'],axis=1)
print(SNYCQ_for_identifiability.shape)


# 2. Estimate the Identifiability Matrix (correlation of each scan to every other scan) --> This is at the scan level.

def get_scan_level_identifiability_matrix(dataframe):
    Id_matrix = dataframe.T.corr(method='pearson').values
    np.fill_diagonal(Id_matrix,np.nan)
    Id_matrix = pd.DataFrame(Id_matrix,index=dataframe.index,columns=dataframe.index)
    return Id_matrix


Id_SNYCQ_scans = get_scan_level_identifiability_matrix(SNYCQ_for_identifiability)
Id_SNYCQ_scans.head(3)


# + vscode={"languageId": "raw"} active=""
# # Compute Identifiability matrix at the scan level
# Id_SNYCQ_scans2 = SNYCQ_for_identifiability.T.corr(method='pearson').values
# # Set diagonal to NaN
# np.fill_diagonal(Id_SNYCQ_scans2,np.nan)
# # Make it a dataframe again
# Id_SNYCQ_scans2 = pd.DataFrame(Id_SNYCQ_scans2,index=scans_for_identifiability,columns=scans_for_identifiability)
# -

# 3. Use the scan-level Identifiability matrix to compute the subject-level identifiability matrix

def get_subject_level_identifiability_matrix(scan_level_Id_matrix):
    """
    Computes the subject-level identifiability matrix from the scan-level identifiability matrix.
    Parameters:
    scan_level_Id_matrix (pd.DataFrame): DataFrame where both rows and columns are scan IDs, and values are identifiability scores.
    Returns:
    pd.DataFrame: Subject-level identifiability matrix.
    """
    subjects_list = scan_level_Id_matrix.index.get_level_values('Subject').unique().tolist()  
    Id_subjects = pd.DataFrame(np.nan,index=subjects_list,columns=subjects_list, dtype=np.float32)
    for sbj_i in tqdm(subjects_list):
        for sbj_j in subjects_list:
            aux = scan_level_Id_matrix.loc[sbj_i][sbj_j].values
            if sbj_i == sbj_j:
                new_value = aux[np.triu_indices_from(aux, k=1)].mean()
                if pd.isna(Id_subjects.loc[sbj_i, sbj_j]):
                    Id_subjects.loc[sbj_i, sbj_j] = new_value
                else:
                    assert Id_subjects.loc[sbj_i, sbj_j] == new_value, "Inconsistent value found!"
            else:
                Id_subjects.loc[sbj_i, sbj_j] = aux.mean().mean() 
    return Id_subjects


Id_SNYCQ_sbjs = get_subject_level_identifiability_matrix(Id_SNYCQ_scans)

# + vscode={"languageId": "raw"} active=""
# Id_SNYCQ_sbjs = pd.DataFrame(np.nan,index=sbjs_for_identifiability,columns=sbjs_for_identifiability, dtype=np.float32)
# for sbj_i in sbjs_for_identifiability:
#     for sbj_j in sbjs_for_identifiability:
#         aux = Id_SNYCQ_scans.loc[sbj_i][sbj_j].values
#         if sbj_i == sbj_j:
#             new_value = aux[np.triu_indices_from(aux, k=1)].mean()
#             if pd.isna(Id_SNYCQ_sbjs.loc[sbj_i, sbj_j]):
#                 Id_SNYCQ_sbjs.loc[sbj_i, sbj_j] = new_value
#             else:
#                 assert Id_SNYCQ_sbjs.loc[sbj_i, sbj_j] == new_value, "Inconsistent value found!"
#         else:
#             Id_SNYCQ_sbjs.loc[sbj_i, sbj_j] = aux.mean().mean() 
#
# -

# 4. Compute ISelf, IOther and IDiff

Iself  = np.diag(Id_SNYCQ_sbjs.values).mean()
Iother = np.concatenate([Id_SNYCQ_sbjs.values[np.triu_indices_from(Id_SNYCQ_sbjs.values, k=1)],Id_SNYCQ_sbjs.values[np.tril_indices_from(Id_SNYCQ_sbjs.values, k=1)]]).mean()
Idiff   = 100 * (Iself - Iother)
print('++ Introspection ISelf = %.2f' %Iself)
print('++ Introspection IOther = %.2f' %Iother)
print ('++ Identifiability based on Introspection data = %.1f %%' % Idiff)


# 5. Plot the Identifiability Matrix at the subject level

# +
fig, ax = plt.subplots(figsize=(5,5))
hm = sns.heatmap(
    Id_SNYCQ_sbjs.values,
    vmin=0.1,
    vmax=0.7,
    cmap='cividis',
    square=True,
    ax=ax,
    cbar=False   # turn off automatic colorbar
)

# Add colorbar manually
cbar = ax.figure.colorbar(hm.get_children()[0], ax=ax, fraction=0.046, pad=0.04)
cbar.ax.tick_params(labelsize=10)
cbar.set_label('Pearson Correlation', fontsize=12)

# Labels and ticks
ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=10)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=10)
ax.set_xlabel('Subjects')
ax.set_ylabel('Subjects')

plt.tight_layout()
plt.show()


# -

# 6. Create histograms of values contributing to Iself and Iother

# +
# Create mask matrix that only contains 1 for scans that are from the same subject
row_subj = Id_SNYCQ_scans.index.get_level_values("Subject").to_numpy()
col_subj = Id_SNYCQ_scans.columns.get_level_values("Subject").to_numpy()

within_sbj_mask = pd.DataFrame(
                    (row_subj[:, None] == col_subj[None, :]).astype(bool),
                     index=Id_SNYCQ_scans.index,
                     columns=Id_SNYCQ_scans.columns)
across_sbj_mask = ~within_sbj_mask
within_sbj_mask.values[range(len(within_sbj_mask)),range(len(within_sbj_mask))] = False 
# -

Iself_values_SNYCQ = Id_SNYCQ_scans[within_sbj_mask].values.flatten()
Iself_values_SNYCQ = Iself_values_SNYCQ[~np.isnan(Iself_values_SNYCQ)]
Iself_SNYCQ        = Iself_values_SNYCQ.mean()
print('++ Introspection Iself = %.5f ' % Iself_SNYCQ)

Iother_values_SNYCQ = Id_SNYCQ_scans[across_sbj_mask].values.flatten()
Iother_values_SNYCQ = Iother_values_SNYCQ[~np.isnan(Iother_values_SNYCQ)]
Iother_SNYCQ        = Iother_values_SNYCQ.mean() 
print('++ Introspection Iother = %.5f ' % Iother_SNYCQ)

# +
fig, ax = plt.subplots(figsize=(3,5))

sns.kdeplot(Iself_values_SNYCQ, label=r'$I_{Self}$', alpha=0.5, color='blue', fill=True, ax=ax)
sns.kdeplot(Iother_values_SNYCQ, label=r'$I_{Other}$', alpha=0.5, color='red', fill=True, ax=ax)

# Move legend inside the plot, bottom-left
ax.legend(
    loc='lower left',
    bbox_to_anchor=(0.05, 0.80),
    frameon=True,
    framealpha=0.8,
    fontsize=12,
)

# Set tick label size
ax.tick_params(axis='both', labelsize=12)

# Set axis labels (if you have them)
ax.set_ylabel('Density', fontsize=12)

# Set axis limits
ax.set_xlim(-1, 1)

plt.tight_layout()
plt.show()

# -

# ## 2.2. Differential Identifiability based on FC
#
# 1. Load Pre-processed FC matrices for scans in these analyses

ATLAS_NAME           = FB_400ROI_ATLAS_NAME

FC_for_identifiability = read_fc_matrices(scans_for_identifiability,DATA_DIR,ATLAS_NAME,'pb06_staticFC')

# 2. Compute the scan-level identifiability matrix

Id_FC_scans = get_scan_level_identifiability_matrix(FC_for_identifiability)
Id_FC_scans.head(3)

# 3. Compute the subject-level identifiability matrix

Id_FC_sbjs = get_subject_level_identifiability_matrix(Id_FC_scans)

# 4. Compute ISelf, IOther and IDiff

Iself  = np.diag(Id_FC_sbjs.values).mean()
Iother = np.concatenate([Id_FC_sbjs.values[np.triu_indices_from(Id_FC_sbjs.values, k=1)],Id_FC_sbjs.values[np.tril_indices_from(Id_FC_sbjs.values, k=1)]]).mean()
Idiff   = 100 * (Iself - Iother)
print('++ Introspection ISelf = %.2f' %Iself)
print('++ Introspection IOther = %.2f' %Iother)
print ('++ Identifiability based on Introspection data = %.1f %%' % Idiff)


# 5. Plot the Identifiability Matrix at the subject level

# +
fig, ax = plt.subplots(figsize=(5,5))
hm = sns.heatmap(
    Id_FC_sbjs.values,
    vmin=0.1,
    vmax=0.7,
    cmap='cividis',
    square=True,
    ax=ax,
    cbar=False   # turn off automatic colorbar
)

# Add colorbar manually
cbar = ax.figure.colorbar(hm.get_children()[0], ax=ax, fraction=0.046, pad=0.04)
cbar.ax.tick_params(labelsize=10)
cbar.set_label('Pearson Correlation', fontsize=12)

# Labels and ticks
ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=10)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=10)
ax.set_xlabel('Subjects')
ax.set_ylabel('Subjects')

plt.tight_layout()
plt.show()


# -

Iself_values_FC = Id_FC_scans[within_sbj_mask].values.flatten()
Iself_values_FC = Iself_values_FC[~np.isnan(Iself_values_FC)]
Iself_FC        = Iself_values_FC.mean()
print('++ Introspection Iself = %.5f ' % Iself_FC)

Iother_values_FC = Id_FC_scans[across_sbj_mask].values.flatten()
Iother_values_FC = Iother_values_FC[~np.isnan(Iother_values_FC)]
Iother_FC        = Iother_values_FC.mean() 
print('++ Introspection Iother = %.5f ' % Iother_FC)

# +
fig, ax = plt.subplots(figsize=(3,5))

sns.kdeplot(Iself_values_FC, label=r'$I_{Self}$', alpha=0.5, color='blue', fill=True, ax=ax)
sns.kdeplot(Iother_values_FC, label=r'$I_{Other}$', alpha=0.5, color='red', fill=True, ax=ax)

# Move legend inside the plot, bottom-left
ax.legend(
    loc='lower left',
    bbox_to_anchor=(0.05, 0.80),
    frameon=True,
    framealpha=0.8,
    fontsize=12,
)

# Set tick label size
ax.tick_params(axis='both', labelsize=12)

# Set axis labels (if you have them)
ax.set_ylabel('Density', fontsize=12)

# Set axis limits
ax.set_xlim(-1, 1)

plt.tight_layout()
plt.show()


# -

# ***
#
# # 3. Identifiability Rate
#
# ## 3.1 Indentifibility Rate based on introspection data
#
# 1. Compute the experimental Identification Rate

def get_id_rate(id_matrix_sbj):
    n_scans          = id_matrix_sbj.shape[0]
    correct_id_count = 0    
    for r,row in id_matrix_sbj.iterrows():
        target_subject       = r[0]
        most_similar_subject = row.sort_values(ascending=False).index[0][0]
        if target_subject == most_similar_subject:
            correct_id_count += 1
    id_rate = 100 * correct_id_count / n_scans
    return id_rate


Id_Rate_SNYCQ = get_id_rate(Id_SNYCQ_scans)
print('++ Introspection-based Identification Rate = %.2f %%' % (identification_rate_SNYCQ))

# 2. Do Permutation analysis

# +
N_NULL_PERMS = 10000
SNYCQ_id_rate_null = pd.DataFrame(0,columns=['ID_Rate'],index=range(N_NULL_PERMS), dtype=float)
SNYCQ_id_rate_null.index.name = 'Permutation'

for i in tqdm(range(N_NULL_PERMS)):
    Id_matrix_shuffled = Id_SNYCQ_scans.copy()
    cols = Id_matrix_shuffled.columns.to_list()
    shuffle(cols)
    Id_matrix_shuffled.columns = pd.MultiIndex.from_tuples(cols, names=['Subject','Run'])
    this_perm_id_rate = get_id_rate(Id_matrix_shuffled)
    SNYCQ_id_rate_null.loc[i,'ID_Rate'] = this_perm_id_rate
# -

fig,ax = plt.subplots(1,1,figsize=(4,5))
plot = sns.kdeplot(data=SNYCQ_id_rate_null,x='ID_Rate',fill=True,label='Null Distribution',ax=ax, color='gray')
plot.set_xlabel("Identification Rate", fontsize=14)
plot.set_ylabel("Density", fontsize=14)
# Add the observed value line with label
plot.axvline(identification_rate_SNYCQ, color='black', linestyle='--', label='Observed value', linewidth=3)
# Force legend to show both entries
plot.set_xlim(0,100)
plot.legend(loc='upper right', fontsize=14)
ax.set_xticklabels(ax.get_xticklabels(), fontsize=14);
ax.set_yticklabels(ax.get_yticklabels(), fontsize=14);

# ## 3.2. Identification Rate based on FC data
#
# 1. Get expiremantal identification rate

Id_Rate_FC = get_id_rate(Id_FC_scans)
print('++ FC-based Identification Rate = %.2f %%' % (Id_Rate_FC))

# 2. Bootstraping analysis

# +
N_NULL_PERMS = 10000
FC_id_rate_null = pd.DataFrame(0,columns=['ID_Rate'],index=range(N_NULL_PERMS), dtype=float)
FC_id_rate_null.index.name = 'Permutation'

for i in tqdm(range(N_NULL_PERMS)):
    Id_matrix_shuffled = Id_FC_scans.copy()
    cols = Id_matrix_shuffled.columns.to_list()
    shuffle(cols)
    Id_matrix_shuffled.columns = pd.MultiIndex.from_tuples(cols, names=['Subject','Run'])
    this_perm_id_rate = get_id_rate(Id_matrix_shuffled)
    FC_id_rate_null.loc[i,'ID_Rate'] = this_perm_id_rate
# -

fig,ax = plt.subplots(1,1,figsize=(4,5))
plot = sns.kdeplot(data=FC_id_rate_null,x='ID_Rate',fill=True,label='Null Distribution',ax=ax, color='gray')
plot.set_xlabel("Identification Rate", fontsize=14)
plot.set_ylabel("Density", fontsize=14)
# Add the observed value line with label
plot.axvline(Id_Rate_FC, color='black', linestyle='--', label='Observed value', linewidth=3)
# Force legend to show both entries
plot.set_xlim(0,100)
plot.legend(loc='upper right', fontsize=14)
ax.set_xticklabels(ax.get_xticklabels(), fontsize=14);
ax.set_yticklabels(ax.get_yticklabels(), fontsize=14);
