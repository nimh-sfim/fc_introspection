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
import os
import pandas as pd
import numpy as np
from shutil import rmtree
from utils.basics import RESOURCES_CPM_DIR, DATA_DIR, FB_ATLAS_NAME, CORTICAL_ATLAS_NAME, ATLASES_DIR
from utils.basics import get_sbj_scan_list
from cpm.cpm import read_fc_matrices
from sfim_lib.io.afni import load_netcc
from tqdm import tqdm
#from cpm.cpm import mk_kfold_indices, split_train_test, get_train_test_data
#from cpm.cpm import select_features, build_model, apply_model
from cpm.cpm import cpm_wrapper
from cpm.plotting import plot_predictions

ATLAS_NAME = FB_ATLAS_NAME

# # 1. Prepare Data in Disk
# 1. Create resources folder for CPM analyses

if not osp.exists(RESOURCES_CPM_DIR):
    print('++ INFO: Creating resources folder for CPM analyses [%s]' % RESOURCES_CPM_DIR)
    os.makedirs(RESOURCES_CPM_DIR)

# 2. Load list of scans that passed all QAs

sbj_list, scan_list, snycq_df = get_sbj_scan_list(when='post_motion', return_snycq=True)

# 3. Load FC data into memory

fc_data = read_fc_matrices(scan_list,DATA_DIR,FB_ATLAS_NAME)

# 4. Save FC data in vectorized form for all scans into a single file for easy access for batch jobs

out_path = osp.join(RESOURCES_CPM_DIR,'fc_data.csv')
fc_data.to_csv(out_path)
print('++ INFO: FC data saved to disk [%s]' % out_path)

out_path = osp.join(RESOURCES_CPM_DIR,'behav_data.csv')
snycq_df.to_csv(out_path)
print('++ INFO: Behavioral data saved to disk [%s]' % out_path)

# ***
# # 3. Run CPM

cpm_kwargs = {'r_thresh': 0.15, 'corr_type': 'spearman', 'verbose': True, 'edge_summary_method':'sum'} # these are defaults, but it's still good to be explicit
cpm_kwargs = {'p_thresh': 0.01, 'corr_type': 'spearman', 'verbose': True, 'edge_summary_method':'sum'} # these are defaults, but it's still good to be explicit
k          = 10
behav      = 'Images'

# %%time 
behav_obs_pred, models = cpm_wrapper(fc_data, snycq_df, behav, k=10, **cpm_kwargs)

r,p,f = plot_predictions(behav_obs_pred, figsize=(5,5), font_scale=1, color='black')
f

from utils.plotting import plot_as_circos
from scipy.spatial.distance import squareform

ATLASINFO_PATH = osp.join(ATLASES_DIR,FB_ATLAS_NAME,'{ATLAS_NAME}.roi_info.csv'.format(ATLAS_NAME=FB_ATLAS_NAME))
roi_info       = pd.read_csv(ATLASINFO_PATH)

thresh           = 0.8
model_consensus  = {}
for tail in ['pos','neg']:
    edge_frac              = models[tail].sum(axis=0)/models[tail].shape[0]
    model_consensus[tail]  = (edge_frac>=thresh).astype(int)
    num_edges_toshow = model_consensus[tail].sum()
    print("For the {tail} tail, {edges} edges were selected in at least {pct}% of folds".format(tail=tail, edges=num_edges_toshow, pct=thresh*100))
model_consensus_to_plot = pd.DataFrame(squareform(model_consensus['pos'])-squareform(model_consensus['neg']),
                          index = roi_info.set_index(['ROI_ID','ROI_Name','Hemisphere','Network']).index,
                          columns= roi_info.set_index(['ROI_ID','ROI_Name','Hemisphere','Network']).index)

plot_as_circos(model_consensus_to_plot,roi_info,figsize=(5,5),edge_weight=1)

# ***
# ## Development of subject-aware split

from collections import OrderedDict
from itertools import chain
from random import shuffle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

scan_list = fc_data.index


# First step of the modeling: assign a k-fold to each entry in the FC matrix
def mk_kfold_subject_aware_indices(scan_list, k = 10):
    """
    Splits scans into k folds taking into account subject identity
    
    INPUTS
    ======
    subj_list: list of scan identifiers (sbj,scan)
    k: number of folds
    
    OUTPUTS
    =======
    indices: np.array with one value per scan indicating k-fold membership
    """
    # Count the number of scans
    n_scans                        = len(scan_list)
    # Shuffle scans to randomize the folds across iterations
    groups    = [sbj for (sbj,scan) in  scan_list]
    # Create GroupKFold object for k splits
    grp_cv  = GroupShuffleSplit(n_splits=k)
    indices = np.zeros(n_scans)
    for fold, (_,ix_test) in enumerate(grp_cv.split(scan_list,groups=groups)):
        indices[ix_test]=fold
    indices = indices.astype(int)
    return indices


indices,groups = mk_kfold_subject_aware_indices(scan_list)
group_ids = LabelEncoder().fit_transform(groups)

# +
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

rng = np.random.RandomState(1338)
cmap_data = plt.cm.Paired
cmap_cv = plt.cm.coolwarm

def visualize_groups(classes, groups, name):
    # Visualize dataset groups
    fig, ax = plt.subplots()
    ax.scatter(
        range(len(groups)),
        [0.5] * len(groups),
        c=groups,
        marker="_",
        lw=50,
        cmap=cmap_data,
    )
    ax.scatter(
        range(len(groups)),
        [3.5] * len(groups),
        c=classes,
        marker="_",
        lw=50,
        cmap=cmap_data,
    )
    ax.set(
        ylim=[-1, 5],
        yticks=[0.5, 3.5],
        yticklabels=["Data\ngroup", "Data\nclass"],
        xlabel="Sample index",
    )


visualize_groups(indices, group_ids, "no groups")
# -

group_ids

original_grouping_vector = X['group']
original_grouping_vector

unique_values, indexes, inverse = np.unique(original_grouping_vector, return_inverse=True, return_index=True)
print(unique_values)
print(indexes)
print(inverse)

# +
new_grouping_vector = indexes[inverse] # This is where the magic happens!

splitter = GroupKFold()
for train, test in splitter.split(X, y, groups=new_grouping_vector):
    print(X.iloc[test, :])
# -

X.sort_values(by='group')


