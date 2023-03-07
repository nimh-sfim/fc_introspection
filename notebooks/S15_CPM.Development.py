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
from utils.basics import RESOURCES_CPM_DIR, DATA_DIR, FB_200ROI_ATLAS_NAME, ATLASES_DIR
from utils.basics import get_sbj_scan_list
from cpm.cpm import read_fc_matrices
from sfim_lib.io.afni import load_netcc
from tqdm import tqdm
#from cpm.cpm import mk_kfold_indices, split_train_test, get_train_test_data
#from cpm.cpm import select_features, build_model, apply_model
from cpm.cpm import cpm_wrapper
from cpm.plotting import plot_predictions
import hvplot.pandas
from sklearn.linear_model import Ridge

ATLAS_NAME = FB_200ROI_ATLAS_NAME

# # 1. Prepare Data in Disk
# 1. Create resources folder for CPM analyses

if not osp.exists(RESOURCES_CPM_DIR):
    print('++ INFO: Creating resources folder for CPM analyses [%s]' % RESOURCES_CPM_DIR)
    os.makedirs(RESOURCES_CPM_DIR)

# 2. Load list of scans that passed all QAs

sbj_list, scan_list, behav_data = get_sbj_scan_list(when='post_motion', return_snycq=True)

# 3. Load FC data into memory

fc_data = read_fc_matrices(scan_list,DATA_DIR,ATLAS_NAME)

# ***
# # 3. Development of initial type of ridge regression

from cpm.cpm import mk_kfold_indices_subject_aware, split_train_test, get_train_test_data, select_features, build_ridge_model, apply_ridge_model

from argparse import Namespace
opts = Namespace()
opts.randomize_behavior = False
opts.behavior           = 'Images'
opts.split_mode         = 'subject_aware'
opts.e_thr_r            = None
opts.e_thr_p            = 0.01
opts.corr_type          = 'pearson'
opts.verbose            = True
opts.e_summary_metric   = 'ridge'
opts.k                  = 10
opts.ridge_alpha        = 1.0

Nscans, n_edges = fc_data.shape
Nbehavs = behav_data.shape[1]
print ('++ INFO [main]: Behaviors table loaded into memory [#Behaviors=%d]' % Nbehavs)
print ('++ INFO [main]: FC data loaded into memory [#Scans=%d, #Connections=%d]' % (Nscans,n_edges))

# 2.1. Randomize Behavior if needed
if opts.randomize_behavior:
    print('++            Randomizing behavior ######## ATTENTION ##########')
    behav_index = behav_data.index
    behav_data  = behav_data.sample(frac=1).reset_index(drop=True)
    behav_data  = behav_data.set_index(behav_index)

# 3. Ensure there is a match in indexes between both data structures
assert fc_data.index.equals(behav_data.index), "++ ERROR [main]:Index in FC dataFrame and behavior dataframe do not match."    

# 4. Ensure selected behavior is available
assert opts.behavior in behav_data.columns, "++ ERROR [cpm_wrapper]:behavior not present in behav_data"

# 5. Extract a list of scan IDs
scan_list = fc_data.index

# Prepare for Cross-validation step
# =================================
print('++ INFO [main]: Generating lists of scans per k-fold [%s]' % opts.split_mode)
if opts.split_mode == 'basic':
    indices = mk_kfold_indices(scan_list, k=opts.k)
if opts.split_mode == 'subject_aware':
    indices = mk_kfold_indices_subject_aware(scan_list, k=opts.k)
# Create dictionary with configurations
cpm_kwargs = {'r_thresh': opts.e_thr_r,
                  'p_thresh': opts.e_thr_p,
                  'corr_type': opts.corr_type,
                  'verbose': opts.verbose,
                  'edge_summary_metric': opts.e_summary_metric}

# Create output data structures
# =============================
all_masks = {}
if opts.e_summary_metric == 'ridge':
    # 1. DataFrame for storing observed and predicted behaviors
    col_list = []
    col_list.append(opts.behavior + " predicted (ridge)")
    col_list.append(opts.behavior + " observed")
    behav_obs_pred = pd.DataFrame(index=scan_list, columns = col_list)
    # 2. Arrays for storing predictive models
    all_masks["pos"]   = np.zeros((opts.k, n_edges))
    all_masks["neg"]   = np.zeros((opts.k, n_edges))
    all_masks["ridge"] = np.zeros((opts.k, n_edges))
else:
    # 1. DataFrame for storing observed and predicted behaviors
    col_list = []
    for tail in ["pos", "neg", "glm"]:
        col_list.append(opts.behavior + " predicted (" + tail + ")")
    col_list.append(opts.behavior + " observed")
    behav_obs_pred = pd.DataFrame(index=scan_list, columns = col_list)
    # 2. Arrays for storing predictive models
    all_masks["pos"]   = np.zeros((opts.k, n_edges))
    all_masks["neg"]   = np.zeros((opts.k, n_edges))

# Run Cross-validation
# ====================
print('++ INFO [main]: Run cross-validation')
for fold in range(opts.k):
    print(" +              doing fold {}".format(fold), end=' --> ')
        
    # Gather testing and training data for this particular fold
    print('split_train_test', end=' --> ')
    train_subs, test_subs              = split_train_test(scan_list, indices, test_fold=fold)
    print('get_train_test_data', end=' --> ')
    train_vcts, train_behav, test_vcts = get_train_test_data(fc_data, train_subs, test_subs, behav_data, behav=opts.behavior)
    train_vcts  = train_vcts.infer_objects()
    train_behav = train_behav.infer_objects()
    test_vcts   = test_vcts.infer_objects()
        
    # Find edges that correlate above threshold with behavior in this fold
    print('select_features', end=' ')
    mask_dict   = select_features(train_vcts, train_behav, **cpm_kwargs)
    print('', end='--> ')
    
    # Gather the edges found to be significant in this fold in the all_masks dictionary
    all_masks["pos"][fold,:] = mask_dict["pos"]
    all_masks["neg"][fold,:] = mask_dict["neg"]
    #print(test_vcts.head(1))
    if opts.e_summary_metric == 'ridge':
            all_masks["ridge"][fold,:] = mask_dict["pos"] + mask_dict["neg"]
            print('build_ridge_model', end=' --> ')
            model       = build_ridge_model(train_vcts, all_masks["ridge"][fold,:], train_behav, opts.ridge_alpha)
            print('apply_model', end='\n')
            predictions = apply_ridge_model(test_vcts, all_masks["ridge"][fold,:], model)
            behav_obs_pred.loc[test_subs, opts.behavior + " predicted (ridge)"] = predictions 
    else:
        # Build model and predict behavior
        print('build_model', end=' --> ')
        model_dict = build_model(train_vcts, mask_dict, train_behav, edge_summary_method=opts.e_summary_metric)
        print('apply_model', end='\n')
        behav_pred = apply_model(test_vcts, mask_dict, model_dict, edge_summary_method=opts.e_summary_metric)
    
        # Update behav_obs_pred with results for this particular fold (the predictions for the test subjects only)
        for tail, predictions in behav_pred.items():
            behav_obs_pred.loc[test_subs, opts.behavior + " predicted (" + tail + ")"] = predictions
# Add observed behavior to the returned dataframe behav_obs_pred
# ==============================================================
behav_obs_pred.loc[scan_list, opts.behavior + " observed"] = behav_data[opts.behavior]
behav_obs_pred = behav_obs_pred.infer_objects()

r,p,f = plot_predictions(behav_obs_pred, tail='ridge', figsize=(5,5), font_scale=1, color='black')
f

from utils.plotting import plot_as_circos
from scipy.spatial.distance import squareform

ATLASINFO_PATH = osp.join(ATLASES_DIR,ATLAS_NAME,f'{ATLAS_NAME}.roi_info.csv')
roi_info       = pd.read_csv(ATLASINFO_PATH)

thresh           = 0.8
model_consensus  = {}
models = all_masks
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
# ## 4. Development of alternative version of ridge regression

# Run Cross-validation
# ====================
print('++ INFO [main]: Run cross-validation')
for fold in range(opts.k):
    print(" +              doing fold {}".format(fold), end=' --> ')
        
    # Gather testing and training data for this particular fold
    print('split_train_test', end=' --> ')
    train_subs, test_subs              = split_train_test(scan_list, indices, test_fold=fold)
    print('get_train_test_data', end=' --> ')
    train_vcts, train_behav, test_vcts = get_train_test_data(fc_data, train_subs, test_subs, behav_data, behav=opts.behavior)
    train_vcts  = train_vcts.infer_objects()
    train_behav = train_behav.infer_objects()
    test_vcts   = test_vcts.infer_objects()
        
    # Find edges that correlate above threshold with behavior in this fold
    print('select_features', end=' ')
    mask_dict   = select_features(train_vcts, train_behav, **cpm_kwargs)
    dfsf
    print('', end='--> ')
    
    # Gather the edges found to be significant in this fold in the all_masks dictionary
    all_masks["pos"][fold,:] = mask_dict["pos"]
    all_masks["neg"][fold,:] = mask_dict["neg"]
    #print(test_vcts.head(1))
    if opts.e_summary_metric == 'ridge':
            all_masks["ridge"][fold,:] = mask_dict["pos"] + mask_dict["neg"]
            print('build_ridge_model', end=' --> ')
            model       = build_ridge_model(train_vcts, all_masks["ridge"][fold,:], train_behav, opts.ridge_alpha)
            print('apply_model', end='\n')
            predictions = apply_ridge_model(test_vcts, all_masks["ridge"][fold,:], model)
            behav_obs_pred.loc[test_subs, opts.behavior + " predicted (ridge)"] = predictions 
    else:
        # Build model and predict behavior
        print('build_model', end=' --> ')
        model_dict = build_model(train_vcts, mask_dict, train_behav, edge_summary_method=opts.e_summary_metric)
        print('apply_model', end='\n')
        behav_pred = apply_model(test_vcts, mask_dict, model_dict, edge_summary_method=opts.e_summary_metric)
    
        # Update behav_obs_pred with results for this particular fold (the predictions for the test subjects only)
        for tail, predictions in behav_pred.items():
            behav_obs_pred.loc[test_subs, opts.behavior + " predicted (" + tail + ")"] = predictions
# Add observed behavior to the returned dataframe behav_obs_pred
# ==============================================================
behav_obs_pred.loc[scan_list, opts.behavior + " observed"] = behav_data[opts.behavior]
behav_obs_pred = behav_obs_pred.infer_objects()

mask_dict


def select_features(train_vcts, train_behav, r_thresh=None, p_thresh=None, d_thresh=None, corr_type='pearson', verbose=False, **other_options):

    assert not((p_thresh is not None) and (r_thresh is not None)), "++ERROR [select_features]: Threshold provided in two different ways. Do not know how to continue."
    assert corr_type in ['pearson','spearman'], "++ERROR [select_features]: Unknown correlation type."
    
    # Compute correlations between each edge and behavior
    n_edges = train_vcts.shape[1]
    r = pd.Series(index=range(n_edges),name='r', dtype=float)
    p = pd.Series(index=range(n_edges),name='p', dtype=float)
    for edge in range(n_edges):
        if corr_type == 'pearson':
            r[edge],p[edge] = pearsonr(train_vcts.loc[:,edge], train_behav)
        if corr_type == 'spearman':
            r[edge],p[edge] = spearmanr(train_vcts.loc[:,edge], train_behav)
    # Select edges according to thresholding criteria
    mask_dict = {}
    if p_thresh is not None:
        mask_dict["pos"] = (r > 0) & (p<p_thresh)
        mask_dict["neg"] = (r < 0) & (p<p_thresh)
    if r_thresh is not None:
        mask_dict["pos"] = r > r_thresh
        mask_dict["neg"] = r < -r_thresh
    if verbose:
        print("Found ({}/{}) edges positively/negatively correlated with behavior in the training set".format(mask_dict["pos"].sum(), mask_dict["neg"].sum()), end='') # for debugging
    return mask_dict


r_thresh = None
p_thresh = None
d_thresh = None
thres_list = [r_thresh,p_thresh,d_thresh]


