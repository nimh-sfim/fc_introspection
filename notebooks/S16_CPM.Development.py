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
import random
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

seed = random.seed(43)

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

# # Fix seed

from cpm.cpm import split_train_test, get_train_test_data, select_features, build_ridge_model, apply_ridge_model, build_model, apply_model
from sklearn.model_selection import GroupShuffleSplit


def mk_kfold_indices_subject_aware(scan_list, k = 10):
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
    grp_cv  = GroupShuffleSplit(n_splits=k, random_state=43)
    indices = np.zeros(n_scans)
    for fold, (_,ix_test) in enumerate(grp_cv.split(scan_list,groups=groups)):
        indices[ix_test]=fold
    indices = indices.astype(int)
    return indices


# ***
# # 3. Development of initial type of ridge regression

from argparse import Namespace
opts = Namespace()
opts.randomize_behavior = False
opts.behavior           = 'Surroundings'
opts.split_mode         = 'subject_aware'
opts.e_thr_r            = None
opts.e_thr_p            = 0.01
opts.corr_type          = 'pearson'
opts.verbose            = True
opts.e_summary_metric   = 'sum'
opts.k                  = 10
opts.ridge_alpha        = 1.0
opts.confounds          = True

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

print(indices)

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

r,p,f = plot_predictions(behav_obs_pred, tail='glm', figsize=(5,5), font_scale=1, color='black', accuracy_metric='pearson')
f

from utils.plotting import plot_as_circos
from scipy.spatial.distance import squareform

ATLASINFO_PATH = osp.join(ATLASES_DIR,ATLAS_NAME,f'{ATLAS_NAME}.roi_info.csv')
roi_info       = pd.read_csv(ATLASINFO_PATH)

thresh           = 0.9
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

from sklearn.feature_selection import f_regression, r_regression
from sklearn.feature_selection import SelectPercentile
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupShuffleSplit
from cpm.cpm import mk_kfold_indices

opts.e_thresh_d = None

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
    print('select_features: ', end=' ')
    if opts.e_summary_metric == 'ridge':
        # Select Features for this fold
        if opts.e_thresh_d is not None:
            feature_selection = SelectPercentile(f_regression,percentile=opts.e_thresh_d).fit(train_vcts, train_behav)
            sel_features      = list(feature_selection.get_feature_names_out(input_features=train_vcts.columns))
        else:
            f_stat, p_val     = f_regression(train_vcts.values, train_behav.values)
            sel_features      = list(np.where(p_val < opts.e_thr_p)[0])
        # Translate into a pd.Series of True and False with index being connectionID
        aux                        = pd.Series(False,index=train_vcts.columns)
        aux[sel_features]          = True
        all_masks['ridge'][fold,:] = aux
        # Add information about positive and negative edges
        r_vals = r_regression(train_vcts[sel_features], train_behav.values)
        all_masks['pos'][fold,sel_features]   = [1 if x > 0 else 0 for x in r_vals]
        all_masks['neg'][fold,sel_features]   = [1 if x < 0 else 0 for x in r_vals]
        print('found [{} pos + {} neg] -> {} features '.format(int(all_masks['pos'][fold,:].sum()),int(all_masks['neg'][fold,:].sum()),int(all_masks['ridge'][fold,:].sum())), end='--> ')
        # Build Ridge Model
        print('build_ridge_model', end=' --> ')
        model       = build_ridge_model(train_vcts, all_masks['ridge'][fold,:], train_behav, opts.ridge_alpha)
        print('apply_model', end='\n')
        predictions = apply_ridge_model(test_vcts, all_masks["ridge"][fold,:], model)
        behav_obs_pred.loc[test_subs, opts.behavior + " predicted (ridge)"] = predictions 
    else:
        mask_dict   = select_features(train_vcts, train_behav, **cpm_kwargs)
        print('', end='--> ')
        # Gather the edges found to be significant in this fold in the all_masks dictionary
        all_masks["pos"][fold,:] = mask_dict["pos"]
        all_masks["neg"][fold,:] = mask_dict["neg"]
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

r,p,f = plot_predictions(behav_obs_pred, tail='glm', figsize=(5,5), font_scale=1, color='black')
f

from utils.plotting import plot_as_circos
from scipy.spatial.distance import squareform

ATLASINFO_PATH = osp.join(ATLASES_DIR,ATLAS_NAME,f'{ATLAS_NAME}.roi_info.csv')
roi_info       = pd.read_csv(ATLASINFO_PATH)

thresh           = 0.9
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

# # Confounds

from utils.basics import RESOURCES_DINFO_DIR, PRJ_DIR
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
motion_confound_cpm_path = osp.join(RESOURCES_DINFO_DIR,'motion_confounds.csv')
mot_confounds = pd.read_csv(motion_confound_cpm_path, index_col=['Subject','Run'])


def get_confounds(train_subs, test_subs, motion=True, age=False, vigilance=True):
    """
    Extracts confounds (e.g., motion, time of day) for each subject
    """
    train_index = pd.MultiIndex.from_tuples(train_subs, names=['Subject','Run'])
    test_index  = pd.MultiIndex.from_tuples(test_subs, names=['Subject','Run'])
    cols = []
    if motion:
        cols.append('Motion')
    if vigilance:
        cols.append('Vigilance')
    if age:
        cols.append('Age')
    train_confounds_df = pd.DataFrame(index=train_index, columns=cols)
    test_confounds_df  = pd.DataFrame(index=test_index, columns=cols)
    # Get motion confounds
    if motion is True:
        train_confounds_df['Motion'] = mot_confounds.loc[train_index].values
        test_confounds_df['Motion']  = mot_confounds.loc[test_index].values
    # Get vigilance confounds
    if vigilance is True:
        train_confounds_df['Vigilance'] = behav_data.loc[train_index,'Vigilance'].values
        test_confounds_df['Vigilance']  = behav_data.loc[test_index,'Vigilance'].values
    return train_confounds_df, test_confounds_df


def residualize(y,confounds):

    for confound in confounds.columns:
        print("R_before = {:.3f}".format(pearsonr(y, confounds[confound])[0]), end=', ')

    lm = LinearRegression().fit(confounds, y)
    y_resid = y - lm.predict(confounds)

    for confound in confounds.columns:
        print("R_after = {:.3f}".format(pearsonr(y_resid, confounds[confound])[0]), end=' ')

    return y_resid, lm


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
    train_conf, test_conf              = get_confounds(train_subs, test_subs, )
    train_vcts  = train_vcts.infer_objects()
    train_behav = train_behav.infer_objects()
    test_vcts   = test_vcts.infer_objects()
    test_behav  = behav_data.loc[test_subs,opts.behavior] 
    if opts.confounds:
        print('residualize motion: ',end='')
        train_confounds, test_confounds = get_confounds(train_subs, test_subs, vigilance=False)
        train_confounds = train_confounds.infer_objects()
        test_confounds  = test_confounds.infer_objects()
        train_behav, confound_model = residualize(train_behav, train_confounds)
        test_behav = test_behav - confound_model.predict(test_confounds)
        print(' --> ', end='')
        
    # Add observed behavior to the returned dataframe behav_obs_pred (need to do this here in case there was confound resid)
    # ======================================================================================================================
    behav_obs_pred.loc[test_subs, opts.behavior + " observed"] = test_behav

    # Find edges that correlate above threshold with behavior in this fold
    print('select_features: ', end=' ')
    if opts.e_summary_metric == 'ridge':
        # Select Features for this fold
        if opts.e_thresh_d is not None:
            feature_selection = SelectPercentile(f_regression,percentile=opts.e_thresh_d).fit(train_vcts, train_behav)
            sel_features      = list(feature_selection.get_feature_names_out(input_features=train_vcts.columns))
        else:
            f_stat, p_val     = f_regression(train_vcts.values, train_behav.values)
            sel_features      = list(np.where(p_val < opts.e_thr_p)[0])
        # Translate into a pd.Series of True and False with index being connectionID
        aux                        = pd.Series(False,index=train_vcts.columns)
        aux[sel_features]          = True
        all_masks['ridge'][fold,:] = aux
        # Add information about positive and negative edges
        r_vals = r_regression(train_vcts[sel_features], train_behav.values)
        all_masks['pos'][fold,sel_features]   = [1 if x > 0 else 0 for x in r_vals]
        all_masks['neg'][fold,sel_features]   = [1 if x < 0 else 0 for x in r_vals]
        print('[{} pos + {} neg] -> {} features '.format(int(all_masks['pos'][fold,:].sum()),int(all_masks['neg'][fold,:].sum()),int(all_masks['ridge'][fold,:].sum())), end='--> ')
        # Build Ridge Model
        print('build_ridge_model', end=' --> ')
        model       = build_ridge_model(train_vcts, all_masks['ridge'][fold,:], train_behav, opts.ridge_alpha)
        print('apply_model', end='\n')
        predictions = apply_ridge_model(test_vcts, all_masks["ridge"][fold,:], model)
        behav_obs_pred.loc[test_subs, opts.behavior + " predicted (ridge)"] = predictions 
    else:
        mask_dict   = select_features(train_vcts, train_behav, **cpm_kwargs)
        print('', end=' --> ')
        # Gather the edges found to be significant in this fold in the all_masks dictionary
        all_masks["pos"][fold,:] = mask_dict["pos"]
        all_masks["neg"][fold,:] = mask_dict["neg"]
        # Build model and predict behavior
        print('build_model', end=' --> ')
        model_dict = build_model(train_vcts, mask_dict, train_behav, edge_summary_method=opts.e_summary_metric)
        print('apply_model', end='\n')
        behav_pred = apply_model(test_vcts, mask_dict, model_dict, edge_summary_method=opts.e_summary_metric)
        # Update behav_obs_pred with results for this particular fold (the predictions for the test subjects only)
        for tail, predictions in behav_pred.items():
            behav_obs_pred.loc[test_subs, opts.behavior + " predicted (" + tail + ")"] = predictions


r,p,f = plot_predictions(behav_obs_pred, tail='glm', figsize=(5,5), font_scale=1, color='black')
f

from utils.plotting import plot_as_circos
from scipy.spatial.distance import squareform

ATLASINFO_PATH = osp.join(ATLASES_DIR,ATLAS_NAME,f'{ATLAS_NAME}.roi_info.csv')
roi_info       = pd.read_csv(ATLASINFO_PATH)

thresh           = 0.9
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



a = mk_kfold_indices_subject_aware(scan_list)
a = mk_kfold_indices(scan_list)

for k in range(10):
    aa = list(np.where(a==k)[0])
    bb = list(np.where(a!=k)[0])
    test_sbjs  = behav_data.iloc[aa].index.get_level_values('Subject').unique()
    train_sbjs = behav_data.iloc[bb].index.get_level_values('Subject').unique()
    mixed_sbjs = [sbj for sbj in test_sbjs if sbj in train_sbjs]
    print(k,len(train_sbjs) + len(test_sbjs), len(mixed_sbjs))

import pickle
opts.behavior = 'Vigilance'
path = '/data/SFIMJGC_Introspec/2023_fc_introspection/code/fc_introspection/resources/cpm/swarm_outputs/real/Schaefer2018_200Parcels_7Networks_AAL2/subject_aware/conf_residualized/pearson_sum/Vigilance/cpm_Vigilance_rep-00001.pkl'
with open(path,'rb') as f:
    a = pickle.load(f)

behav_obs_pred = a['behav_obs_pred']
behav_obs_pred = behav_obs_pred.infer_objects()

for tail in ["pos","neg","glm"]:
    print('++ INFO [main]: R(obs,pred-%s) = %.3f' % (tail, behav_obs_pred[opts.behavior + " observed"].corr(behav_obs_pred[opts.behavior + " predicted (" + tail + ")"])))
    

np.corrcoef(behav_obs_pred[opts.behavior + " observed"].values,behav_obs_pred[opts.behavior + " predicted (" + tail + ")"].values)[0,1]

aa = behav_obs_pred[opts.behavior + " observed"].values
bb = behav_obs_pred[opts.behavior + " predicted (" + tail + ")"].values
#np.corrcoef(aa,bb)

bb


