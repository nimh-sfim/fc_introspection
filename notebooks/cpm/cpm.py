# This file contains code copied from Emily Finn's github account 

import pandas as pd
import os.path as osp
from scipy.spatial.distance import squareform
from scipy.stats import pearsonr, spearmanr
from sfim_lib.io.afni import load_netcc
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import GroupShuffleSplit
# Input Functions
# ===============
def read_fc_matrices(scan_list,data_dir,atlas_name,fisher_transform=False):
    """Read all FC matrices into memory and return a pandas dataframe where
    each row contains a vectorized version of the FC matrix of a given scan.
    
    INPUTS:
    =======
    scan_list: list of tuples (subj_id,run_id) with information about all the scans to enter the analyses.
    
    data_dir: path to the location where data was pre-processed.
    
    atlas_name: name of the atlas used to extract the representative timseries and FC matrices.
    
    fisher_transform: whether or not to do fisher transform of the correlation values.
    
    OUTPUT:
    =======
    df : pd.DataFrame (#rows = # unique scans, #columns = # unique connections)
    """
    # Load One Matrix to obtain number of connections
    sbj,run    = scan_list[0]
    _,_,_,_,run_num,_,run_acq = run.split('-')
    netcc_path = osp.join(data_dir,'PrcsData',sbj,'preprocessed','func','pb06_staticFC','{run_acq}_run-{run_num}.{ATLAS_NAME}_000.netcc'.format(run_acq = run_acq, run_num = run_num, ATLAS_NAME = atlas_name))
    netcc      = load_netcc(netcc_path)
    Nrois      = netcc.shape[0]
    Nconns     = int(Nrois*(Nrois-1)/2)
    # Initialize data structure that will hold all connectivity matrices
    df = pd.DataFrame(index=pd.MultiIndex.from_tuples(scan_list, names=['Subject','Run']), columns=range(Nconns))
    # Load all FC and place them into the final dataframe
    for sbj,run in tqdm(scan_list, desc='Scan'):
        _,_,_,_,run_num,_,run_acq = run.split('-')
        netcc_path = osp.join(data_dir,'PrcsData',sbj,'preprocessed','func','pb06_staticFC','{run_acq}_run-{run_num}.{ATLAS_NAME}_000.netcc'.format(run_acq = run_acq, run_num = run_num, ATLAS_NAME = atlas_name))
        netcc      = load_netcc(netcc_path)
        # Apply Fisher-transform if needed
        if fisher_transform:
            netcc = netcc.apply(np.arctanh)
        # Set diagonal to zero
        aux = netcc.values
        np.fill_diagonal(aux,0)
        netcc = pd.DataFrame(aux,index=netcc.index, columns=netcc.columns)
        # Vectorize
        netcc_vector = squareform(netcc)
        # Add matrix to output
        df.loc[sbj,run] = netcc_vector
    return df

# ==========================================
# === Cross-validation related functions ===
# ==========================================
# First step of the modeling: assign a k-fold to each entry in the FC matrix
def mk_kfold_indices(subj_list, k = 10):
    """
    Splits list of subjects into k folds for cross-validation.
    
    INPUTS
    ======
    subj_list: list of scan identifiers
    k: number of folds
    
    OUTPUTS
    =======
    indices: np.array with one value per scan indicating k-fold membership
    """
    
    n_subs = len(subj_list)
    n_subs_per_fold = n_subs//k # floor integer for n_subs_per_fold

    indices = [[fold_no]*n_subs_per_fold for fold_no in range(k)] # generate repmat list of indices
    remainder = n_subs % k # figure out how many subs are left over
    remainder_inds = list(range(remainder))
    indices = [item for sublist in indices for item in sublist]    
    [indices.append(ind) for ind in remainder_inds] # add indices for remainder subs

    assert len(indices)==n_subs, "Length of indices list does not equal number of subjects, something went wrong"

    np.random.shuffle(indices) # shuffles in place

    return np.array(indices)

# First step of the modeling: assign a k-fold to each entry in the FC matrix
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
    grp_cv  = GroupShuffleSplit(n_splits=k)
    indices = np.zeros(n_scans)
    for fold, (_,ix_test) in enumerate(grp_cv.split(scan_list,groups=groups)):
        indices[ix_test]=fold
    indices = indices.astype(int)
    return indices

# Then get the subject/scan names for the given fold (train and test)
def split_train_test(subj_list, indices, test_fold):
    """
    For a subj list, k-fold indices, and given fold, returns lists of train_subs and test_subs
    """

    train_inds = np.where(indices!=test_fold)
    test_inds = np.where(indices==test_fold)

    train_subs = []
    for sub in subj_list[train_inds]:
        train_subs.append(sub)

    test_subs = []
    for sub in subj_list[test_inds]:
        test_subs.append(sub)

    return (train_subs, test_subs)

# Then get the FC data for the training and test subjects and the behavioral data for the training subjects
def get_train_test_data(all_fc_data, train_subs, test_subs, behav_data, behav):

    """
    Extracts requested FC and behavioral data for a list of train_subs and test_subs
    """

    train_vcts = all_fc_data.loc[train_subs, :]
    test_vcts = all_fc_data.loc[test_subs, :]

    train_behav = behav_data.loc[train_subs, behav]

    return (train_vcts, train_behav, test_vcts)
   
# =====================================================
# ===                  CPM Functions                ===
# =====================================================
# Then correlate the edges with the training behavioral data and select the edges that pass the given threshold 
def select_features(train_vcts, train_behav, r_thresh=None, p_thresh=None, corr_type='pearson', verbose=False, **other_options):

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

##def select_features(train_vcts, train_behav, r_thresh=None, p_thresh=None, corr_type='pearson', verbose=False, **other_options):
##    """
##    Runs the CPM feature selection step: 
##    - correlates each edge with behavior, and returns a mask of edges that are correlated above some threshold, one for each tail (positive and negative)
##    """
##
##    assert train_vcts.index.equals(train_behav.index), "Row indices of FC vcts and behavior don't match!"
##    
##    # Correlate all edges with behav vector
##    
##    corr = train_vcts.corrwith(train_behav, method=corr_type)
##    
##    # Define positive and negative masks
##    mask_dict = {}
##    mask_dict["pos"] = corr > r_thresh
##    mask_dict["neg"] = corr < -r_thresh
##    
##    if verbose:
##        print("Found ({}/{}) edges positively/negatively correlated with behavior in the training set".format(mask_dict["pos"].sum(), mask_dict["neg"].sum())) # for debugging
##
##    return mask_dict
   
def build_model(train_vcts, mask_dict, train_behav, edge_summary_method='sum'):
    """
    Builds a CPM model:
    - takes a feature mask, sums all edges in the mask for each subject, and uses simple linear regression to relate summed network strength to behavior
    
    INPUTS
    ======
    train_vects: np.array(#scan,#connections) with the FC training data
    
    mask_dict: dictionary with two keys ('pos' and 'neg') with information about which edges were selected as meaningful during the edge selection step
    
    train_behav: np.array(#scans,) with the behavior to be predicted
    
    edge_summary_method: whether to add or average edge strengths across all edges selected for a model. Initial version of CPM uses sum, but a variant
                         with mean was reported by Jangraw et al. 2018. This variant should in principle be less senstive to the number of edges entering
                         the model.
    
    OUTPUTS
    =======
    model_dict: contains one entry per model (pos, neg and glm). For each of the three models contains a tuple with two values, first the slope and then
                the intercept.
    
    NOTE: This function has been updated relative to the code in the cpm tutorial so that it does not fail for null models. When null models are present
          the the model is filled with np.nans, which in the other functions should be interpreted as a non-existing model.
    """
    
    assert train_vcts.index.equals(train_behav.index), "Row indices of FC vcts and behavior don't match!"
    assert edge_summary_method in ['mean','sum'], "Edge summary method not recognized"
    model_dict = {}
    X_glm      = None
    # FOR BOTH THE POSITIVE AND NEGATIVE MODEL
    for t, (tail, mask) in enumerate(mask_dict.items()):
        if mask.sum()>0:       # At least one edge entered the model
            if edge_summary_method == 'sum':
                X                  = train_vcts.values[:, mask].sum(axis=1) # Pick the values for the edges in each subject and sum them.
            elif edge_summary_method == 'mean':
                X                  = train_vcts.values[:, mask].mean(axis=1) # Pick the values for the edges in each subject and average them.
            y                  = train_behav
            (slope, intercept) = np.polyfit(X, y, 1)
            model_dict[tail]   = (slope, intercept)
            if X_glm is None:
                X_glm = np.reshape(X,(X.shape[0],1))
            else:
                X_glm = np.c_[X_glm,X]
        else:
            print("++ WARNING [build_model_new,%s,%d]: No edges entered the model --> Setting slope and intercept to np.nan" % (tail,mask.sum()))
            model_dict[tail] = (np.nan, np.nan)
    # CONSTRUCT THE FULL MODEL WITH POSITIVE AND NEGATIVE TOGETHER
    if X_glm is None:
        print("++ WARNING [build_model_new,glm]: No edges entered the model --> Setting slope and intercept to np.nan")
        model_dict["glm"] = (np.nan, np.nan)
    else:
        X_glm = np.c_[X_glm, np.ones(X_glm.shape[0])]
        model_dict["glm"] = tuple(np.linalg.lstsq(X_glm, y, rcond=None)[0])
    return model_dict
   
def apply_model(test_vcts, mask_dict, model_dict, edge_summary_method='sum'):
    """
    Applies a previously trained linear regression model to a test set to generate predictions of behavior.
    
    INPUTS
    ======
    test_vcts: np.array(#scan,#connections) with the FC training data
    
    mask_dict: dictionary with two keys ('pos' and 'neg') with information about which edges were 
               selected as meaningful during the edge selection step.
    
    
    model_dict: dictionary with three keys ('pos', 'neg' and 'glm'). For each models, the dictionary
                object contains a tuple with two values (or three for 'glm'), first the slope/s and 
                then the intercept.
    
    edge_summary_method: whether to add or average edge strengths across all edges selected for a model. 
                         Initial version of CPM uses sum, but a variant with mean was reported by Jangraw
                         et al. 2018. This variant should in principle be less senstive to the number of 
                         edges entering the model.
    OUTPUT
    ======
    behav_pred: dictionary with one key per model. For each model it contains a pd.Series with the
                predicted behaviors. Unless model was empty.
    """
    
    assert edge_summary_method in ['mean','sum'], "Edge summary method not recognized"
    behav_pred = {}
    X_glm = None

    for t, (tail, mask) in enumerate(mask_dict.items()):
        if mask.sum()>0: # At least one edge entered the model
            if edge_summary_method == 'sum':
                X                = test_vcts.loc[:, mask].sum(axis=1).values
            elif edge_summary_method == 'mean':
                X                = test_vcts.loc[:, mask].mean(axis=1).values
            slope, intercept = model_dict[tail]
            behav_pred[tail] = pd.Series(slope*X + intercept).set_axis(test_vcts.index)
            if X_glm is None:
                X_glm = np.reshape(X,(X.shape[0],1))
            else:
                X_glm = np.c_[X_glm,X] 
        else:
            behav_pred[tail] = pd.Series(np.ones(test_vcts.shape[0])*np.nan).set_axis(test_vcts.index)
    
    if X_glm is None:
       behav_pred["glm"] = pd.Series(np.ones(test_vcts.shape[0])*np.nan).set_axis(test_vcts.index)
    else:
       X_glm             = np.c_[X_glm, np.ones(X_glm.shape[0])]
       behav_pred["glm"] = pd.Series(np.dot(X_glm, model_dict["glm"])).set_axis(test_vcts.index)
    return behav_pred
   
def cpm_wrapper(fc_data, behav_data, behav, k=10, **cpm_kwargs):
    """This function will run the whole CPM algorithm given a set of connectivity data, a target behavior to predict and a few hyper-parameters.
    
    INPUTS
    ======
    fc_data: a pd.DataFrame with rows denoting scans and columns denoting connections (#scans, #unique connections).
    
    behav_data: a pd.DataFrame with rows denoting scans and columns denoting behaviors (#scans, #behaviors).
    
    behav: column identifier to select a behavior to be predicted among those presented as columns in behav_data
    
    k: number of k-folds for the cross-validation procedure. [default = 10]
    
    cpm_kwargs: additional hyper-parameters for the CPM algoritm in the form of a dictionary.
                * r_threh: edge-based threshold for the feature selection step.
                * corr_type: correlation type for the edge selection step (pearson or spearman).
                * verbose: show additional information.
    
    OUTPUTS
    =======
    behav_obs_pred: 
    
    all_masks:
    """
    # Check input requirements
    # ========================
    assert isinstance(fc_data,pd.DataFrame), "++ ERROR [cpm_wrapper]: fc_data is not an instance of pd.DataFrame"
    assert isinstance(behav_data,pd.DataFrame), "++ ERROR [cpm_wrapper]:fc_data is not an instance of pd.DataFrame"
    assert behav in behav_data.columns, "++ ERROR [cpm_wrapper]:behavior not present in behav_data"
    assert fc_data.index.equals(behav_data.index), "++ ERROR [cpm_wrapper]:Index in FC dataFrame and behavior dataframe do not match"
    assert (('r_thresh' in cpm_kwargs) & ('p_thresh' not in cpm_kwargs)) | (('r_thresh' not in cpm_kwargs) & ('p_thresh' in cpm_kwargs)), "++ ERROR [cpm_wrapper]: Provided edge-level threshold as both p-value and r-value. Remove one"
    
    if ('r_thresh' not in cpm_kwargs):
        cpm_kwargs['r_thresh'] = None
    if ('p_thresh' not in cpm_kwargs):
        cpm_kwargs['p_thresh'] = None
    print(cpm_kwargs)
    
    # Extract list of scan identifiers from the fc_data.index
    # =======================================================
    scan_list = fc_data.index

    # Split the same into k equally sized (as much as possible) folds
    # ===============================================================
    indices = mk_kfold_indices(scan_list, k=k)
    
    # Verbose
    # =======    
    if cpm_kwargs['verbose']:
        print('++ INFO [cpm_wrapper]: Number of scans                      = %d' % len(scan_list))
        print('++ INFO [cpm_wrapper]: Number K-folds                       = %d' % k)
        print('++ INFO [cpm_wrapper]: Correlation mode                     = %s' % cpm_kwargs['corr_type'])
        if not (cpm_kwargs['r_thresh'] is None):
            print('++ INFO [cpm_wrapper]: Edge Selection Threshold (r)         = %.3f' % cpm_kwargs['r_thresh'])
        if not (cpm_kwargs['p_thresh'] is None):
            print('++ INFO [cpm_wrapper]: Edge Selection Threshold (p-val)     = %.3f' % cpm_kwargs['p_thresh'])
        print('++ INFO [cpm_wrapper]: Target Beahvior Label                = %s' % behav)
        print('++ INFO [cpm_wrapper]: Edge Summarization Method            = %s' % cpm_kwargs['edge_summary_method'])
        
    # Initialize df for storing observed and predicted behavior for the three models
    col_list = []
    for tail in ["pos", "neg", "glm"]:
        col_list.append(behav + " predicted (" + tail + ")")
    col_list.append(behav + " observed")
    behav_obs_pred = pd.DataFrame(index=scan_list, columns = col_list)

    # Initialize array for storing feature masks to all zeros
    n_edges = fc_data.shape[1]
    all_masks = {}
    all_masks["pos"] = np.zeros((k, n_edges))
    all_masks["neg"] = np.zeros((k, n_edges))

    # For each cross-validation fold
    for fold in range(k):
        print(" + doing fold {}".format(fold), end=' --> ')
        # Gather testing and training data for this particular fold
        # =========================================================
        train_subs, test_subs              = split_train_test(scan_list, indices, test_fold=fold)
        train_vcts, train_behav, test_vcts = get_train_test_data(fc_data, train_subs, test_subs, behav_data, behav=behav)
        train_vcts  = train_vcts.infer_objects()
        train_behav = train_behav.infer_objects()
        test_vcts   = test_vcts.infer_objects()
        # Find edges that correlate above threshold with behavior in this fold
        # ====================================================================
        mask_dict   = select_features(train_vcts, train_behav, **cpm_kwargs)
        # Gather the edges found to be significant in this fold in the all_masks dictionary
        all_masks["pos"][fold,:] = mask_dict["pos"]
        all_masks["neg"][fold,:] = mask_dict["neg"]
        # Build model and predict behavior
        # ================================
        model_dict = build_model(train_vcts, mask_dict, train_behav)
        behav_pred = apply_model(test_vcts, mask_dict, model_dict)
    
        # Update behav_obs_pred with results for this particular fold (the predictions for the test subjects only)
        # ========================================================================================================
        for tail, predictions in behav_pred.items():
            behav_obs_pred.loc[test_subs, behav + " predicted (" + tail + ")"] = predictions

    # Add observed behavior to the returned dataframe behav_obs_pred
    # ==============================================================
    behav_obs_pred.loc[scan_list, behav + " observed"] = behav_data[behav]
    
    return behav_obs_pred, all_masks