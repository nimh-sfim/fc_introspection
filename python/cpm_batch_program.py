import argparse
import os.path as osp
import os
import pandas as pd
import numpy as np
from sfim_lib.cpm.cpm import mk_kfold_test_indices, mk_kfold_test_indices_subject_aware, split_train_test, get_train_test_data, select_features, build_model, apply_model, get_confounds, residualize
import pickle

def read_command_line():
    parser = argparse.ArgumentParser(description='Run the CPM algorithm given a set of FC matrices (vectorized) and an external quantity (e.g., behavior) to be predicted.')
    parser.add_argument('-b','--behav_path',          type=str,   help="Path to dataframe with behaviors",             required=True, dest='behav_path')
    parser.add_argument('-f','--fc_a_path',           type=str,   help="Path to dataframe with fc vectors (case 1)",            required=True, dest='fc_a_path')
    parser.add_argument('-F','--fc_b_path',           type=str,   help="Path to dataframe with fc vectors (case 2)",            required=True, dest='fc_b_path')
    parser.add_argument('-l','--fc_a_label',          type=str,   help="Label for fc vectors (case 1)",            required=True, dest='fc_a_label')
    parser.add_argument('-L','--fc_b_label',          type=str,   help="Label for fc vectors (case 2)",            required=True, dest='fc_b_label')
    parser.add_argument('-t','--target_behav',        type=str,   help="Target behavior to predict",                   required=True, dest='behavior')
    parser.add_argument('-k','--number_of_folds',     type=int,   help="Number of cross-validation folds",             required=False, default=10, dest='k')
    parser.add_argument('-o','--output_dir',          type=str,   help="Output folder where to write results",         required=True, dest='output_dir')
    parser.add_argument('-i','--iter',                type=int,   help="Repetition/Iteration number",                  required=True, default=0, dest='repetition')
    parser.add_argument('-p','--edge_threshold_p',    type=float, help="Threshold for edge-selection (as p-value)",    required=False, default=None, dest='e_thr_p')
    parser.add_argument('-r','--edge_threshold_r',    type=float, help="Threshold for edge-selection (as r-value)",    required=False, default=None, dest='e_thr_r')
    parser.add_argument('-d','--edge_threshold_d',    type=float, help="Threshold for edge-selection (as density in percent - eg 10 for keeping 10% connections)",    required=False, default=None, dest='e_thr_d')
    parser.add_argument('-c','--corr_type',           type=str,   help="Correlation type to use in edge-selection",    required=False, default='spearman', choices=['spearman','pearson'], dest='corr_type')
    parser.add_argument('-s','--edge_summary_metric', type=str,   help="How to summarize across edges in final model", required=False, default='sum',      choices=['sum','mean','ridge'], dest='e_summary_metric')
    parser.add_argument('-m','--split_mode',          type=str,   help="Type of data split for k-fold",                required=False, default='basic',    choices=['basic','subject_aware'], dest='split_mode')
    parser.add_argument('-n','--randomize_behavior', action='store_true', help="Randomized behavioral values for creating null distibutions", dest='randomize_behavior', required=False)
    parser.add_argument('-v','--verbose',            action='store_true', help="Show additional information on screen",                       dest='verbose', required=False)
    parser.add_argument('-a','--ridge_alpha', type=float, help='Alpha parameter for ridge regression', required=False, default=1.0, dest='ridge_alpha')
    parser.add_argument('-C','--residualize_motion', action='store_true', help="Remove motion from prediction target", dest='confounds', required=False)
    parser.add_argument('-M','--confound_path',       type=str, required=False ,help='Path to confound file',dest='confounds_path', default=None)
    parser.set_defaults(verbose=False)
    parser.set_defaults(randomize_behavior=False)
    parser.set_defaults(confounds=False)
    return parser.parse_args()

def main():
    opts = read_command_line()
    assert opts.e_summary_metric != 'ridge', "++ ERROR: This DUAL version of the program is not yet ready to deal with Ridge regression approach."
    # ===========================================
    print('++ ----------------------------------------------------------')
    print('   * Behavioral DataFrame Path           : %s' % opts.behav_path)
    assert osp.exists(opts.behav_path),'++ ERROR [main]: Behavioral Dataframe not found.'
    print('   * [%s] FC Dataframe Path            : %s' % (opts.fc_label,opts.fc_path))
    assert osp.exists(opts.fc_path), '++ ERROR [main]: FC Dataframe not found.'
    aux_out_dir = osp.join(opts.output_dir, opts.fc_label)
    print('   * [%s] Output Folder                : %s' % (opts.fc_label,aux_out_dir), end=' | ')
    if osp.exists(aux_out_dir):
        print(' ALREADY EXISTS', end='\n')
    else:
        os.makedirs(aux_out_dir)
        print(' JUST CREATED', end='\n')
    print('   * Behavior to predict                 : %s' % opts.behavior)
    print('   * Number of folds                     : %d folds' % opts.k)
    print('   * Repetition Number                   : %d' % opts.repetition)
    print('   * [Cross-validation] Spliting mode    : %s' % opts.split_mode)
    # Check only one thresholding option is provided
    thresh_list = [opts.e_thr_p, opts.e_thr_r, opts.e_thr_d]
    thresh_list = [t for t in thresh_list if t is not None]
    num_thresh_options = len(thresh_list)
    assert num_thresh_options == 1, '++ ERROR [main]: Edge thresholding provided in more than one way. Please correct'
    if (opts.e_summary_metric == 'ridge') & (opts.e_thr_p is not None):
       print('++ ERROR [main]: Ridge regression with R-based threshold not implemented. Please provide a D- or p- based threshold')
       exit
    if opts.e_thr_p is not None:
       print('   * [Edge Selection] - Threshold        : p < %.3f'% opts.e_thr_p)
    if opts.e_thr_r is not None:
       print('   * [Edge Selection] - Threshold        : R > %.3f' % opts.e_thr_r)
    if opts.e_thr_d is not None:
       print('   * [Edge Selection] - Threshold        : Density > %.3f' % opts.e_thr_d)
    print('   * [Edge Selection] - Correlation Type : %s' % opts.corr_type)
    if opts.confounds:
       print('   * [Edge Selection] - Mot Residuali    : [ACTIVATED]')
    print('   * [Model] - Summarization Method      : %s' % opts.e_summary_metric)
    if opts.e_summary_metric == 'ridge':
       print('   * [Model] - Alpha for Ridge Regression      : %f' % opts.ridge_alpha)
    if opts.randomize_behavior:
       print('   * Randomization of Behavior Selected --> ####### Creating a set of null results #####')
    assert (opts.confounds == False) | (opts.confounds == True) & (osp.exists(opts.confounds_path)), "++ ERROR [main]: Confound regression requested, but provided confound file does not exists."
    if opts.confounds:
        print('   * Confound Regression                 : [ACTIVATED]')
        print('   * Confound File                       : %s' % opts.confounds_path)
    else:
        print('   * Confound Regression                 : [NOT ACTIVATED]')
    print('++ ----------------------------------------------------------')

    # Read Inputs
    # =========== 
    # 1. Load FC vectors into memory
    print("++ INFO [main]: Loading [%s] FC matrices from %s" % (opts.fc_label, opts.fc_path))
    fc_data         = pd.read_csv(opts.fc_path,index_col=['Subject','Run'])
    Nscans, n_edges = fc_data.shape
    fc_data.columns = range(n_edges) # Needed to ensure the proper type on the column ids
    print('++ INFO [main]: FC data loaded into memory [FC Label=%s, #Scans=%d, #Connections=%d]' % (opts.fc_label,Nscans,n_edges))

    # 2. Load Behaviors table into memory
    behav_data = pd.read_csv(opts.behav_path, index_col=['Subject','Run'])
    Nbehavs = behav_data.shape[1]
    print ('++ INFO [main]: Behaviors table loaded into memory [#Behaviors=%d]' % Nbehavs)

    # 3. Load Motion Confound information (if needed)
    if opts.confounds:
       print ('++ INFO [main]: Loading confounds for residalization')
       confounds = pd.read_csv(opts.confounds_path, index_col=['Subject','Run'])
       assert 'Mean Rel Motion' in confounds.columns, "++ ERROR: Confound file does not contain expected column Mean Rel Motion"
       assert fc_data.index.equals(confounds.index), "++ ERROR: Index of confound and input dataframes do not match" 

    # 4. Randomize Behavior if needed
    if opts.randomize_behavior:
       print('++            Randomizing behavior ######## ATTENTION ##########')
       behav_index = behav_data.index
       behav_data  = behav_data.sample(frac=1).reset_index(drop=True)
       behav_data  = behav_data.set_index(behav_index)

    # 5. Ensure there is a match in indexes between both data structures
    assert fc_data.index.equals(behav_data.index), "++ ERROR [main]:Index in FC dataFrame [%s] and behavior dataframe do not match." % opts.fc_label
    if opts.confounds:
       assert fc_data.index.equals(confounds.index), "++ ERROR [main]: Index in FC dataframe [%s] and motion confound do not match." % opts.fc_label
    
    # 6. Ensure selected behavior is available
    assert opts.behavior in behav_data.columns, "++ ERROR [cpm_wrapper]:behavior not present in behav_data"
    
    # 7. Extract a list of scan IDs (we rely on A becuase we have already checked that the indixes are the same)
    scan_list = fc_data.index
    print("++ INFO [main]: Total number of available scans = %d scans" % len(scan_list) )
    print("++ INFO [main]: Identified of first scan        = %s" % str(scan_list[0]))
    
    # Prepare for Cross-validation step
    # =================================
    print('++ INFO [main]: Generating lists of scans per k-fold [%s]' % opts.split_mode)
    if opts.split_mode == 'basic':
        indices = mk_kfold_test_indices(scan_list, random_seed=opts.repetition, k=opts.k, verb=opts.verbose)
    if opts.split_mode == 'subject_aware':
        indices = mk_kfold_test_indices_subject_aware(scan_list, random_seed=opts.repetition, k=opts.k, verb=opts.verbose)
    # Create dictionary with configurations
    cpm_kwargs = {'r_thresh': opts.e_thr_r,
                  'p_thresh': opts.e_thr_p,
                  'd_thresh': opts.e_thr_d,
                  'corr_type': opts.corr_type,
                  'verbose': opts.verbose,
                  'edge_summary_metric': opts.e_summary_metric,
                  'ridge_alpha':opts.ridge_alpha,
                  'confounds':opts.confounds}

    # Create output data structures
    # =============================
    all_masks = {}
    behav_obs_pred = None
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
        all_masks[("pos")]   = np.zeros((opts.k, n_edges))
        all_masks[("neg")]   = np.zeros((opts.k, n_edges))

    # Run Cross-validation
    # ====================
    print('++ INFO [main]: Run cross-validation')
    for fold in range(opts.k):
        print(f" +              doing fold {fold}/{opts.k}", end=' --> ')
        # Gather testing and training data for this particular fold
        print('split_train_test', end=' --> ')
        tr_scans, tt_scans              = split_train_test(scan_list, indices, test_fold=fold)
        print('get_train_test_data', end=' --> ')
    
        tr_vcts, tr_behav, tt_vcts = get_train_test_data(fc_data, tr_scans, tt_scans, behav_data, prediction_target=opts.behavior)
        tr_vcts  = tr_vcts.infer_objects()
        tt_vcts  = tt_vcts.infer_objects()
        tr_behav = tr_behav.infer_objects()
        tt_behav = behav_data.loc[tt_scans,opts.behavior].infer_objects()
    
        # Motion residualization if needed
        if opts.confounds:
            print('residualize motion: ',end='')
            tr_confounds, tt_confounds = get_confounds(tr_scans, tt_scans, confounds)
            tr_confounds  = tr_confounds.infer_objects()
            tt_confounds  = tt_confounds.infer_objects()
            tr_behav, confound_model = residualize(tr_behav, tr_confounds)
            tt_behav                 = tt_behav - confound_model.predict(tt_confounds)
            print(' --> ', end='')
    
        # Add observed behavior to the returned dataframe behav_obs_pred (need to do this here in case there was confound resid)
        behav_obs_pred.loc[tt_scans, opts.behavior + " observed"] = tt_behav
    
        print('select_features: ', end=' ')
        # RIDGE-BASED REGRESSION
        # ======================
        if opts.e_summary_metric == 'ridge':
            assert False,"++ ERROR: This part of the code needs to be tested before being used."
            # SELECT FEATURES: Find edges that correlate above threshold with behavior in this fold
            if opts.e_thr_d is not None:
                # If using a density based threshold
                feature_selection = SelectPercentile(f_regression,percentile=opts.e_thr_d).fit(tr_vcts, tr_behav)
                sel_features      = list(feature_selection.get_feature_names_out(input_features=tr_vcts.columns))
            else:
                # Otherwise (which in the case of ridge is based on p-value) ==> R value threshold not implemented for ridge
                f_stat, p_val     = f_regression(tr_vcts.values, tr_behav.values)
                sel_features      = list(np.where(p_val < opts.e_thr_p)[0])
            # Translate into a pd.Series of True and False with index being connectionID
            aux                        = pd.Series(False,index=tr_vcts.columns)
            aux[sel_features]          = True
            all_masks['ridge'][fold,:] = aux
            # Add information about positive and negative edges
            r_vals = r_regression(tr_vcts[sel_features], tr_behav.values)
            all_masks['pos'][fold,sel_features]   = [1 if x > 0 else 0 for x in r_vals]
            all_masks['neg'][fold,sel_features]   = [1 if x < 0 else 0 for x in r_vals]
            print('[{} pos + {} neg] -> {} features '.format(int(all_masks['pos'][fold,:].sum()),int(all_masks['neg'][fold,:].sum()),int(all_masks['ridge'][fold,:].sum())), end='--> ')
            # BUILD MODEL: Build Ridge Model
            print(f'[{fold}/{opts.k}] build_ridge_model', end=' --> ')
            model       = build_ridge_model(tr_vcts, all_masks['ridge'][fold,:], tr_behav, opts.ridge_alpha)
            # APPLY MODE: Apply Ridge model
            print(f'[{fold}/{opts.k}] apply_model', end='\n')
            predictions = apply_ridge_model(tt_vcts, all_masks["ridge"][fold,:], model)
            behav_obs_pred.loc[tt_scans, opts.behavior + " predicted (ridge)"] = predictions
    
        # TRADITIONAL CPM (OLS + Summary Statistic across selected edges)
        # ===============================================================
        else:
            # SELECT FEATURES
            print(f'\n', end='')
            mask_dict   = select_features(tr_vcts, tr_behav, **cpm_kwargs)
            print('', end='\n')
            # Gather the edges found to be significant in this fold in the all_masks dictionary
            all_masks["pos"][fold,:] = mask_dict["pos"]
            all_masks["neg"][fold,:] = mask_dict["neg"]
            # BUILD MODEL
            print('build_model', end=' --> ')
            model_dict = build_model(tr_vcts, mask_dict, tr_behav, edge_summary_method=opts.e_summary_metric)
            # APPLY MODEL
            print('apply_model', end='\n')
            behav_pred = apply_model(tt_vcts, mask_dict, model_dict, edge_summary_method=opts.e_summary_metric)
            # Update behav_obs_pred with results for this particular fold (the predictions for the test subjects only)
            for tail, predictions in behav_pred.items():
                behav_obs_pred.loc[tt_scans, opts.behavior + " predicted (" + tail + ")"] = predictions
            
    behav_obs_pred = behav_obs_pred.infer_objects()

    # Save Results to Disk
    # ====================
    # 1. Create list of objects to save
    cpm_outputs = {'cpm_kwargs':cpm_kwargs,
                   'models':all_masks,
                   'behav_obs_pred': behav_obs_pred,
                    'kfold_test_idx':indices}
    # 2. Create output path
    output_file = osp.join(opts.output_dir,opts.fc_label,'cpm_{beh}_rep-{rep}.pkl'.format(rep=str(opts.repetition).zfill(5), beh=opts.behavior))
    # 3. Dump to disk
    with open(output_file, 'wb') as f:
        pickle.dump(cpm_outputs, f)
    
    print('++ INFO [main]: Results written to disk [%s]' % (output_file))
    # Write final messages with info about prediction in this iteration
    # =================================================================
    if opts.e_summary_metric == 'ridge':
        print('++ INFO [main]: Summary of results for ridge regression')
        print('++ INFO [main]: R(obs,pred-%s) = %.3f' % ('ridge', behav_obs_pred[opts.behavior + " observed"].corr(behav_obs_pred[opts.behavior + " predicted (ridge)"])))
    else:
        print('++ INFO [main]: Summary of results for basic regression')
        for tail in ["pos","neg","glm"]:
            print('++ INFO [main]: R(obs,pred-%s,%s) = %.3f' % (tail,opts.fc_label, behav_obs_pred[opts.behavior + " observed"].corr(behav_obs_pred[opts.behavior + " predicted (" + tail + ")"])))

if __name__ == "__main__":
    main()
