import argparse
import os
import os.path as osp
import pickle
import pandas as pd
import numpy as np
from cpm import mk_kfold_indices, split_train_test, get_train_test_data, select_features, build_model, apply_model, mk_kfold_indices_subject_aware, build_ridge_model, apply_ridge_model

def read_command_line():
    parser = argparse.ArgumentParser(description='Run the CPM algorithm given a set of FC matrices (vectorized) and an external quantity (e.g., behavior) to be predicted.')
    parser.add_argument('-b','--behav_path',          type=str,   help="Path to dataframe with behaviors",             required=True, dest='behav_path')
    parser.add_argument('-f','--fc_path',             type=str,   help="Path to dataframe with fc vectors",            required=True, dest='fc_path')
    parser.add_argument('-t','--target_behav',        type=str,   help="Target behavior to predict",                   required=True, dest='behavior')
    parser.add_argument('-k','--number_of_folds',     type=int,   help="Number of cross-validation folds",             required=False, default=10, dest='k')
    parser.add_argument('-o','--output_dir',          type=str,   help="Output folder where to write results",         required=True, dest='output_dir')
    parser.add_argument('-i','--iter',                type=int,   help="Repetition/Iteration number",                  required=False, default=0, dest='repetition')
    parser.add_argument('-p','--edge_threshold_p',    type=float, help="Threshold for edge-selection (as p-value)",    required=False, default=None, dest='e_thr_p')
    parser.add_argument('-r','--edge_threshold_r',    type=float, help="Threshold for edge-selection (as r-value)",    required=False, default=None, dest='e_thr_r')
    parser.add_argument('-c','--corr_type',           type=str,   help="Correlation type to use in edge-selection",    required=False, default='spearman', choices=['spearman','pearson'], dest='corr_type')
    parser.add_argument('-s','--edge_summary_metric', type=str,   help="How to summarize across edges in final model", required=False, default='sum',      choices=['sum','mean','ridge'], dest='e_summary_metric')
    parser.add_argument('-m','--split_mode',          type=str,   help="Type of data split for k-fold",                required=False, default='basic',    choices=['basic','subject_aware'], dest='split_mode')
    parser.add_argument('-n','--randomize_behavior', action='store_true', help="Randomized behavioral values for creating null distibutions", dest='randomize_behavior', required=False)
    parser.add_argument('-v','--verbose',            action='store_true', help="Show additional information on screen",                       dest='verbose', required=False)
    parser.add_argument('-a','--ridge_alpha', type=float, help='Alpha parameter for ridge regression', required=False, default=1.0, dest='ridge_alpha')
    parser.set_defaults(verbose=False)
    parser.set_defaults(randomize_behavior=False)
    return parser.parse_args()

def main():
    opts = read_command_line()     
    print('++ INFO [main]: Command Line Configuration ...............................................')
    print('   * Behavioral DataFrame Path           : %s' % opts.behav_path)
    assert osp.exists(opts.behav_path),'++ ERROR [main]: Behavioral Dataframe not found.'
    print('   * FC Dataframe Path                   : %s' % opts.fc_path)
    assert osp.exists(opts.fc_path), '++ ERROR [main]: FC Dataframe not found.'
    print('   * Output Folder                       : %s' % opts.output_dir, end=' | ')
    if osp.exists(opts.output_dir):
       print('[ALREADY EXISTS]')
    else:
       os.makedirs(opts.output_dir)
       print('[JUST CREATED]')
    print('   * Behavior to predict                 : %s' % opts.behavior)
    print('   * Number of folds                     : %d folds' % opts.k)
    print('   * Repetition Number                   : %d' % opts.repetition)
    print('   * [Cross-validation] Spliting mode    : %s' % opts.split_mode)

    assert ((opts.e_thr_p is not None) & (opts.e_thr_r is None)) | ((opts.e_thr_p is None) & (opts.e_thr_r is not None)), '++ ERROR [main]: Edge Thesholding must be set either as a p-value or R. You provided both. Program will exit.'
    if opts.e_thr_p is not None:
       print('   * [Edge Selection] - Threshold        : p < %.3f'% opts.e_thr_p)
    if opts.e_thr_r is not None:
       print('   * [Edge Selection] - Threshold        : R > %.3f' % opts.e_thr_r)
    print('   * [Edge Selection] - Correlation Type : %s' % opts.corr_type)
    print('   * [Model] - Summarization Method      : %s' % opts.e_summary_metric)
    if opts.e_summary_metric == 'ridge':
       print('   * [Model] - Alpha for Ridge Regression      : %f' % opts.ridge_alpha)
    if opts.randomize_behavior:
       print('   * Randomization of Behavior Selected --> ####### Creating a set of null results #####')
          
    # Read Inputs
    # ===========
   
    # 1. Load FC vectors into memory
    fc_data         = pd.read_csv(opts.fc_path,index_col=['Subject','Run'])
    Nscans, n_edges = fc_data.shape
    fc_data.columns = range(n_edges) # Needed to ensure the proper type on the column ids
    print('++ INFO [main]: FC data loaded into memory [#Scans=%d, #Connections=%d]' % (Nscans,n_edges))
    
    # 2. Load Behaviors table into memory
    behav_data = pd.read_csv(opts.behav_path, index_col=['Subject','Run'])
    Nbehavs = behav_data.shape[1]
    print ('++ INFO [main]: Behaviors table loaded into memory [#Behaviors=%d]' % Nbehavs)
   
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
                  'edge_summary_metric': opts.e_summary_metric,
                  'ridge_alpha':opts.ridge_alpha}
    
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
    # Save Results to Disk
    # ====================
    cpm_outputs = {'cpm_kwargs':cpm_kwargs,
                   'models':all_masks,
                   'behav_obs_pred': behav_obs_pred}
    # if opts.randomize_behavior:
    #     output_file = osp.join(opts.output_dir,'cpm_{beh}_rep-{rep}_NULL.pkl'.format(rep=str(opts.repetition).zfill(5), beh=opts.behavior))
    # else:
    #     output_file = osp.join(opts.output_dir,'cpm_{beh}_rep-{rep}.pkl'.format(rep=str(opts.repetition).zfill(5), beh=opts.behavior))
    output_file = osp.join(opts.output_dir,'cpm_{beh}_rep-{rep}.pkl'.format(rep=str(opts.repetition).zfill(5), beh=opts.behavior))
    with open(output_file, 'wb') as f:
        pickle.dump(cpm_outputs, f)
    print('++ INFO [main]: Results written to disk [%s]' % output_file)
    if opts.e_summary_metric == 'ridge':
        print('++ INFO [main]: R(obs,pred-%s) = %.3f' % ('ridge', behav_obs_pred[opts.behavior + " observed"].corr(behav_obs_pred[opts.behavior + " predicted (ridge)"])))
    else:
        for tail in ["pos","neg","glm"]:
            print('++ INFO [main]: R(obs,pred-%s) = %.3f' % (tail, behav_obs_pred[opts.behavior + " observed"].corr(behav_obs_pred[opts.behavior + " predicted (" + tail + ")"])))
    
if __name__ == "__main__":
    main()
