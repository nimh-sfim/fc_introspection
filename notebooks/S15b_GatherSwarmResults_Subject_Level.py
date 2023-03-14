import argparse
import os
import pickle 
import os.path as osp
import xarray as xr
import pandas as pd
from tqdm import tqdm
from utils.basics import get_sbj_scan_list

def read_command_line():
    parser = argparse.ArgumentParser(description='This program reads the results from all swarm jobs for a given configuration and saves them into a single pickle file')
    parser.add_argument('-i','--input_folder', type=str, help='Path to folder with results generated via swarm jobs', required=True, dest='input_path')
    parser.add_argument('-o','--output_path',  type=str, help='Path to output pickle file', required=True, dest='output_path')
    parser.add_argument('-n','--niter', type=int, help='Number of iterations per target', required=True, dest='niter')
    #parser.add_argument('-c','--corr_type', type=str, help='Metric to evaluate relation between observed / predicted in final model eval', required=True, dest='corr_type', choices=['pearson','spearman'])
    parser.add_argument('-t','--target', type=str, help='Use if you want to only load files for an specific target', required=False, dest='target', default=None)
    parser.add_argument('-T','--test_only', help='Do not load files, just check for their existence', dest='test', required=False, action='store_true')
    parser.add_argument('-v','--verbose', help='Show additional information', required=False, dest='verbose', action='store_true')
    parser.set_defaults(test=False)
    parser.set_defaults(verbose=False)
    return parser.parse_args()

def main():
    next_swarm_grep = 'grep'
    opts = read_command_line()
    CPM_NITERATIONS = opts.niter
    #CORR_TYPE = opts.corr_type
    print('++ INFO [main]: Input path  = %s' % opts.input_path)
    print('++ INFO [main]: Output file = %s' % opts.output_path)
    print('++ INFO [main]: Number of iterations = %d' % CPM_NITERATIONS)
    if opts.test:
        print('++ INFO [main]: Test only')
    # Grab list of available targets
    if opts.target is None:
        TARGETS = next(os.walk(opts.input_path))[1]
    else:
        TARGETS = [opts.target]
    
    print('++ INFO [main]: Number of targets = %d' % len(TARGETS))
    print('++ INFO [main]: Targets = %s' % str(TARGETS))   
    num_missing = {t:0 for t in TARGETS} 
    # Load list of scans and sessions
    sbj_list, scan_list = get_sbj_scan_list(when='post_motion', return_snycq=False)
    
    # Load reference file
    REF_TARGET = 'Images'
    ref_path = osp.join(opts.input_path,REF_TARGET,'cpm_{b}_rep-{r}.pkl'.format(b=REF_TARGET,r=str(1).zfill(5)))
    with open(ref_path,'rb') as f:
        data = pickle.load(f)
    type_cols = list(data['behav_obs_pred'].columns)
    type_cols = [s.split(' ',1)[1] for s in type_cols]
    print('++ INFO [main]: Type coordinates in predictions_xr: %s' % str(type_cols)) 
    # Create xr.DataArray to hold all data
    predictions_xr = xr.DataArray(dims   = ['Behavior','Iteration','Subject','Type'], 
                                  coords = {'Behavior':TARGETS, 
                                            'Iteration':range(CPM_NITERATIONS), 
                                            'Subject':sbj_list,
                                           'Type':list(type_cols)})
       
    for TARGET in TARGETS:
        for r in tqdm(range(CPM_NITERATIONS), desc='Iteration [%s]' % TARGET):
            path = osp.join(opts.input_path,TARGET,'cpm_{b}_rep-{r}.pkl'.format(b=TARGET,r=str(r+1).zfill(5)))
            if opts.test is True:
                if not osp.exists(path):
                    num_missing[TARGET] = num_missing[TARGET] + 1
                    if opts.verbose:
                        print('++ WARNING [test]: Missing file [%s]' % path)
                    next_swarm_grep = next_swarm_grep + ' -e "NUM_ITER={r} "'.format(r=r+1)
                continue
            try:
                with open(path,'rb') as f:
                    data = pickle.load(f)
            except:
               num_missing[TARGET] = num_missing[TARGET] + 1
               print('++ WARNING [main]: Missing file [%s]' % path) 
               continue
            # Access the DataFrame with the observed and predicted values
            pred = data['behav_obs_pred']
            # Save all observed and predicted values
            predictions_xr.loc[TARGET,r,:,'observed']        = pred[TARGET+' observed'].values # Kind of redundant, but handy later when averaging across dimensions
            for tail in ['pos','neg','glm','ridge']:
                col_name = 'predicted ('+tail+')'
                if TARGET + ' ' + col_name in pred.columns:
                    predictions_xr.loc[TARGET,r,:,col_name] = pred[TARGET+' '+col_name].values
    # Save to disk as a single structure to save time        
    print('++ INFO: Saving generated data structures to disk [%s]' % opts.output_path)
    #data_to_disk = {'real_pred_r':real_pred_r, 'predictions_xr':predictions_xr, 'real_pred':real_pred}
    with open(opts.output_path,'wb') as f:
        pickle.dump(predictions_xr,f)
    # Summary
    print('++ INFO: Summary of missing files per target....')
    print(num_missing)
    if opts.verbose:
        print('++ INFO: Grep command to select unfinished swarm jobs')
        print(next_swarm_grep)

if __name__ == "__main__":
    main()
