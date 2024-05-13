import os.path as osp
import pandas as pd
from sfim_lib.io.afni import load_netcc
import numpy as np
from scipy.spatial.distance import squareform
from tqdm import tqdm

def read_fc_matrices(scan_list,data_dir,atlas_name,pb_folder,fisher_transform=False):
    """Read all FC matrices into memory and return a pandas dataframe where
    each row contains a vectorized version of the FC matrix of a given scan.
    
    INPUTS:
    =======
    scan_list: list of tuples (subj_id,run_id) with information about all the scans to enter the analyses.
    
    data_dir: path to the location where data was pre-processed.
    
    atlas_name: name of the atlas used to extract the representative timseries and FC matrices.
    
    pb_folder: pbXX folder in the pdn folder. possible values: 'pb06_staticFC','pb07_qpp'

    fisher_transform: whether or not to do fisher transform of the correlation values.
    
    OUTPUT:
    =======
    df : pd.DataFrame (#rows = # unique scans, #columns = # unique connections)
    """
    # Load One Matrix to obtain number of connections
    sbj,run    = scan_list[0]
    _,_,_,_,run_num,_,run_acq = run.split('-')
    netcc_path = osp.join(data_dir,'PrcsData',sbj,'preprocessed','func',pb_folder,'{run_acq}_run-{run_num}.{ATLAS_NAME}_000.netcc'.format(run_acq = run_acq, run_num = run_num, ATLAS_NAME = atlas_name))
    netcc      = load_netcc(netcc_path)
    Nrois      = netcc.shape[0]
    Nconns     = int(Nrois*(Nrois-1)/2)
    # Initialize data structure that will hold all connectivity matrices
    df = pd.DataFrame(index=pd.MultiIndex.from_tuples(scan_list, names=['Subject','Run']), columns=range(Nconns))
    # Load all FC and place them into the final dataframe
    for sbj,run in tqdm(scan_list, desc='Scan'):
        _,_,_,_,run_num,_,run_acq = run.split('-')
        netcc_path = osp.join(data_dir,'PrcsData',sbj,'preprocessed','func',pb_folder,'{run_acq}_run-{run_num}.{ATLAS_NAME}_000.netcc'.format(run_acq = run_acq, run_num = run_num, ATLAS_NAME = atlas_name))
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
