import os
import numpy as np
import time
import argparse
import pickle
import os.path as osp
import scipy
import sklearn

from datetime import datetime

from sklearn.model_selection import KFold

import warnings

def kfold_select(P, Krow, Kcol):
    block_list = []
    
    if (Krow > 1) & (Kcol > 1):
        krowf = KFold(n_splits=Krow, shuffle=True)
        kcolf = KFold(n_splits=Kcol, shuffle=True)
        for _, HOrow_idx in krowf.split(np.arange(P.shape[0]).astype('int')):
            for _, HOcol_idx in kcolf.split(np.arange(P.shape[1]).astype('int')):
                uHO = np.zeros(P.shape[0])
                uHO[HOrow_idx] = 1
                vHO = np.zeros(P.shape[1])
                vHO[HOcol_idx] = 1
                block = [uHO, vHO]
                block_list.append(block)
    if (Krow > 1) & (Kcol == 1):
        krowf = KFold(n_splits=Krow, shuffle=True)
        for _, HOrow_idx in krowf.split(np.arange(P.shape[0]).astype('int')):
            uHO = np.zeros(P.shape[0])
            uHO[HOrow_idx] = 1
            vHO = np.ones(P.shape[1])
            
            block = [uHO, vHO]
            block_list.append(block)
    if (Krow == 1) & (Kcol > 1):
        kcolf = KFold(n_splits=Kcol, shuffle=True)
        for _, HOcol_idx in kcolf.split(np.arange(P.shape[1]).astype('int')):
            uHO = np.ones(P.shape[0])
            vHO = np.zeros(P.shape[1])
            vHO[HOcol_idx] = 1
            block = [uHO, vHO]
            block_list.append(block)

    return block_list


    
def block_CV_mask(P, Krow, Kcol, J=10):
    
    block_list = kfold_select(P, Krow, Kcol)
    
    idx = np.arange(len(block_list))
    n = idx.shape[0]
    nc = int(np.ceil(n/J))
    if n % J != 0:
        warnings.warn('nblocks is not divisible!')
    np.random.shuffle(idx)
    
    mask_train_list = []
    mask_valid_list = []
    for j in range(J):
        block_idx = idx[j*nc:(j+1)*nc]
        
        mask_valid = np.zeros_like(P)
        for k in block_idx:
            block_vectors = block_list[k]
            mask_valid += np.outer(block_vectors[0], block_vectors[1])
        
        mask_valid[mask_valid > 1] == 1
        mask_train = 1 - mask_valid
        mask_train_list.append(mask_train)
        mask_valid_list.append(mask_valid)
        
    return mask_train_list, mask_valid_list
    
def random_CV_mask(P, J=10):
    idx = np.arange(P.shape[0]*P.shape[1])
    n = idx.shape[0]
    nc = int(np.ceil(n/J))
    np.random.shuffle(idx)
    mask_train_list = []
    mask_valid_list = []
    for j in range(J):
        mask_train = np.ones_like(P).reshape(-1)
        mask_train[idx[j*nc:(j+1)*nc]] = 0
        mask_valid = 1 - mask_train
        mask_train_list.append(mask_train.reshape(P.shape[0], P.shape[1]))
        mask_valid_list.append(mask_valid.reshape(P.shape[0], P.shape[1]))
    return mask_train_list, mask_valid_list



# PARSE INPUT ARGUMENTS
# =====================
parser = argparse.ArgumentParser()
parser.add_argument('-f', type=str, help="matrix file with key 'P' and 'data_mask' ", required=True)
parser.add_argument('-r', type=int, default=20, help="numbers of repeat")
parser.add_argument('-Kr', type=int, default=5, help="held out row number")
parser.add_argument('-Kc', type=int, default=5, help="held out column number")
parser.add_argument('-t', type=str, default="block", help="mask type: block or random")
parser.add_argument('-J', type=int, default=10, help="Split number of CV")
parser.add_argument('-o', type=str, default='./', help='output folder')
args = parser.parse_args()

# Extract input file path
print('=================================================')
datafile = args.f
print('++ INFO: Input datafile      = %s' % datafile)

# Extract outpuf folder
out_dir = args.o
print('++ INFO: Output folder       = %s' % out_dir)

# Extract and print all other parameter values
mask_type = args.t
print('++ INFO: Mask Type           = %s' % mask_type)
Krow=args.Kr
print('++ INFO: Held out row number = %d' % Krow)
Kcol=args.Kc
print('++ INFO: Held out col number = %d' % Kcol)
J = args.J
print('++ INFO: Split number of CV  = %d' % J)
repeat = args.r
print('++ INFO: Number of repeats   = %d' % repeat)
print('=================================================')

# LOAD INPUT DATA
data = np.load(datafile)
P = data['P']
print('++ INFO: P Matrix size    = %s' % str(P.shape))
data_mask = data['data_mask']
print('++ INFO: Mask Matrix size = %s' % str(data_mask.shape))

checkmask_start = time.time()

now = datetime.now()
dt_string = now.strftime("%d%m%Y-%H%M%S")

# Create path to output folders
mask_dir = osp.join(out_dir,'masks')
save_dir = osp.join(out_dir,'results')
print('++ INFO: Path to mask folder    = %s' % mask_dir)
print('++ INFO: Path to results folder = %s' % save_dir)
                        
if not os.path.exists(mask_dir):
    os.makedirs(mask_dir)
    print('++ INFO: mask folder created (it did not exist)')
    
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    print('++ INFO: results folder created (it did not exist)')

print('=================================================')
for r in range(repeat):
    print ('++ INFO: Repeat number = %d' % r)
    np.random.seed()
    
    if mask_type == 'block':
        if r==0:
           print('++ INFO: Working with mask type = BLOCK')
        mask_train_list, mask_valid_list = block_CV_mask(P, Krow=Krow, Kcol=Kcol, J=J)
    if mask_type == 'random':
        if r==0:
           print('++ INFO: Working with mask type = RANDOM')
        mask_train_list, mask_valid_list = random_CV_mask(P, J=J)
    for j in range(len(mask_train_list)):
        filename = os.path.join(mask_dir, "masks-{}-{}.npz".format(r, j))
        if not os.path.exists(filename):
            np.savez(filename, mask_train=mask_train_list[j], mask_valid=mask_valid_list[j])
            print( 'created mask list {}.'.format(filename) )
        
checkmask_end = time.time()
checkmask_time = checkmask_end-checkmask_start
