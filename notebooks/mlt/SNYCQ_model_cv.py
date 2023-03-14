import os
import numpy as np
import time
import argparse
import pickle

#from ..utils.basics import PRJ_DIR
# Top of directory structure
import os.path as osp
BIOWULF_SHARE_FOLDER = '/data/SFIMJGC_Introspec/'
PRJ_DIR              = osp.join(BIOWULF_SHARE_FOLDER,'2023_fc_introspection')

import scipy
import sklearn
import os.path as osp
from cvxopt import matrix, solvers
from numpy.linalg import norm
from datetime import datetime

import sys
sys.path.append('../')

from NMF_algorithm import NMF_model


parser = argparse.ArgumentParser()
parser.add_argument('-f', type=str, help="matrix file with key 'P' and 'data_mask' ", required=True)
parser.add_argument('-d', type=int, help="embedding dimension", required=True)
parser.add_argument('-s', type=float, help="sparsity parameter", required=True)
parser.add_argument('-r', type=int, help="repeat ID", required=True)
parser.add_argument('-j', type=int, help="j-th CV ID", required=True)
args = parser.parse_args()


datafile = args.f

data = np.load(datafile)
P = data['P']
data_mask = data['data_mask']


dim = args.d
sparsity = args.s
r = args.r
j = args.j

embed_start = time.time()

mask_data = np.load(osp.join(PRJ_DIR,'data','snycq','masks','masks-{}-{}.npz'.format(r,j)))
mask_train = mask_data['mask_train']
mask_valid = mask_data['mask_valid']

now = datetime.now()
dt_string = now.strftime("%d%m%Y-%H%M%S")
filename = "cv-d{}-s{}-r{}-j{}-".format(dim, sparsity, r, j)+dt_string

model = NMF_model(P, data_mask, dim, method='admm',
                  Wbound=(True, 1.0),
                  sparsity_parameter=sparsity)

output = model.embed_holdout(mask_train, mask_valid)
# Javier: want to also have the cv and reps available, not only on the file name
output = [r,j]+output
# End of change by Javier.
embed_end = time.time()
embed_time = embed_end-embed_start


output_start = time.time()
output_path  = osp.join(PRJ_DIR,'data','snycq','results',filename+".npy")
np.save(output_path, output)
output_end = time.time() 
output_time = output_end-output_start
print('Embed time: {}, Output time: {}'.format(embed_time, output_time))
print('Output file written to disk: %s' % output_path)

