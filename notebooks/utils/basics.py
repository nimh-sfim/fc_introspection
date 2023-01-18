import os.path as osp

# Top of directory structure
BIOWULF_SHARE_FOLDER = '/data/SFIMJGC_Introspec/'
PRJ_DIR         = osp.join(BIOWULF_SHARE_FOLDER,'2023_fc_introspection')

# Folders with original data (fmri and behav)
ORIG_FMRI_DIR        = '/data/DSST/MPI_LEMON/ds000221-download/'
ORIG_BEHAV_DIR  = osp.join(PRJ_DIR,'downloads','behavioral')
