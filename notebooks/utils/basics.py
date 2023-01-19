import os.path as osp

# Top of directory structure
BIOWULF_SHARE_FOLDER = '/data/SFIMJGC_Introspec/'
PRJ_DIR              = osp.join(BIOWULF_SHARE_FOLDER,'2023_fc_introspection')
DATA_DIR             = osp.join(BIOWULF_SHARE_FOLDER,'pdn')

# Folders with original data (fmri and behav)
ORIG_FMRI_DIR        = '/data/DSST/MPI_LEMON/ds000221-download/'
ORIG_BEHAV_DIR  = osp.join(PRJ_DIR,'downloads','behavioral')

# Code folders
NOTEBOOKS_DIR = osp.join(PRJ_DIR,'code','fc_introspection','notebooks')
SCRIPTS_DIR   = osp.join(PRJ_DIR,'code','fc_introspection','bash')
PREPROCESSING_NOTES_DIR = osp.join(PRJ_DIR,'code','fc_introspection','resources','preprocessing_notes')

# Resources folders
RESOURCES_DINFO_DIR = osp.join(PRJ_DIR,'resources/dataset_info')

# QA Configuration
FINAL_NUM_VOLS    = 652 # Number of volumes in fully pre-processed runs
REL_MOT_THRESHOLD = 0.2 # Maximum Relative Displacement (in mm)