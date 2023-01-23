import os.path as osp
import pandas as pd

# Top of directory structure
BIOWULF_SHARE_FOLDER = '/data/SFIMJGC_Introspec/'
PRJ_DIR              = osp.join(BIOWULF_SHARE_FOLDER,'2023_fc_introspection')
DATA_DIR             = osp.join(BIOWULF_SHARE_FOLDER,'pdn')

# Folders with original data (fmri and behav)
ORIG_FMRI_DIR   = '/data/DSST/MPI_LEMON/ds000221-download/'
ORIG_BEHAV_DIR  = osp.join(PRJ_DIR,'downloads','behavioral')

# Code folders
NOTEBOOKS_DIR           = osp.join(PRJ_DIR,'code','fc_introspection','notebooks')
SCRIPTS_DIR             = osp.join(PRJ_DIR,'code','fc_introspection','bash')
PREPROCESSING_NOTES_DIR = osp.join(PRJ_DIR,'code','fc_introspection','resources','preprocessing_notes')
RESOURCES_DINFO_DIR     = osp.join(PRJ_DIR,'code','fc_introspection','resources','dataset_info')
PROC_BEHAV_DIR          = osp.join(PRJ_DIR,'data','snycq')

# QA Configuration
FINAL_NUM_VOLS     = 652 # Number of volumes in fully pre-processed runs
REL_MOT_THRESHOLD  = 0.2 # Maximum Relative Displacement (in mm)
MAX_CENSOR_PERCENT = 30  # Maximum percentage of censored volumes accepted per scan

# Paths to files we use often
# ===========================
ORIG_SNYCQ_PATH      = osp.join(PROC_BEHAV_DIR, 'SNYCQ_Preproc.csv')                # Files with all SNYCQ answers compiled into a single dataframe
SBJS_WITH_SNYCQ_PATH = osp.join(RESOURCES_DINFO_DIR,'NC_withSNYCQ_subjects.txt')    # List of subjects with at least one rest scan with SNYCQ
ANAT_PATHINFO_PATH   = osp.join(RESOURCES_DINFO_DIR,'NC_anat_info.csv')             # Where to find anatomical for each subject (sess-01 or sess-02)
BAD_STPROC_LIST_PATH = osp.join(PREPROCESSING_NOTES_DIR, 'NC_struct_fail_list.csv') # List of scans that failed structural pre-processing
BAD_FNPROC_LIST_PATH = osp.join(PREPROCESSING_NOTES_DIR, 'NC_func_fail_list.csv')   # List of scans that failed functional pre-processing
BAD_MOTION_LIST_PATH = osp.join(PREPROCESSING_NOTES_DIR,'NC_func_too_much_motion_list.csv') # List of scans with excessive motion

# Parcellation Configuration
# ==========================
ATLASES_DIR = osp.join(PRJ_DIR,'atlases')
ATLAS_NAME  = 'Schaefer2018_200Parcels_7Networks'
ATLAS_PATH  = osp.join(ATLASES_DIR,ATLAS_NAME)

# Functions

def get_sbj_scan_list(when='orig', return_snycq=True):
    if when not in ['orig','post_struct','post_funct','post_motion']:
        print("++ ERROR: Wrong when argument, returning empty lists")
        return None,None
    snycq_df = pd.read_csv(ORIG_SNYCQ_PATH)
    snycq_df = snycq_df.set_index(['Subject','Run'])
    # Discard subjects that failed structural pre-porcessing
    # ======================================================
    if when == 'post_struct':
        bad_struct_sbj_df   = pd.read_csv(BAD_STPROC_LIST_PATH)
        bad_struct_sbj_list = list(bad_struct_sbj_df['Subject'].values)
        snycq_df            = snycq_df.drop(snycq_df.loc[bad_struct_sbj_list,:].index)
    # Discard subjects that failed functional pre-porcessing
    # ======================================================
    if when == 'post_funct':
        bad_struct_sbj_df   = pd.read_csv(BAD_STPROC_LIST_PATH)
        bad_struct_sbj_list = list(bad_struct_sbj_df['Subject'].values)
        snycq_df            = snycq_df.drop(snycq_df.loc[bad_struct_sbj_list,:].index)
        
        bad_func_scans_df   = pd.read_csv(BAD_FNPROC_LIST_PATH)
        bad_func_scans_midx = pd.MultiIndex.from_frame(bad_func_scans_df[['Subject','Run']])
        snycq_df = snycq_df.drop(bad_func_scans_midx)
    if when == 'post_motion':
        bad_struct_sbj_df   = pd.read_csv(BAD_STPROC_LIST_PATH)
        bad_struct_sbj_list = list(bad_struct_sbj_df['Subject'].values)
        snycq_df            = snycq_df.drop(snycq_df.loc[bad_struct_sbj_list,:].index)
        
        bad_func_scans_df   = pd.read_csv(BAD_FNPROC_LIST_PATH)
        bad_func_scans_midx = pd.MultiIndex.from_frame(bad_func_scans_df[['Subject','Run']])
        snycq_df = snycq_df.drop(bad_func_scans_midx)
        
        bad_moti_scans_df   = pd.read_csv(BAD_MOTION_LIST_PATH)
        bad_moti_scans_midx = pd.MultiIndex.from_frame(bad_moti_scans_df[['Subject','Run']])
        snycq_df = snycq_df.drop(bad_moti_scans_midx)
     
    sbj_list  = snycq_df.index.get_level_values('Subject').unique()
    scan_list = list(snycq_df.index)
    print ("++ [%s] Number of subjects: %d subjects" % (when,len(sbj_list)))
    print ("++ [%s] Number of scans:    %d scans"    % (when,len(scan_list)))
    if return_snycq:
     return sbj_list, scan_list, snycq_df
    else:
     return sbj_list, scan_list