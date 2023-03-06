import os.path as osp
import pandas as pd

# SNYCQ Information
SNYCQ_Questions = {'Positive':'I thought about something positive',
                  'Negative':'I thought about something negative', 
                  'Future':'I thought about future events',
                  'Past':'I thought about past events',
                  'Myself':'I thought about myself',
                  'People':'I thought about other people',
                  'Surroundings':'I thought about my present environment/surrounding',
                  'Vigilance':'I was completely awake',
                  'Images':'My thoughts were in the form of images',
                  'Words':'My thoughts were in the form of words',
                  'Specific':'My thoughts were more specific than vague',
                  'Intrusive':'My thoughts were intrusive'}

SNYCQ_Question_type = {'Positive': 'Content',
                       'Negative': 'Content',
                       'Future': 'Content',
                       'Past': 'Content',
                       'Myself': 'Content',
                       'People': 'Content',
                       'Surroundings': 'Content',
                       'Vigilance': 'Misc',
                       'Images': 'Form',
                       'Words': 'Form',
                       'Specific': 'Form',
                       'Intrusive': 'Form'}

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
PROC_SNYCQ_DIR          = osp.join(PRJ_DIR,'data','snycq')
RESOURCES_SNYCQ_DIR     = osp.join(PRJ_DIR,'code','fc_introspection','resources','snycq')
RESOURCES_NBS_DIR       = osp.join(PRJ_DIR,'code','fc_introspection','resources','nbs')
RESOURCES_CPM_DIR       = osp.join(PRJ_DIR,'code','fc_introspection','resources','cpm')

# QA Configuration
FINAL_NUM_VOLS     = 652 # Number of volumes in fully pre-processed runs
REL_MOT_THRESHOLD  = 0.2 # Maximum Relative Displacement (in mm)
MAX_CENSOR_PERCENT = 30  # Maximum percentage of censored volumes accepted per scan


CPM_NITERATIONS      = 100       # Number of iterations on real data (to evaluate robustness against fold generation)
CPM_NULL_NITERATIONS = 10000     # Number of iterations used to build a null distribution

# Paths to files we use often
# ===========================
ORIG_SNYCQ_PATH      = osp.join(PROC_SNYCQ_DIR, 'SNYCQ_Preproc.csv')                # Files with all SNYCQ answers compiled into a single dataframe
SBJS_WITH_SNYCQ_PATH = osp.join(RESOURCES_DINFO_DIR,'NC_withSNYCQ_subjects.txt')    # List of subjects with at least one rest scan with SNYCQ
SNYCQ_CLUSTERS_INFO_PATH =  osp.join(RESOURCES_SNYCQ_DIR,'SNYCQ_clusters_info.csv') # Information with scan membership for each cluster.
ANAT_PATHINFO_PATH   = osp.join(RESOURCES_DINFO_DIR,'NC_anat_info.csv')             # Where to find anatomical for each subject (sess-01 or sess-02)
BAD_STPROC_LIST_PATH = osp.join(PREPROCESSING_NOTES_DIR, 'NC_struct_fail_list.csv') # List of scans that failed structural pre-processing
BAD_FNPROC_LIST_PATH = osp.join(PREPROCESSING_NOTES_DIR, 'NC_func_fail_list.csv')   # List of scans that failed functional pre-processing
BAD_MOTION_LIST_PATH = osp.join(PREPROCESSING_NOTES_DIR,'NC_func_too_much_motion_list.csv') # List of scans with excessive motion

# Parcellation Configuration
# ==========================
ATLASES_DIR            = osp.join(PRJ_DIR,'atlases')
SUBCORTICAL_ATLAS_NAME          = 'aal2'                                       # This atlas only contains definitions for sub-cortical regions
SUBCORTICAL_ATLAS_PATH          = osp.join(ATLASES_DIR,SUBCORTICAL_ATLAS_NAME) 
SUBCORTICAL_BRAINNET_NODES_PATH = osp.join(RESOURCES_NBS_DIR,SUBCORTICAL_ATLAS_NAME,'{SUBCORTICAL_ATLAS_NAME}_BrainNet_Nodes.node'.format(SUBCORTICAL_ATLAS_NAME=SUBCORTICAL_ATLAS_NAME))

CORTICAL_200ROI_ATLAS_NAME    = 'Schaefer2018_200Parcels_7Networks'          # This atlas only contains definitions for cortical regions
CORTICAL_200ROI_ATLAS_PATH    = osp.join(ATLASES_DIR,CORTICAL_200ROI_ATLAS_NAME)
FB_200ROI_ATLAS_NAME          = 'Schaefer2018_200Parcels_7Networks_AAL2'     # This atlas is a combination of the cortical and subcortical atlas above
FB_200ROI_ATLAS_PATH          = osp.join(ATLASES_DIR,FB_200ROI_ATLAS_NAME)
CORTICAL_200ROI_BRAINNET_NODES_PATH = osp.join(RESOURCES_NBS_DIR,CORTICAL_200ROI_ATLAS_NAME,'{CORTICAL_ATLAS_NAME}_BrainNet_Nodes.node'.format(CORTICAL_ATLAS_NAME=CORTICAL_200ROI_ATLAS_NAME))
FB_200ROI_BRAINNET_NODES_PATH       = osp.join(RESOURCES_NBS_DIR,FB_200ROI_ATLAS_NAME,      '{FB_ATLAS_NAME}_BrainNet_Nodes.node'.format(FB_ATLAS_NAME=FB_200ROI_ATLAS_NAME))


CORTICAL_400ROI_ATLAS_NAME    = 'Schaefer2018_400Parcels_7Networks'          # This atlas only contains definitions for cortical regions
CORTICAL_400ROI_ATLAS_PATH    = osp.join(ATLASES_DIR,CORTICAL_400ROI_ATLAS_NAME)
FB_400ROI_ATLAS_NAME          = 'Schaefer2018_400Parcels_7Networks_AAL2'     # This atlas is a combination of the cortical and subcortical atlas above
FB_400ROI_ATLAS_PATH          = osp.join(ATLASES_DIR,FB_400ROI_ATLAS_NAME)
CORTICAL_400ROI_BRAINNET_NODES_PATH = osp.join(RESOURCES_NBS_DIR,CORTICAL_400ROI_ATLAS_NAME,'{CORTICAL_ATLAS_NAME}_BrainNet_Nodes.node'.format(CORTICAL_ATLAS_NAME=CORTICAL_400ROI_ATLAS_NAME))
FB_400ROI_BRAINNET_NODES_PATH       = osp.join(RESOURCES_NBS_DIR,FB_400ROI_ATLAS_NAME,      '{FB_ATLAS_NAME}_BrainNet_Nodes.node'.format(FB_ATLAS_NAME=FB_400ROI_ATLAS_NAME))


# Functions
# =========

def rgb2hex(r,g,b):
    """
    Converts between two different versions of RGB color codes. Input as three separate integers between 0 and 256.
    Output will be in hexadecimal code.
    """
    return "#{:02x}{:02x}{:02x}".format(r,g,b)
 
 
def get_sbj_scan_list(when='orig', return_snycq=True):
    """
    Get list of scans, subjects and snycq table at different pre-processing stages as selected via the when input.
    
    INPUTS:
    
    when: this can take the following values
        'orig' --> all available data
        'post_struct' --> only data that successfully completed the anatomical pre-processing pipeline.
        'post_funct' --> only data that successfully completed the functional pre-processing pipeline.
        'post_motion' --> only data that has low motion
        
    return_snycq: flag to select whether or not to return the corresponding sncq table
    
    OUTPUTS:
    
    sbj_list: list with subject ids
    scan_list: list with scan ids as a combination of subject_id and run_id
    snycq_df: dataframe with SNYCQ answers
    """
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