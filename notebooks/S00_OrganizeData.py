# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.12.0
#   kernelspec:
#     display_name: FC Instrospection (Jan 2023)
#     language: python
#     name: fc_introspection
# ---

# # Dataset Description
#
# ## s-NYCQ: When was it used?
#
# The s-NYCQ was administered in 7 different occasions: 
#
# * **Prior to entering the scanner**: 'pre-ses-02'
# * **After each of the 4 resting-state scans**: 'post-ses-02-run-02-acq-AP', 'post-ses-02-run-02-acq-PA', 'post-ses-02-run-01-acq-AP', 'post-ses-02-run-01-acq-PA'
# * **After a couple of computerized tasks**: ('post-ses-02-task-ETS','post-ses-02-task-CPTS').
#
# Here we will use only the questionnaires administered following each resting state scan.
#
# ## s-NYCQ: Content/Structure
#
# Desciption of the sNYCQ - as described in <a href='https://www.nature.com/articles/sdata2018307'>Mendes et al. (2019)</a>:
#
# * sNYCQ is similar to NYC-Q, but only includes 12 questions (instead of 31).
# * sNYCQ also attempts to measure the form and content of mind-wandering (as it is the case with the original version of the NYC-Q).
# * sNYCQ was administered digitally (while in the scanner) using a digital format of a scale bar:
#     * Resolution = increments of 5%
#     * 0% = "describes my thoughts not at all"
#     * 100% = "describes my thoughts completely"
#
# **<u>NOTE</u>**: Although <a href='https://www.nature.com/articles/sdata2018307'>Mendes et al. (2019)</a> mentions that this shorter version of the NYC-Q was previosly used by <a href='https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0077554'>Ruby et al. (2013)</a>, that is not exactly correct. Ruby et al. only used 9 questions (not 12 as in here), and their questions were about content and mood, which is an additional difference.
#
#
# ## MPI-Dataset: Subject / Scan counts
#
# * MPI Mind-Brain-Body Dataset total number of subjects: 318*
# * LEMON (ses-01) number of subjects: 228 subjects
# * N&C (ses-02) number of subjects: 199 subjects** 
#     * 194 subjects with 4 runs: (Not listed)
#     * 3   subjects with 3 runs: 010029, 010083, 010133
#     * 1   subjects with 2 runs: 010087
#     * 1   subjects with 1 runs: 010180
#     * 5   subjects with 0 scans: 010020, 010032, 010061, 010079 and 010081 (see note 2 below)
#
# **<u>NOTE 1</u>**: There is overlap between the two samples. That is why the total is not 228 + 199 = 427
#
# **<u>NOTE 2</u>**: Despite SNYCQ answers being available in the SNYCQ table for subjects 010020, 010032, 010061, 010079 and 010081; there is not fMRI data for session 02 for these subjects. The fact that SNYCQ answers exists suggest those scans were acquired (as the SNYCQ contains responses about the thoughts during the scans); yet the associated fMRI scans are nowhere to be found. 
#
# In summary: 
# * There is fMRI data for (194 sbjs * 4 scans/sbjs) + (3 sbjs * 3 scans/sbjs) + (1 sbj * 2 scans/sbj) + (1 sbj * 1scan/sbj) = 788 resting-state scans
# * <u>But not all scans have a completed sNYCQ. The questionnaire was only recorded following only 693 runs (out of the 788 available) </u> (see <a href='https://www.nature.com/articles/sdata2018307/tables/3'>Table 2 in Mendes et al. 2019</a>). Those are the ones being considered in this work
#
# ***

# # Description of this notebook
# This notebook helps us get a basic understanding of the components of the MPI dataset that will be used in this project. Here we are only interested in the NC subset becuase it contains the resting-state scans that were followed by the short NYCQ.
#
# This notebook will generate the following output files in the ```resources/dataset_info``` folder:
#
# * ```SNYCQ_Preproc.csv```: dataframe with responses to SNYCQ administered only following resting state scans. Subjects with no responses have been removed. Data re-sorted so there is one row per scan (693) and one column per question (12). New meaningful labels for each quesion.
# * ```NC_sbj_rest_and_snycq.txt```: text file with the list of subjects that have at least one resting-state scan with SNYCQ answers. These deserve attempting pre-processing of their anatomical data.
#
# * ```NC_anat_info.csv```: table with information regarding the location of the anatomical data for each subject. This is necessary becuase the anatomical scan was not acquired on the same session for all subjects.
#
# #### Notes/Facts to keep in mind
#
# This notebook assumes you have a local copy of the NC portion of the Leipzing Mind-Brain-Body dataset located in folder ```ORIG_FMRI_DIR```
#
# First, it loads both the SNYCQ.tsv and SNYCQ.json files to extract high level information such as: number of subjects and scans with SNYCQ responses, ID of different questionaire adminstrations, ID of questions, the questions themselves, etc.

import pandas as pd
import json
import os.path as osp
from   collections import OrderedDict

from utils.basics import PRJ_DIR, ORIG_BEHAV_DIR, ORIG_FMRI_DIR, RESOURCES_DINFO_DIR

# ***
# ## 1. Download the behavioral data from the link in Mendes et al. 2019 paper
#
# Prior to running the next set of cells make sure to download the behavioral data associated with the MPI dataset from this location: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/VMJ6NV
#
# Next, unzip the contents of the downloaded file into ```ORIG_BEHAV_DIR```

# ## 2. Create paths to all files/folders of interest

# +
orig_files_dir      = osp.join(ORIG_BEHAV_DIR,'behavioral_data_MPILMBB','phenotype') # Path to the SNYCQ files
RESOURCES_DINFO_DIR = osp.join(PRJ_DIR,'resources/dataset_info')                     # Output path for this notebook

# Input Files
# ===========
snycq_data_path = osp.join(orig_files_dir,'SNYCQ.tsv')
snycq_json_path = osp.join(orig_files_dir,'SNYCQ.json')

# Output Files
# ============
snycq_proc_path     = osp.join(RESOURCES_DINFO_DIR,'SNYCQ_Preproc.csv')
final_sbj_list_path = osp.join(RESOURCES_DINFO_DIR,'NC_withSNYCQ_subjects.txt')
anat_info_path      = osp.join(RESOURCES_DINFO_DIR,'NC_anat_info.csv')

# + [markdown] tags=[]
# ***
# ## 3. Check basic Information in the sNYCQ downloaded files 
# -

print('++ Basic Information:')
print('++ =================')
snycq      = pd.read_csv(snycq_data_path,sep='\t')
snycq.set_index('participant_id', inplace=True, drop=True)
# 1) We drop the participant_id.1 column becuase it consist of duplicated subject IDs that do not apply to in-scanner data (internal email communication with authors)
snycq.drop(['participant_id.1'],axis=1,inplace=True)         # This ID corresponds to the data acquired outside the scanner. We can ignore it here.
# 2) Get subject IDs for all subjects in the MNI dataset (both components)
subjectIDs       = list(snycq.index)
# 3) List of all entries in the SNYCQ table (this include answers to administrations after tasks and pre-scanning that do not apply to this project)
metricIDs        = list(snycq.columns)
# 4) Get list of columns that contain answers only to SNYCQ administrations following resting-state scans (the ones of interest here)
rest_cols        = [metric for metric in metricIDs if 'run' in metric]
# 5) List of scan labels (all scans)
all_scanIDs      = list(OrderedDict.fromkeys([ s.split('_')[1] for s in metricIDs]))
# 6) List of scan labels for resting-state scans
rest_scanIDs     = [item for item in all_scanIDs if 'run' in item]
# 7) Labels per question (without the scan ID)
rest_questionIDs = list(OrderedDict.fromkeys([ s.split('_')[2] for s in metricIDs if 'run' in s]))
print(' + Number of Subjects                   : %d (of these only 199 are in the N&C study)' % len(subjectIDs))
print(' + Number of Metrics (columns)          : %d' % len(metricIDs))
print(' + Number of Rest-related metrics       : %d (12 questions * 4 rest scans = 48)' % len(rest_cols))
print(' + Number of Scans (per subject)        : %d' % len(all_scanIDs))
print(' + Number of Resting Scans (per subject): %d' % len(rest_scanIDs))
print(' + Scan IDs: %s' % all_scanIDs )
print(' + Rest Scan IDs     [%d] : %s' % (len(rest_scanIDs),rest_scanIDs ))
print(' + Rest Question IDs [%d]: %s' % (len(rest_questionIDs),rest_questionIDs))

# ***
# ## 4. Extract the description about when each questionaire took place

scan_desc = {}
print('++ When did the questionaire happened?')
print('++ ===================================')
sample_question_names = [s for s in metricIDs if 'positive' in s]
with open(snycq_json_path) as f:
    snycq_descr = json.load(f)
for item in sample_question_names:
    desc = snycq_descr[item]['Description'].split('\n')[1]
    item_name = item.split('_')[1]
    scan_desc[item_name] = desc
    print(' + %s --> %s' % (item_name, desc))

# ***
# ## 5. Extract the questions associated with each item of the questionaire

rest_question_desc = {}
print('++ Questions in the SNYC-Questionaire:')
print('++ ===================================')
sample_question_names = [s for s in metricIDs if 'SNYCQ_'+rest_scanIDs[0] in s]
for i,item in enumerate(sample_question_names):
    desc                         = snycq_descr[item]['Description'].split('\n')[0]
    question                     = item.split('_')[2]
    rest_question_desc[question] = desc
    print(' + [%d] %s --> %s' % (i+1,question, desc))

# ***
# ## 6. Reorganize the sNYCQ data obtained from the public repository
#
# The original way the data is organized, there is one entry per subject, and then there is one column per question/per administration. So there are as many rows as subjects, but there are more columns than questions in a questionaire, becuase the same quesion will have multiple columns (as many times as the number of times the questionaire was administered). 

snycq.head()

# We will create a new version that has the following structure:
#
# * The newly created dataframe will contain one row per run.
#
# * The index is a multi-index with subject and run name
#
# * The column names will be the identifiers of each of the questions

snycq_data = pd.DataFrame(index=pd.MultiIndex.from_product([subjectIDs,rest_scanIDs],names=['Subject','Run']), columns=rest_questionIDs)
for subject in subjectIDs:
    for run in rest_scanIDs:
        for question in rest_questionIDs:
            snycq_data[question][(subject,run)] = snycq['SNYCQ_'+run+'_'+question][subject]
snycq_data.dropna(inplace=True)

# There is a spelling error on the work surroundings on the original version of the SNYCQ data. To correct that, we provide again the list of questions with correct spelling

snycq_data.columns = ['Positive','Negative','Future','Past','Myself','People','Surroundings','Vigilance','Images','Words','Specific','Intrusive']

snycq_data.head(10)

sbj_list = list(snycq_data.index.get_level_values('Subject').drop_duplicates())
print('++ INFO: Number of subjects with at least one run with SNYC data: %d' % len(sbj_list))

# ***
# ## 7. Save the newly formated questionaire answers to disk

print ("++ INFO: Saving snycq_data to disk [%s]." % snycq_proc_path)
snycq_data.to_csv(snycq_proc_path)

# + [markdown] tags=[]
# ***
#
# ## 8. Write to disk a list of subjects with at least one valid rest + SNYCQ run
#
# This is the list of subjects, for which according to the SNYCQ there is data of interest for us to analyze. 
#
# It is only for these subjects that we will attempt the pre-processing of their anatomical and functional data. 
#
# Those analyses may fail for a subset of them (e.g., missing data, freesurfer errors). We will remove the entries for these "bad" scans from the SNYCQ dataframe in a later notebook prior to any analyses of the SNYCQ data.
# -

# List of subjects with at least one resting-state run that has sNYCQ
# For these subjects, I will need to run the structural pre-processing pipeline
subjects_to_analyze = list(OrderedDict.fromkeys(snycq_data.index.get_level_values('Subject')))
print('++ INFO: Number of subjects with at least 1 rest + sNYCQ: %s subjects' % len(subjects_to_analyze))
with open(final_sbj_list_path, 'w') as filehandle:
    for listitem in subjects_to_analyze:
        filehandle.write('%s\n' % listitem)
print('++ INFO: Subject IDs available at [%s]' % final_sbj_list_path)

# ***
# ## 9. Get information about when was anat acquired
#
# As some subjects participated in both parts of the Mind-Body-Brain study, there are subjects for whom the anatomical data is located under ses-01, and for others under ses-02. 
#
# The following cell creates a new dataframe with information about the location of the anatomical scans for all 175 subjects. 
#
# We will use this information in ```S03_NC_run_structural``` to provide the correct anatomical input to the structural pre-processing pipeline.

anat_loc_df = pd.DataFrame(index=subjects_to_analyze,columns=['ses-01','ses-02','anat_path'])
anat_loc_df.index = anat_loc_df.index.rename('subject')
for sbj in subjects_to_analyze:
    path_ses01 = osp.join(ORIG_FMRI_DIR,sbj,'ses-01','anat')
    path_ses02 = osp.join(ORIG_FMRI_DIR,sbj,'ses-02','anat')
    if osp.exists(path_ses01):
        anat_loc_df['ses-01'][sbj] = True
        anat_loc_df['anat_path'][sbj] = path_ses01
    else:
        anat_loc_df['ses-01'][sbj] = False
    
    if osp.exists(path_ses02):
        anat_loc_df['ses-02'][sbj] = True
        anat_loc_df['anat_path'][sbj] = path_ses02
    else:
        anat_loc_df['ses-02'][sbj] = False
print('++ INFO: Total Number of Subjects: %d' % anat_loc_df.shape[0])
print(' + Number of subjects with anat in each session:')
print(anat_loc_df.sum())
print(' + Number of subjects with 2 anatomicals: %d' % anat_loc_df[anat_loc_df.drop('anat_path',axis=1).sum(axis=1)==2].shape[0])
anat_loc_df[anat_loc_df.drop('anat_path',axis=1).sum(axis=1)==2]
print('++ INFO: This information is now available at [%s]' % anat_info_path)
anat_loc_df.to_csv(anat_info_path)

anat_loc_df.head()
