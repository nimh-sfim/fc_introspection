# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: FC Instrospection (2023 | 3.10)
#     language: python
#     name: fc_introspection_2023_py310
# ---

# # Description
#
# This notebook will look at the additional tests that participants completed. All tests can be grouped in the following categories:
#
# * Personality and Habituation Behaviors
# * Mind-wandering / Mindfulness
# * Synesthesia
# * Cognitive Control / Sustained Attention
# * Creativity

import bokeh
print(bokeh.__version__)

import pandas as pd
import numpy as np
import xarray as xr
import seaborn as sns
import matplotlib.pyplot as plt
import hvplot.pandas
import panel as pn
import holoviews as hv
import os.path as osp
from scipy.stats import pearsonr, spearmanr, ttest_ind, mannwhitneyu
from tqdm import tqdm
from utils.basics import ORIG_BEHAV_DIR, SNYCQ_CLUSTERS_INFO_PATH,RESOURCES_SNYCQ_DIR
from utils.basics import get_sbj_scan_list


# We will use this function to compute Cohen's d across scan sets

def cohens_d(x0, x1):
    m0, m1 = np.nanmean(x0), np.nanmean(x1)
    s0, s1 = np.nanstd(x0, ddof=1), np.nanstd(x1, ddof=1)
    n0, n1 = np.sum(~np.isnan(x0)), np.sum(~np.isnan(x1))
    sp = np.sqrt(((n0-1)*s0**2 + (n1-1)*s1**2) / (n0 + n1 - 2))
    return (m1 - m0) / sp if sp > 0 else np.nan


# We will use this function to show heatmaps with srtatistical significance marked via BOLD black outlines

def show_results(data_val,data_pval=None,pval_thr=0.05, clabel=None,height=600,width=700,cmap='RdBu_r', fontscale=1, clim=(-.7,.7)):

    # Get the bokeh plot hooks
    def highlight_cell_hook(plot, element):
        # This hook will be called during the plot creation process
        fig = plot.state  # The figure object
        
        # Coordinates of the cell to highlight (e.g., cell at x=5, y=3)
        for r,row in data_pval_indexed_long.iterrows():
            if row['pval']:
                highlight_x = row['col'] + 0.5
                highlight_y = row['index'] + 0.5
                fig.rect(x=highlight_x,y=highlight_y,width=1,height=1,line_color='black',line_width=2,fill_alpha=0,name='highlight')
                
    # Create basic heatmap
    data_val_long    = data_val.melt(ignore_index=False,var_name=data_val.columns.name,value_name=data_val.name).reset_index()
    data_val_long    = data_val_long[[data_val.columns.name,data_val.index.name,data_val.name]]
    data_val_heatmap = hv.HeatMap(data_val_long).opts(tools=['hover'], 
                                                      height=height, width=width,
                                                      fontscale=fontscale,cmap=cmap,clim=clim, line_color='k',line_width=.1).opts(xrotation=90, colorbar=True, clabel=clabel)
    # If not enough information to highlight statistical significance is provided then return basic heatmap
    if (data_pval is None) or (pval_thr is None):
        return data_pval_heatmap
    # Prepare pval df so that it has indexes and cols as integers --> these are the plotting coordinates
    data_pval_indexed      = (data_pval < pval_thr).reset_index(drop=True).T.reset_index(drop=True).T
    data_pval_indexed.columns.name = data_pval.columns.name
    data_pval_indexed.index.name   = data_pval.index.name
    data_pval_indexed.name         = data_pval.name
    data_pval_indexed_long = data_pval_indexed.melt(ignore_index=False,var_name=data_pval.columns.name,value_name=data_pval.name).reset_index()
    data_pval_indexed_long = data_pval_indexed_long[[data_pval.columns.name,data_pval.index.name,data_pval.name]]
    data_pval_indexed_long.columns = ['col','index','pval']

    data_val_heatmap_highlighted = data_val_heatmap.opts(hooks=[highlight_cell_hook])
    
    return data_val_heatmap_highlighted
    


# We will use this function to plot distribution of items across both scan Sets

def plot_distributions_across_NBS_groups(phenotype,df,groups_info, show_hist=True):
    # Extract gouping info:
    group_labels = list(groups_info.keys())
    group1       = groups_info[group_labels[0]]
    group2       = groups_info[group_labels[1]]
    # Check for missing subjects
    N_missing = 0
    for sbj in group1+group2:
        if sbj not in df.dropna().index:
            print(sbj,end=',')
            N_missing += 1
    print('\n++ Number of missing subjects for [%s] = %d' % (phenotype,N_missing))
    # Create dataframe for plotting
    df_NBS = df.copy()
    df_NBS = df_NBS.loc[group1+group2]
    df_NBS['Group'] = 'N/A'
    df_NBS.loc[group1,'Group'] = group_labels[0]
    df_NBS.loc[group2,'Group'] = group_labels[1]
    df_NBS = df_NBS.dropna()
    print(df_NBS['Group'].value_counts())
    print('============================')
    # Create plot
    if show_hist:
        fig = (df_NBS.hvplot.hist(by='Group',normed=True, alpha=0.5, bins=25) * df_NBS.hvplot.kde(by='Group')).opts(legend_position='top_right', title=phenotype)
    else:
        fig = df_NBS.hvplot.kde(by='Group').opts(legend_position='top_right', title=phenotype)
    # Perform statistics
    df_NBS = df_NBS.reset_index(drop=True).set_index('Group')
    print(ttest_ind(df_NBS.loc[group_labels[0]].values,df_NBS.loc[group_labels[1]].values))
    print(mannwhitneyu(df_NBS.loc[group_labels[0]].values,df_NBS.loc[group_labels[1]].values))
    return fig


# ***
#
# # 1. Create data structures with basic information about behavioral measures available as part of the Mind-Brain-Body Dataset
#
# 1. Here is a list of all available surveys organized by category
#
# This list was created taking into accoun that a few test are not available at the [download page](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/VMJ6NV). Those are:
#
# * FBI (Facebook intensity scale)
# * MGIQ (Multi-gender identity questionnaire)
#
# Also, for two CogControl task, there is the need for computing summary metrics, so we will ignored them by now
#
# * I found code for CCPT
# * Not sure what to do with ETS

# +
surveys_per_category = {
    'Personality':          ['AMAS','ASR','BDI','BISBAS','BCQ','BPS','SCS','ESS','GoldMSI','HADS','IAT','IMIS','MPU','MMI','NEO','PSSI','SE','SD3','SDS','TPS','UPPS'],
    'Mind-wandering':       ['FFMQ','MCQ','SDMW','VISQ'],
    'Synesthesia':          ['SYN'],
    'CogControl-Attention': ['ACS'],
    'Creativity':           ['AUT','CAQ','RAT','TCIA']
}
all_surveys =  [ x for xs in surveys_per_category.values() for x in xs ]

for survey in surveys_per_category:
    print('%s has %d surveys' % (survey, len(surveys_per_category[survey])))
# -

# The next dicitionary contains information about the different summary metrics available per survey

metrics_per_survey = {'AMAS':['AMAS_sum'],
                      'ASR':['ASR_summary_adaptiveFunctioning_friends_sum','ASR_summary_adaptiveFunctioning_spouse_sum','ASR_summary_adaptiveFunctioning_family_sum','ASR_summary_adaptiveFunctioning_job_sum','ASR_summary_adaptiveFunctioning_education_sum',
                           'ASR_scale_substanceUse_tabaco_perday', 'ASR_scale_substanceUse_alcohol_daysdrunk','ASR_scale_substanceUse_drugs_daysused','ASR_summary_criticalItems_sum', 
                           'ASR_summary_syndromeProfiles_anxiousdepressed_sum', 'ASR_summary_syndromeProfiles_withdrawn_sum','ASR_summary_syndromeProfiles_somaticComplaints_sum','ASR_summary_syndromeProfiles_thoughtProblems_sum',
                           'ASR_summary_syndromeProfiles_attentionProblems_sum','ASR_summary_syndromeProfiles_aggressiveBehavior_sum','ASR_summary_syndromeProfiles_rulebreakingBehavior_sum','ASR_summary_syndromeProfiles_intrusive_sum',
                           'ASR_summary_syndromeProfiles_internalizing_sum','ASR_summary_syndromeProfiles_externalizing_sum'],
                      'BDI':['BDI_summary_sum'],
                      'BISBAS':['BISBAS_BIS_sum','BISBAS_BAS_sum'],
                      'BCQ':['BCQ_private_body_mean', 'BCQ_public_body_mean','BCQ_body_competence_mean'],
                      'BPS':['BPS_sum'],
                      'SCS':['SCS_SelfCtrl_sum'],
                      'ESS':['ESS_summary_sum'],
                      'GoldMSI': ['GoldMSI_Active_sum', 'GoldMSI_Training_sum'],
                      'HADS':['HADS-A_summary_sum', 'HADS-D_summary_sum'],
                      'IAT':['IAT_sum'],
                      'IMIS':['IMIS_NegVal_sum', 'IMIS_Help_sum', 'IMIS_Movement_sum','IMIS_PersRef_sum'],
                      'MPU':['MPU_1','MPU_2','MPU_3','MPU_4','MPU_5','MPU_6','MPU_7','MPU_8','MPU_9','MPU_10','MPU_11','MPU_12','MPU_13','MPU_14','MPU_15','MPU_16','MPU_17','MPU_18','MPU_19'],
                      'MMI':['MMI_score'], 
                      'NEO':['NEO_N','NEO_E','NEO_O','NEO_A','NEO_C'],
                      'PSSI':['PSSI_PN', 'PSSI_SZ', 'PSSI_ST', 'PSSI_BL', 'PSSI_HI', 'PSSI_NA','PSSI_SU', 'PSSI_AB', 'PSSI_ZW', 'PSSI_NT', 'PSSI_DP', 'PSSI_SL','PSSI_RH', 'PSSI_AS'],
                      'SE':['SE_Mean_SelfEst'],
                      'SDS':['SDS_sum'],
                      'TPS':['TPS_D_sum'],
                      'UPPS':['UPPS_Mean_NegUrg', 'UPPS_Mean_Premed', 'UPPS_Mean_Persev','UPPS_Mean_SS', 'UPPS_Mean_PosUrg'],
                      'FFMQ':['FFMQ_observe_sum', 'FFMQ_describe_sum', 'FFMQ_act_awareness_sum','FFMQ_nonjudge_sum', 'FFMQ_nonreact_sum'],
                      'MCQ':['MCQ_lack_of_cogn_conf_mean', 'MCQ_pos_bel_about_worry_mean','MCQ_cogn_self-consc_mean', 'MCQ_neg_bel_about_uncontr_danger_mean','MCQ_need_contr_thoughts_mean'],
                      'SDMW':['S-D-MW_delib_mean', 'S-D-MW_spont_mean'],
                      'VISQ':['VIS_dialog_sum', 'VIS_condensed_sum', 'VIS_other_sum', 'VIS_eval_sum'],
                      'ACS':['ACS_sum'],
                      'AUT':['AUT_Fluency', 'AUT_creative_quality', 'AUT_Elaboration_mean', 'AUT_Average_Uniqueness'],
                      'CAQ':['CAQ_score'],
                      'RAT':['RAT_CORRECT_NR', 'RAT_PERCENT', 'RAT_Rtmeanforcorrectanswers'],
                      'TCIA':['TCIA_Vividness_mean', 'TCIA_Orig_mean', 'TCIA_Transform_mean']}

# 2. Gather the path to the files associated with each survey

SURVEYS_DIR = osp.join(ORIG_BEHAV_DIR,'behavioral_data_MPILMBB','phenotype')

# Create strucutres with access to the files associated with them
survey_data_paths = {}
survey_info_paths = {}
for test in tqdm(all_surveys, desc='Test'):
    data_file = osp.join(SURVEYS_DIR,f"{test}.tsv")
    info_file = osp.join(SURVEYS_DIR,f"{test}.json")
    if (not osp.exists(data_file)) or (not osp.exists(info_file)):
        print(f"++ WARNING: Cannot find both files for test {test}")
    else:
        survey_data_paths[test] = data_file
        survey_info_paths[test] = info_file

# ***
# # 2. Analysis regarding the role of dispositional traits in population differences (NBS results)
#
# 1. Load the list of all scans in the NBS groups

emb_plus     = pd.read_csv(osp.join(RESOURCES_SNYCQ_DIR, 'SNYCQ_tsne_embeddings_plus.csv'), index_col=[0,1])
clusters_info = emb_plus[['Set Label','Group Probability']]

# 2. Get the subjects in each group

# +
# All subjects entering NBS analyses
sbjs_in_SetA    = list(clusters_info[clusters_info['Set Label']=='Set A'].index.get_level_values('Subject').unique())
sbjs_in_SetB    = list(clusters_info[clusters_info['Set Label']=='Set B'].index.get_level_values('Subject').unique())
sbjs_in_Both    = list(set(sbjs_in_SetA).intersection(set(sbjs_in_SetB)))
NBS_all_sbjs    = sbjs_in_SetA + sbjs_in_SetB

print('++ Scans in NBS groups (this will include scans and subjects that are in both groups)')
print('++ Number of scans/subjects in group [Set A]: %d/%d' % (clusters_info[clusters_info['Set Label']=='Set A'].shape[0],len(sbjs_in_SetA)))
print('++ Number of scans/subjects in group [Set B]: %d/%d' % (clusters_info[clusters_info['Set Label']=='Set B'].shape[0],len(sbjs_in_SetB)))
print('++ Number of scans/subjects in both groups:   %d/%d' % (clusters_info[clusters_info['Set Label']!='Ambiguous'].shape[0],len(sbjs_in_Both)))
# -

# 3. Keep only subjects that are in one of the two sets only

# +
sbjs_in_SetB    = [item for item in sbjs_in_SetB if item not in sbjs_in_Both]
sbjs_in_SetA    = [item for item in sbjs_in_SetA if item not in sbjs_in_Both]
NBS_all_sbjs    = sbjs_in_SetA + sbjs_in_SetB

print('++ Number of scans/subjects in group [Set A]: %d' % len(sbjs_in_SetA))
print('++ Number of scans/subjects in group [Set B]: %d' % len(sbjs_in_SetB))
# -

# 4. For a few surveys, subjects name do not agree at all (most likely associagted with the other portion of the dataset that we do not analyze here). We identify such surveys by seeing there is zero overlap in subject labels between the subjects entering NBS and the subjects in the survey index. Such surveys are eliminated

# Look for surveys that do not include information about the subjects we are using
surveys_to_remove_insuficient_subjects = []
for survey in tqdm(all_surveys):
    df      = pd.read_csv(survey_data_paths[survey], sep='\t', index_col=0)
    df_sbjs = list(df.index.values)
    intersection_w_NBS = set(df_sbjs).intersection(NBS_all_sbjs)
    if (len(intersection_w_NBS) == 0) :
        surveys_to_remove_insuficient_subjects.append(survey)
print('++ Surveys that contain information about another subject group: %s' % surveys_to_remove_insuficient_subjects)

# Remove those surveys from the list of available surveys
aux = {}
for category,surveys in surveys_per_category.items():
    new_list = [i for i in surveys if i not in surveys_to_remove_insuficient_subjects]
    if len(new_list) > 0:
        aux[category] = new_list
NBS_surveys = [ x for xs in aux.values() for x in xs ]
print('++ New set of surveys:')
NBS_surveys_per_category = aux
for category,surveys in NBS_surveys_per_category.items():
    print(' + %s has %d surveys in it: %s' % (category, len(surveys), str(surveys)))

# 5. For the surveys still remaining, we will now remove all surveys that do not provide information in at least 70% of subjects in the NBS analyses

NBS_sbjs_avail_per_survey                   = {'N_avail':pd.DataFrame(columns=['NBS_SetA','NBS_SetB'])}
surveys_to_remove_insuficient_subjects_NBS  = []
percentage_needed_subjects = 0.7
for category, surveys in NBS_surveys_per_category.items():
    for survey in surveys:
        df         = pd.read_csv(survey_data_paths[survey], sep='\t', index_col=0)
        df         = df.loc[:, metrics_per_survey[survey]]
        avail_sbjs = list(df.dropna().index)
        num_avail_NBS_A = len(set(avail_sbjs).intersection(set(sbjs_in_SetA)))
        num_avail_NBS_B = len(set(avail_sbjs).intersection(set(sbjs_in_SetB)))
        NBS_sbjs_avail_per_survey['N_avail'].loc[survey,'NBS_SetA'] = num_avail_NBS_A
        NBS_sbjs_avail_per_survey['N_avail'].loc[survey,'NBS_SetB'] = num_avail_NBS_B
        if (num_avail_NBS_A < (len(sbjs_in_SetA) * percentage_needed_subjects)) or (num_avail_NBS_B < (len(sbjs_in_SetB) * percentage_needed_subjects)):
                surveys_to_remove_insuficient_subjects_NBS.append(survey)
surveys_to_remove_insuficient_subjects_NBS

# Remove those surveys from the list of available surveys
aux = {}
for category,surveys in NBS_surveys_per_category.items():
    new_list = [i for i in surveys if i not in surveys_to_remove_insuficient_subjects_NBS]
    if len(new_list) > 0:
        aux[category] = new_list
NBS_surveys = [ x for xs in aux.values() for x in xs ]
NBS_surveys_per_category = aux
print('++ New set of surveys:')
for category,surveys in NBS_surveys_per_category.items():
    print(' + %s has %d surveys in it: %s' % (category, len(surveys), str(surveys)))

# 6. Compute T-test and Mann-Whitney for all selected surveys

NBS_ttest = pd.DataFrame(columns=['T','pval'])
NBS_mw    = pd.DataFrame(columns=['U','pval'])
NBS_cohen_d = pd.DataFrame(columns=['d'])

for category,surveys in NBS_surveys_per_category.items():
    for survey in surveys:
        df_survey   = pd.read_csv(survey_data_paths[survey], sep='\t', index_col=0)
        for metric in tqdm(metrics_per_survey[survey],desc=survey):
            df                            = pd.DataFrame(df_survey[metric]) # In case there is more than one metric
            df['Group']                   = 'N/A'
            df.loc[sbjs_in_SetA,'Group']  = 'SetA'
            df.loc[sbjs_in_SetB,'Group']  = 'SetB'
            df                            = df.dropna()
            df                            = df.reset_index(drop=True).set_index('Group')
            set_a_values = df.loc['SetA'].values.squeeze()
            set_b_values = df.loc['SetB'].values.squeeze()
            NBS_ttest.loc[metric,'T'],NBS_ttest.loc[metric,'pval'] =    ttest_ind(set_a_values, set_b_values, alternative='two-sided')
            NBS_mw.loc[metric,'U'],   NBS_mw.loc[metric,'pval']    = mannwhitneyu(set_a_values, set_b_values, alternative='two-sided')
            NBS_cohen_d.loc[metric,'d'] = cohens_d(set_b_values,set_a_values)
NBS_ttest = NBS_ttest.infer_objects()
NBS_mw    = NBS_mw.infer_objects()

# +
pBonf_NBS = 0.05 / NBS_ttest.shape[0]
NBS_ttest['T (pBonf<0.05)'] = NBS_ttest['T']
NBS_ttest['T (p<0.05)'] = NBS_ttest['T']
NBS_ttest.loc[NBS_ttest['pval'] > pBonf_NBS, 'T (pBonf<0.05)'] = np.nan
NBS_ttest.loc[NBS_ttest['pval'] > 0.05, 'T (p<0.05)'] = np.nan

NBS_mw['U (pBonf<0.05)'] = NBS_mw['U']
NBS_mw['U (p<0.05)'] = NBS_mw['U']
NBS_mw.loc[NBS_mw['pval'] > pBonf_NBS, 'U (pBonf<0.05)'] = np.nan
NBS_mw.loc[NBS_mw['pval'] > 0.05, 'U (p<0.05)'] = np.nan
# -

NBS_cohen_d.hvplot.heatmap(height=600, fontscale=1,line_color='k',line_width=1, width=245, clim=(-2,2), cmap='RdBu_r').opts(xrotation=90, 
                                                                                                              clabel='Cohen`s d (Set A - Set B)', 
                                                                                                              shared_axes=False) + \
NBS_ttest.drop('pval',axis=1).hvplot.heatmap(height=600, fontscale=1,cmap='RdBu_r',clim=(-3,3), line_color='k',line_width=1, width=275).opts(xrotation=90, clabel='Paired T-stat', shared_axes=False) + \
NBS_mw.drop('pval',axis=1).hvplot.heatmap(height=600, fontscale=1,cmap='RdBu_r',line_color='k',line_width=1, width=275).opts(xrotation=90, clabel='Mann-Whitney U statistic')

NBS_ttest.drop(['T (pBonf<0.05)'],axis=1).dropna()

NBS_mw.drop(['U (pBonf<0.05)'],axis=1).dropna()

# ***
# # 2.  Role of dispositional traits in Predictive Analyses (CPMs)
#
# 1. Load the SNYCQ answers

_, _, snycq_df = get_sbj_scan_list(when='post_motion', return_snycq=True)
snycq_df.drop(['Vigilance'],axis=1,inplace=True)

# 2. Extract list of 469 scans entering the CPM analyses

# We do it this way so that we remove the two scans marked as outliers during the initial evaluation of the experiential data
scan_list = emb_plus.index


# 3. Load the ICQF Thought Pattern loadings

W = pd.read_csv('../resources/icqf/W.csv', index_col=[0,1])
W.columns = ['TP1','TP2']
W = W.loc[scan_list]

# 4. Update the ```SNYCA_df``` dataframe with the thought patterns loadings just loaded

snycq_df = pd.concat([snycq_df, W], axis=1)

# 5. Behavioral data is only available once per subject, so that we need a single measure of experiential data per subject --> we use the mean   

snyq_by_subject_df = snycq_df.groupby('Subject').mean()
sbj_list = list(snyq_by_subject_df.index)

# All diemsnions available to be checked
introspection_dimensions = list(snyq_by_subject_df.columns)

# Final list of subjects entering the CPM analyses
CPM_all_sbjs = list(snyq_by_subject_df.index.values)
print(len(CPM_all_sbjs))

# 6. Compute correlations between in-scanner experience measures and dispositional trait measures

CPM_R  = pd.DataFrame(columns=snycq_df.columns); CPM_R.name  = 'R'
CPM_Rp = pd.DataFrame(columns=snycq_df.columns); CPM_Rp.name = 'R_pval'
CPM_S  = pd.DataFrame(columns=snycq_df.columns); CPM_S.name  = 'S'
CPM_Sp = pd.DataFrame(columns=snycq_df.columns); CPM_Sp.name = 'S_pval'

# +
for category,surveys in NBS_surveys_per_category.items():
    for survey in surveys:
        df_survey      = pd.read_csv(survey_data_paths[survey], sep='\t', index_col=0)
        sbjs_in_survey = list(df_survey.dropna().index)
        sbjs_to_use    = list(set(sbj_list).intersection(set(sbjs_in_survey)))
        df_survey      = df_survey.loc[sbjs_to_use, metrics_per_survey[survey]]
        for metric in tqdm(metrics_per_survey[survey],desc=survey):
            df = pd.DataFrame(df_survey[metric]) # In case there is more than one metric
            df = df.dropna()
            for col in introspection_dimensions:
                CPM_R.loc[metric,col], CPM_Rp.loc[metric,col] = pearsonr(snyq_by_subject_df.loc[sbjs_to_use,col].values.squeeze(), df.loc[sbjs_to_use,metric].values.squeeze())
                CPM_S.loc[metric,col], CPM_Sp.loc[metric,col] = spearmanr(snyq_by_subject_df.loc[sbjs_to_use,col].values.squeeze(),df.loc[sbjs_to_use,metric].values.squeeze())
for aux in [CPM_R,CPM_Rp,CPM_S,CPM_Sp]:
    aux = aux.infer_objects()
    aux.index.name  = 'Other Traits'
    aux.columns.name = 'SNYCQ + Thought Patterns'
    
CPM_pBonf = 0.05 / CPM_R.shape[0]
print(CPM_pBonf)
# -

# 7. Plot the results based both in R and Spearman R

show_results(CPM_R,CPM_Rp,CPM_pBonf,clabel='Pearson Correlation (R) | pBONF < 0.05') + show_results(CPM_S,CPM_Sp,CPM_pBonf,clabel='Spearman Correlation (S) | pBONF < 0.05')
