# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Generic Kernel (2025a)
#     language: python
#     name: generic_2025a
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

# +
# Sometimes, bokeh does not render properly in jupyter notebooks. The code on the following cell helps resolve this issue

# allows visualisation in notebook
from bokeh.io import output_notebook
from bokeh.resources import INLINE
output_notebook(INLINE)
# -

surveys_per_category = {
    'Personality':          ['AMAS','ASR','BDI','BISBAS','BCQ','BPS','SCS','ESS','GoldMSI','HADS','IAT','IMIS','MPU','MMI','NEO','PSSI','SE','SD3','SDS','TPS','UPPS'],
    'Mind-wandering':       ['FFMQ','MCQ','SDMW','VISQ'],
    'Synesthesia':          ['SYN'],
    'CogControl-Attention': ['ACS'],
    'Creativity':           ['AUT','CAQ','RAT','TCIA']
}
all_surveys =  [ x for xs in surveys_per_category.values() for x in xs ]
#CPPT, ETS is a task

# There are a few test that are not available at the [download page](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/VMJ6NV). Those are:
#
# * FBI (Facebook intensity scale)
# * MGIQ (Multi-gender identity questionnaire)
#
# Also, for two CogControl task, there is the need for computing summary metrics, so we will ignored them by now
#
# * I found code for CCPT
# * Not sure what to do with ETS

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
from utils.basics import ORIG_BEHAV_DIR, SNYCQ_CLUSTERS_INFO_PATH
from utils.basics import get_sbj_scan_list


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
    


# # 1.  Load SNYCQ and its low dimensional representation

SURVEYS_DIR = osp.join(ORIG_BEHAV_DIR,'behavioral_data_MPILMBB','phenotype')

sbj_list, scan_list, snycq_df = get_sbj_scan_list(when='post_motion', return_snycq=True)
sbj_list = list(sbj_list)

factorization_path = './mlt/output/factorization/factorization_fulldata_confound.npz'
factorization_results = np.load(factorization_path)
W = pd.DataFrame(factorization_results['W'],index=snycq_df.index, columns=['TP1', 'TP2'])

snycq_df = pd.concat([snycq_df,W],axis=1)

snyq_by_subject_df = snycq_df.groupby('Subject').mean()

# All diemsnions available to be checked
introspection_dimensions = list(snyq_by_subject_df.columns)

# Final list of subjects entering the CPM analyses
CPM_all_sbjs = list(snyq_by_subject_df.index.values)

# # 2. Load information about subjects in NBS groups

clusters_info = pd.read_csv(SNYCQ_CLUSTERS_INFO_PATH, index_col=['Subject','Run'])

# All subjects entering NBS analyses
sbjs_in_Image_Pos_Others = list(clusters_info[clusters_info['Cluster Label']=='Image-Pos-Others'].index.get_level_values('Subject').unique())
sbjs_in_Surr_Neg_Self    = list(clusters_info[clusters_info['Cluster Label']=='Surr-Neg-Self'].index.get_level_values('Subject').unique())
sbjs_in_Both             = list(set(sbjs_in_Image_Pos_Others).intersection(set(sbjs_in_Surr_Neg_Self)))
print('++ Number of subjects in group [Image-Pos-Others]: %d' % len(sbjs_in_Image_Pos_Others))
print('++ Number of subjects in group [Surr-Neg-Self]:    %d' % len(sbjs_in_Surr_Neg_Self))
print('++ Number of subjects in both groups:              %d' % len(sbjs_in_Both))

# +
# Reduced list of subjects so that there are no subject in both groups
# -

sbjs_in_Surr_Neg_Self    = [item for item in sbjs_in_Surr_Neg_Self if item not in sbjs_in_Both]
sbjs_in_Image_Pos_Others = [item for item in sbjs_in_Image_Pos_Others if item not in sbjs_in_Both]
print('++ Number of subjects in group [Image-Pos-Others]: %d' % len(sbjs_in_Image_Pos_Others))
print('++ Number of subjects in group [Surr-Neg-Self]:    %d' % len(sbjs_in_Surr_Neg_Self))

# Get a list with all the subjects across both groups
NBS_groups   = {'Image-Pos-Others':sbjs_in_Image_Pos_Others, 'Surr-Neg-Self':sbjs_in_Surr_Neg_Self}
NBS_all_sbjs = sbjs_in_Image_Pos_Others + sbjs_in_Surr_Neg_Self


# ***

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


# # 2. Explore survey data so we can select the ones that have sufficient data

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

# +
# Look for surveys that do not include information about the subjects we are using
surveys_to_remove_insuficient_subjects = []
for survey in tqdm(all_surveys):
    df      = pd.read_csv(survey_data_paths[survey], sep='\t', index_col=0)
    df_sbjs = list(df.index.values)
    intersection_w_NBS = set(df_sbjs).intersection(NBS_all_sbjs)
    intersection_w_CPM = set(df_sbjs).intersection(CPM_all_sbjs)
    if (len(intersection_w_NBS) < len(NBS_all_sbjs)) or (len(intersection_w_CPM) < len(CPM_all_sbjs)):
        surveys_to_remove_insuficient_subjects.append(survey)


print('++ Surveys that contain information about another subject group: %s' % surveys_to_remove_insuficient_subjects)
# -

# Remove those surveys from the list of available surveys
new_dict = {}
for category,surveys in surveys_per_category.items():
    new_list = [i for i in surveys if i not in surveys_to_remove_insuficient_subjects]
    if len(new_list) > 0:
        new_dict[category] = new_list
surveys_per_category = new_dict
all_surveys = [ x for xs in surveys_per_category.values() for x in xs ]
print('++ New set of surveys:')
for category,surveys in surveys_per_category.items():
    print(' + %s has %d surveys in it: %s' % (category, len(surveys), str(surveys)))

# Different surveys have different entries. Some are individual responses, some are summary entries. Here we are providing the ones that are summary level as these are the ones we are interested in analyzing.

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

# We only want to keep surveys that have information in at leat 70% of relevant subjects. For that reason we first check how many subjects are available

results = {'N_avail':pd.DataFrame(columns=['NBS_Image-Pos-Others','NBS_Surr-Neg-Self','CPM'])}

metrics_to_remove_insuficient_subjects_NBS = []
metrics_to_remove_insuficient_subjects_CPM = []
percentage_needed_subjects = 0.7
for category, surveys in surveys_per_category.items():
    for survey in surveys:
        df         = pd.read_csv(survey_data_paths[survey], sep='\t', index_col=0)
        df         = df.loc[sbj_list, metrics_per_survey[survey]]
        avail_sbjs = list(df.dropna().index)
        for metric in metrics_per_survey[survey]:
            num_avail_NBS_IPO = len(set(avail_sbjs).intersection(set(sbjs_in_Image_Pos_Others)))
            num_avail_NBS_SNS = len(set(avail_sbjs).intersection(set(sbjs_in_Surr_Neg_Self)))
            num_avail_CPM     = len(set(avail_sbjs).intersection(set(CPM_all_sbjs)))
            results['N_avail'].loc[survey,'NBS_Image-Pos-Others'] = num_avail_NBS_IPO
            results['N_avail'].loc[survey,'NBS_Surr-Neg-Self']    = num_avail_NBS_SNS
            results['N_avail'].loc[survey,'CPM']                  = num_avail_CPM
            if (num_avail_NBS_IPO < (len(sbjs_in_Image_Pos_Others) * percentage_needed_subjects)) or (num_avail_NBS_SNS < (len(sbjs_in_Surr_Neg_Self) * percentage_needed_subjects)):
                metrics_to_remove_insuficient_subjects_NBS.append(metric)
            if num_avail_CPM < (len(CPM_all_sbjs) * percentage_needed_subjects):
                metrics_to_remove_insuficient_subjects_CPM.append(metric)

# Now we get the list of surveys that include at least 70% of the subjects used in the CPM analyses

surveys_for_CPM = list(results['N_avail'][results['N_avail']['CPM'] > len(CPM_all_sbjs) * .7].index)
print(surveys_for_CPM)

# Now we get the list of surveys that include at least 70% of the subjects used in the NBS analyses

surveys_for_NBS = list(  results['N_avail'][(results['N_avail']['NBS_Image-Pos-Others'] > len(sbjs_in_Image_Pos_Others) * .7) & (results['N_avail']['NBS_Image-Pos-Others'] > len(sbjs_in_Surr_Neg_Self) * .7)].index  )
print(surveys_for_NBS)

# Remove surveys with insuficient subjects
new_dict = {}
for category,surveys in surveys_per_category.items():
    new_list = [i for i in surveys if (i in surveys_for_CPM) & (i in surveys_for_NBS)]
    if len(new_list) > 0:
        new_dict[category] = new_list
surveys_per_category = new_dict
all_surveys = [ x for xs in surveys_per_category.values() for x in xs ]

# +
print('++ Final list of surveys entering the analyses:')
print('++ ============================================')

for category,surveys in surveys_per_category.items():
    print(' + %s has %d surveys in it: %s' % (category, len(surveys), str(surveys)))
# -

#
# # 3. Relationship between Phenotypes and NBS results

NBS_ttest = pd.DataFrame(columns=['T','pval'])
NBS_mw    = pd.DataFrame(columns=['U','pval'])

for category,surveys in surveys_per_category.items():
    for survey in surveys:
        df_survey   = pd.read_csv(survey_data_paths[survey], sep='\t', index_col=0)
        df_survey   = df_survey.loc[NBS_all_sbjs, metrics_per_survey[survey]]
        for metric in tqdm(metrics_per_survey[survey],desc=survey):
            df                                       = pd.DataFrame(df_survey[metric]) # In case there is more than one metric
            df['Group']                              = 'N/A'
            df.loc[sbjs_in_Image_Pos_Others,'Group'] = 'Image-Pos-Others'
            df.loc[sbjs_in_Surr_Neg_Self,'Group']    = 'Surr-Neg-Self'
            df                                       = df.dropna()
            df                                       = df.reset_index(drop=True).set_index('Group')
            NBS_ttest.loc[metric,'T'],NBS_ttest.loc[metric,'pval'] = ttest_ind(df.loc['Image-Pos-Others'].values.squeeze(),df.loc['Surr-Neg-Self'].values.squeeze(),alternative='two-sided')
            NBS_mw.loc[metric,'U'],   NBS_mw.loc[metric,'pval']    = mannwhitneyu(df.loc['Image-Pos-Others'].values.squeeze(),df.loc['Surr-Neg-Self'].values.squeeze(),alternative='two-sided')
NBS_ttest = NBS_ttest.infer_objects()
NBS_mw    = NBS_mw.infer_objects()

# +
pBonf_NBS = 0.05 / NBS_ttest.shape[0]
NBS_ttest['T (pBonf<0.05'] = NBS_ttest['T']
NBS_ttest['T (p<0.05)'] = NBS_ttest['T']
NBS_ttest.loc[NBS_ttest['pval'] > pBonf_NBS, 'T (pBonf<0.05'] = np.nan
NBS_ttest.loc[NBS_ttest['pval'] > 0.05, 'T (p<0.05)'] = np.nan

NBS_mw['U (pBonf<0.05'] = NBS_mw['U']
NBS_mw['U (p<0.05)'] = NBS_mw['U']
NBS_mw.loc[NBS_mw['pval'] > pBonf_NBS, 'U (pBonf<0.05'] = np.nan
NBS_mw.loc[NBS_mw['pval'] > 0.05, 'U (p<0.05)'] = np.nan
# -

NBS_ttest.drop('pval',axis=1).hvplot.heatmap(height=600, fontscale=1,cmap='RdBu_r',clim=(-3,3), line_color='k',line_width=1, width=275).opts(xrotation=90, clabel='Paired T-stat', shared_axes=False) + \
NBS_mw.drop('pval',axis=1).hvplot.heatmap(height=600, fontscale=1,cmap='RdBu_r',line_color='k',line_width=1, width=275).opts(xrotation=90, clabel='Mann-Whitney U statistic')

# # 4. Relationship between CPM and phenotypes

CPM_R  = pd.DataFrame(columns=snycq_df.columns); CPM_R.name  = 'R'
CPM_Rp = pd.DataFrame(columns=snycq_df.columns); CPM_Rp.name = 'R_pval'
CPM_S  = pd.DataFrame(columns=snycq_df.columns); CPM_S.name  = 'S'
CPM_Sp = pd.DataFrame(columns=snycq_df.columns); CPM_Sp.name = 'S_pval'

# +
for category,surveys in surveys_per_category.items():
    for survey in surveys:
        df_survey   = pd.read_csv(survey_data_paths[survey], sep='\t', index_col=0)
        df_survey   = df_survey.loc[CPM_all_sbjs, metrics_per_survey[survey]]
        for metric in tqdm(metrics_per_survey[survey],desc=survey):
            df                                       = pd.DataFrame(df_survey[metric]) # In case there is more than one metric
            df                                       = df.dropna()
            common_sbjs                              = list(df.index)
            for col in introspection_dimensions:
                CPM_R.loc[metric,col], CPM_Rp.loc[metric,col] = pearsonr(snyq_by_subject_df.loc[common_sbjs,col].values.squeeze(),df.values.squeeze())
                CPM_S.loc[metric,col], CPM_Sp.loc[metric,col] = spearmanr(snyq_by_subject_df.loc[common_sbjs,col].values.squeeze(),df.values.squeeze())
for aux in [CPM_R,CPM_Rp,CPM_S,CPM_Sp]:
    aux = aux.infer_objects()
    aux.index.name  = 'Phenotype'
    aux.columns.name = 'Thoughts'
    
CPM_pBonf = 0.05 / CPM_R.shape[0]
print(CPM_pBonf)
# -

show_results(CPM_R,CPM_Rp,CPM_pBonf,clabel='Pearson Correlation (R) | pBONF < 0.05') + show_results(CPM_S,CPM_Sp,CPM_pBonf,clabel='Spearman Correlation (S) | pBONF < 0.05')

show_results(CPM_R,CPM_Rp,0.05,clabel='Pearson Correlation (R) | p < 0.05') + show_results(CPM_S,CPM_Sp,0.05,clabel='Spearman Correlation (S) | p < 0.05')

# *** 
# # PREVIOUS ANALYSES - DEPRECATED, BUT SOME INFO MIGHT BE USEFUL
#
# ***
#
# ## 1.1 Abbreviated Math Anxiety Scale (AMAS). 
#
# The AMAS is a self-report inventory measuring the subjectively experienced level of anxiety in mathematical contexts. 
#
# It consists of nine items, related to the question “How anxious do you feel when …”, that can be scored on a five-point Likert scale (1 = “notat all” to 5 = “a lot”).
#
# > Reference: [The Abbreviated Math Anxiety Scale (AMAS): Construction, Validity, and Reliability](https://journals.sagepub.com/doi/10.1177/1073191103010002008)

survey = 'AMAS'
df     = pd.read_csv(survey_data_paths[survey], sep='\t', index_col=0)
df     = df.loc[sbj_list]
#pn.Row(pn.pane.DataFrame(df.sample(10)), df.hvplot.hist())

s = plot_distributions_across_NBS_groups('AMAS',df,NBS_groups)
s

common_sbjs = list(df.dropna().index) #.corrwith(snyq_by_subject_df.loc[list(df.dropna().index)],axis=0)
print("++ Number of subjects with data: %d / %d" % (len(common_sbjs),snyq_by_subject_df.shape[0]))

snycq_Rp_with_behavs

for col in introspection_dimensions:
    snycq_R_with_behavs.loc['AMAS',col], snycq_Rp_with_behavs.loc['AMAS',col] = pearsonr(snyq_by_subject_df.loc[common_sbjs,col].values,df.loc[common_sbjs,'AMAS_sum'].values)
    snycq_S_with_behavs.loc['AMAS',col], snycq_Sp_with_behavs.loc['AMAS',col] = spearmanr(snyq_by_subject_df.loc[common_sbjs,col].values,df.loc[common_sbjs,'AMAS_sum'].values)
    if snycq_Rp_with_behavs.loc['AMAS',col] < 0.05:
        print(f'AMAS vs. {col} | ',col,snycq_R_with_behavs.loc['AMAS',col], snycq_Rp_with_behavs.loc['AMAS',col])
    if snycq_Sp_with_behavs.loc['AMAS',col] < 0.05:
        print(f'AMAS vs. {col} | ',snycq_S_with_behavs.loc['AMAS',col], snycq_Sp_with_behavs.loc['AMAS',col])

# ## 1.2. Adult Self Report (ASR)
#
# The ASR assesses mental problems in adults between 18 and 59 years-old.
#
# It has four major scales related to the following topics: adaptive functioning, psychological syndromes, DSM-oriented problems, and substance use. 
#     
# Adaptive functioning comprises 36 items in the form of either a three or four-point Likert scale describing the quantity and quality of relationships, educationlevel, and job satisfaction. 
# Scales of psychological syndromes, DSM-oriented problems, and substance use comprise 126 items that can be scored on a three-point Likert scale (0 = “does not apply” to 2 = “exactly or does happen often”). 
# Two items were erroneously excluded (i.e., item 56.h “Heart pounding or racing”; item 56.i “Numbness ortingling in body parts”). These affect somatic complaints and internalizing subscales of the psychological syndromes scale. 
#
# Reference: [Ratings of Relations Between DSM-IV Diagnostic Categories and Items of the Adult Self-Report (ASR) and Adult Behavior Checklist (ABCL)](https://aseba.org/wp-content/uploads/2019/02/dsm-adultratings.pdf)
#
# > **NOTE: Don't know what to do with this one yet**

survey = 'ASR'
df     = pd.read_csv(survey_data_paths[survey], sep='\t', index_col=0)
ASR_adaptive_functioning_cols = [c for c in df.columns if 'adaptiveFunctioning' in c]
ASR_substance_use_cols        = [c for c in df.columns if 'substanceUse' in c]
ASR_syndrome_profiles_cols    = [c for c in df.columns if 'syndromeProfiles' in c]
ASR_critical_items_cols       = [c for c in df.columns if 'criticalItems' in c]
pn.Column(pn.Row(pn.pane.DataFrame(df.loc[sbj_list,ASR_adaptive_functioning_cols].sample(5),width=500), df.loc[sbj_list,ASR_adaptive_functioning_cols].hvplot.hist()),
          pn.Row(pn.pane.DataFrame(df.loc[sbj_list,ASR_substance_use_cols].sample(5),width=500), df.loc[sbj_list,ASR_substance_use_cols].hvplot.hist()),
          pn.Row(pn.pane.DataFrame(df.loc[sbj_list,ASR_syndrome_profiles_cols].sample(5),width=500), df.loc[sbj_list,ASR_syndrome_profiles_cols].hvplot.hist()),
          pn.Row(pn.pane.DataFrame(df.loc[sbj_list,ASR_critical_items_cols].sample(5),width=500), df.loc[sbj_list,ASR_critical_items_cols].hvplot.hist()))

# ## 1.3. Beck Depression Inventar-II (BDI)
#
# The BDI-II measures the severity of various depressive symptoms in adolescents and adults over the two weeks prior to completion of the inventar. It consists of 21 items that require multiple-choice answers that best describe statements about subjectively experienced states. The items can be scored on a four-point Likert scale (e.g., 0 = “I do not feel sad.” to 3 = “I am sosad or unhappy that I can’t stand it”). We used the German BDI version.

survey = 'BDI'
df     = pd.read_csv(survey_data_paths[survey], sep='\t', index_col=0)
df     = pd.DataFrame(df.loc[sbj_list,'BDI_summary_sum'])

s = plot_distributions_across_NBS_groups('BDI',df,NBS_groups)
s

common_sbjs = list(df.dropna().index) #.corrwith(snyq_by_subject_df.loc[list(df.dropna().index)],axis=0)
print("++ Number of subjects with data: %d / %d" % (len(common_sbjs),snyq_by_subject_df.shape[0]))

for col in introspection_dimensions:
    snycq_R_with_behavs.loc['BDI_summary_sum',col], snycq_Rp_with_behavs.loc['BDI_summary_sum',col] = pearsonr(snyq_by_subject_df.loc[common_sbjs,col].values,df.loc[common_sbjs,'BDI_summary_sum'].values)
    snycq_S_with_behavs.loc['BDI_summary_sum',col], snycq_Sp_with_behavs.loc['BDI_summary_sum',col] = spearmanr(snyq_by_subject_df.loc[common_sbjs,col].values,df.loc[common_sbjs,'BDI_summary_sum'].values)
    if snycq_Rp_with_behavs.loc['BDI_summary_sum',col] < 0.05:
        print(f'BDI vs. {col} | ',col,snycq_R_with_behavs.loc['BDI_summary_sum',col], snycq_Rp_with_behavs.loc['BDI_summary_sum',col])
    if snycq_Sp_with_behavs.loc['BDI_summary_sum',col] < 0.05:
        print(f'BDI vs. {col} | ',snycq_S_with_behavs.loc['BDI_summary_sum',col], snycq_Sp_with_behavs.loc['BDI_summary_sum',col])

# ## 1.4 Behavioral Inhibition and Approach System (BIS/BAS). 
# The BIS/BAS 18 measures individual differences in response to two motivational systems: behavioral inhibition and behavioral approach(systems postulated by Gray19,20). 
# It comprises a total of 24 items that can be scored using a four-pointLikert-type scale (1 = “not true for me at all” to 4 = “very true for me”). We used the German version ofthe questionnaire.

survey = 'BISBAS'
df     = pd.read_csv(survey_data_paths[survey], sep='\t', index_col=0)
df     = df.loc[sbj_list]

df_BIS = pd.DataFrame(df['BISBAS_BIS_sum'])
df_BAS = pd.DataFrame(df['BISBAS_BAS_sum'])

s = plot_distributions_across_NBS_groups('BISBAS_BIS_sum',df_BIS,NBS_groups)
s

s = plot_distributions_across_NBS_groups('BISBAS_BAS_sum',df_BAS,NBS_groups)
s

common_sbjs = list(df.dropna().index) #.corrwith(snyq_by_subject_df.loc[list(df.dropna().index)],axis=0)
print("++ Number of subjects with data: %d / %d" % (len(common_sbjs),snyq_by_subject_df.shape[0]))

for col in introspection_dimensions:
    snycq_R_with_behavs.loc['BISBAS_BIS_sum',col], snycq_Rp_with_behavs.loc['BISBAS_BIS_sum',col] = pearsonr(snyq_by_subject_df.loc[common_sbjs,col].values,df.loc[common_sbjs,'BISBAS_BIS_sum'].values)
    snycq_S_with_behavs.loc['BISBAS_BIS_sum',col], snycq_Sp_with_behavs.loc['BISBAS_BIS_sum',col] = spearmanr(snyq_by_subject_df.loc[common_sbjs,col].values,df.loc[common_sbjs,'BISBAS_BIS_sum'].values)
    snycq_R_with_behavs.loc['BISBAS_BAS_sum',col], snycq_Rp_with_behavs.loc['BISBAS_BAS_sum',col] = pearsonr(snyq_by_subject_df.loc[common_sbjs,col].values,df.loc[common_sbjs,'BISBAS_BAS_sum'].values)
    snycq_S_with_behavs.loc['BISBAS_BAS_sum',col], snycq_Sp_with_behavs.loc['BISBAS_BAS_sum',col] = spearmanr(snyq_by_subject_df.loc[common_sbjs,col].values,df.loc[common_sbjs,'BISBAS_BAS_sum'].values)
    if snycq_Rp_with_behavs.loc['BISBAS_BIS_sum',col] < 0.05:
        print(f'BIS vs. {col} | ',col,snycq_R_with_behavs.loc['BISBAS_BIS_sum',col], snycq_Rp_with_behavs.loc['BISBAS_BIS_sum',col])
    if snycq_Sp_with_behavs.loc['BISBAS_BIS_sum',col] < 0.05:
        print(f'BIS vs. {col} | ',snycq_S_with_behavs.loc['BISBAS_BIS_sum',col], snycq_Sp_with_behavs.loc['BISBAS_BIS_sum',col])
    if snycq_Rp_with_behavs.loc['BISBAS_BAS_sum',col] < 0.05:
        print(f'BIS vs. {col} | ',col,snycq_R_with_behavs.loc['BISBAS_BAS_sum',col], snycq_Rp_with_behavs.loc['BISBAS_BAS_sum',col])
    if snycq_Sp_with_behavs.loc['BISBAS_BAS_sum',col] < 0.05:
        print(f'BIS vs. {col} | ',snycq_S_with_behavs.loc['BISBAS_BAS_sum',col], snycq_Sp_with_behavs.loc['BISBAS_BAS_sum',col])

# ## 1.5 Body Consciousness Questionnaire (BCQ). 
#
# The BCQ assesses three components of body consciousness: private body (e.g., heartbeat perception), public body (perception of outward appearance), and body competence (aspects of the body, e.g., strength). 
#
# The questionnaire consists of 15 items that can be scored on a five-point Likert scale (0 = “extremely uncharacteristic” to 4 = “extremelycharacteristic”). We used a German translated version of the original English questionnaire.

survey = 'BCQ'
df     = pd.read_csv(survey_data_paths[survey], sep='\t', index_col=0)
df     = df.loc[sbj_list]

df_BCQ_private    = pd.DataFrame(df['BCQ_private_body_mean'])
df_BCQ_public     = pd.DataFrame(df['BCQ_public_body_mean'])
df_BCQ_competence = pd.DataFrame(df['BCQ_body_competence_mean'])

s = plot_distributions_across_NBS_groups('BCQ_private_body_mean',df_BCQ_private,NBS_groups)
s

s = plot_distributions_across_NBS_groups('BCQ_public_body_mean',df_BCQ_public,NBS_groups)
s

s = plot_distributions_across_NBS_groups('BCQ_body_competence_mean',df_BCQ_competence,NBS_groups)
s

for col in introspection_dimensions:
    snycq_R_with_behavs.loc['BCQ_private_body_mean',col], snycq_Rp_with_behavs.loc['BCQ_private_body_mean',col] = pearsonr(snyq_by_subject_df.loc[common_sbjs,col].values,df.loc[common_sbjs,'BCQ_private_body_mean'].values)
    snycq_S_with_behavs.loc['BCQ_private_body_mean',col], snycq_Sp_with_behavs.loc['BCQ_private_body_mean',col] = spearmanr(snyq_by_subject_df.loc[common_sbjs,col].values,df.loc[common_sbjs,'BCQ_private_body_mean'].values)
    
    snycq_R_with_behavs.loc['BCQ_public_body_mean',col], snycq_Rp_with_behavs.loc['BCQ_public_body_mean',col] = pearsonr(snyq_by_subject_df.loc[common_sbjs,col].values,df.loc[common_sbjs,'BCQ_public_body_mean'].values)
    snycq_S_with_behavs.loc['BCQ_public_body_mean',col], snycq_Sp_with_behavs.loc['BCQ_public_body_mean',col] = spearmanr(snyq_by_subject_df.loc[common_sbjs,col].values,df.loc[common_sbjs,'BCQ_public_body_mean'].values)

    snycq_R_with_behavs.loc['BCQ_body_competence_mean',col], snycq_Rp_with_behavs.loc['BCQ_body_competence_mean',col] = pearsonr(snyq_by_subject_df.loc[common_sbjs,col].values,df.loc[common_sbjs,'BCQ_body_competence_mean'].values)
    snycq_S_with_behavs.loc['BCQ_body_competence_mean',col], snycq_Sp_with_behavs.loc['BCQ_body_competence_mean',col] = spearmanr(snyq_by_subject_df.loc[common_sbjs,col].values,df.loc[common_sbjs,'BCQ_body_competence_mean'].values)

# ## 1.6 Boredom Proneness Scale (BPS) 
# The BP measures the tendency to experience boredom, in particularthe self-reported lack of internal and external stimulation. It consists of 28 items that can be scored on aseven-point Likert scale (1 = “total disagreement” to 7 = “total agreement”). We used a German translatedversion of the original English scale.

survey = 'BPS'
df     = pd.read_csv(survey_data_paths[survey], sep='\t', index_col=0)
df     = pd.DataFrame(df.loc[sbj_list,'BPS_sum'])

s = plot_distributions_across_NBS_groups('BPS_sum',df,NBS_groups)
s

common_sbjs = list(df.dropna().index) #.corrwith(snyq_by_subject_df.loc[list(df.dropna().index)],axis=0)
print("++ Number of subjects with data: %d / %d" % (len(common_sbjs),snyq_by_subject_df.shape[0]))

for col in introspection_dimensions:
    snycq_R_with_behavs.loc['BPS_sum',col], snycq_Rp_with_behavs.loc['BPS_sum',col] = pearsonr(snyq_by_subject_df.loc[common_sbjs,col].values,df.loc[common_sbjs,'BPS_sum'].values)
    snycq_S_with_behavs.loc['BPS_sum',col], snycq_Sp_with_behavs.loc['BPS_sum',col] = spearmanr(snyq_by_subject_df.loc[common_sbjs,col].values,df.loc[common_sbjs,'BPS_sum'].values)

# ## 1.7 Brief Self-Control Scale (SCS). 
# The SCS is a self-report measurement assessing the capacity for self-control. Self-control was operationalized as the capability to modify or override one’s own response tendencies. 
# It consists of 13 items that can be scoredon a five-point Likert scale (1 = “do not agree at all” to 5 = “completely agree”).

survey = 'SCS'
df     = pd.read_csv(survey_data_paths[survey], sep='\t', index_col=0)
df     = pd.DataFrame(df.loc[sbj_list,'SCS_SelfCtrl_sum'])

s = plot_distributions_across_NBS_groups('SCS_SelfCtrl_sum',df,NBS_groups)
s

common_sbjs = list(df.dropna().index) #.corrwith(snyq_by_subject_df.loc[list(df.dropna().index)],axis=0)
print("++ Number of subjects with data: %d / %d" % (len(common_sbjs),snyq_by_subject_df.shape[0]))

for col in introspection_dimensions:
    snycq_R_with_behavs.loc['SCS_SelfCtrl_sum',col], snycq_Rp_with_behavs.loc['SCS_SelfCtrl_sum',col] = pearsonr(snyq_by_subject_df.loc[common_sbjs,col].values,df.loc[common_sbjs,'SCS_SelfCtrl_sum'].values)
    snycq_S_with_behavs.loc['SCS_SelfCtrl_sum',col], snycq_Sp_with_behavs.loc['SCS_SelfCtrl_sum',col] = spearmanr(snyq_by_subject_df.loc[common_sbjs,col].values,df.loc[common_sbjs,'SCS_SelfCtrl_sum'].values)

# ## 1.8 Epworth Sleepiness Scale (ESS). 
# The ESS measures tendencies of sleepiness in everyday life. The scale consists of eight items addressing the subjective propensity to fall asleep in different situations. 
# The items can be scored on a four-point Likert scale (0 = “would never doze” to 3 = “high chance of dozing”).

survey = 'ESS'
df     = pd.read_csv(survey_data_paths[survey], sep='\t', index_col=0)
df     = pd.DataFrame(df.loc[sbj_list,'ESS_summary_sum'])

s = plot_distributions_across_NBS_groups('ESS_summary_sum',df,NBS_groups)
s

common_sbjs = list(df.dropna().index) #.corrwith(snyq_by_subject_df.loc[list(df.dropna().index)],axis=0)
print("++ Number of subjects with data: %d / %d" % (len(common_sbjs),snyq_by_subject_df.shape[0]))

for col in introspection_dimensions:
    snycq_R_with_behavs.loc['ESS_summary_sum',col], snycq_Rp_with_behavs.loc['ESS_summary_sum',col] = pearsonr(snyq_by_subject_df.loc[common_sbjs,col].values,df.loc[common_sbjs,'ESS_summary_sum'].values)
    snycq_S_with_behavs.loc['ESS_summary_sum',col], snycq_Sp_with_behavs.loc['ESS_summary_sum',col] = spearmanr(snyq_by_subject_df.loc[common_sbjs,col].values,df.loc[common_sbjs,'ESS_summary_sum'].values)

# ## 1.9. Goldsmiths Musical Sophistication Index (GoldMSI). 
# The Gold-MSI measures the level of experience with and understanding of music in community samples. 
# A subset of 16 items was measured, including the active engagement subscale and the musical training subscales (the item order isexplained in the ∗.txt file of this index). 
# The subscales perceptual abilities, singing abilities, and emotions were not included in the measurement. The items can be scored on a seven-point Likert scale(1 = “completely disagree” to 7 = “completely agree”).

survey = 'GoldMSI'
df     = pd.read_csv(survey_data_paths[survey], sep='\t', index_col=0)

df     = df.loc[sbj_list]
pn.Row(pn.pane.DataFrame(df.loc[sbj_list].sample(10)), df.loc[sbj_list].hvplot.hist())

common_sbjs = list(df.dropna().index) #.corrwith(snyq_by_subject_df.loc[list(df.dropna().index)],axis=0)
print("++ Number of subjects with data: %d / %d" % (len(common_sbjs),snyq_by_subject_df.shape[0]))

for col in introspection_dimensions:
    snycq_R_with_behavs.loc['GoldMSI_Active_sum',col], snycq_Rp_with_behavs.loc['GoldMSI_Active_sum',col] = pearsonr(snyq_by_subject_df.loc[common_sbjs,col].values,df.loc[common_sbjs,'GoldMSI_Active_sum'].values)
    snycq_S_with_behavs.loc['GoldMSI_Active_sum',col], snycq_Sp_with_behavs.loc['GoldMSI_Active_sum',col] = spearmanr(snyq_by_subject_df.loc[common_sbjs,col].values,df.loc[common_sbjs,'GoldMSI_Active_sum'].values)
    
    snycq_R_with_behavs.loc['GoldMSI_Training_sum',col], snycq_Rp_with_behavs.loc['GoldMSI_Training_sum',col] = pearsonr(snyq_by_subject_df.loc[common_sbjs,col].values,df.loc[common_sbjs,'GoldMSI_Training_sum'].values)
    snycq_S_with_behavs.loc['GoldMSI_Training_sum',col], snycq_Sp_with_behavs.loc['GoldMSI_Training_sum',col] = spearmanr(snyq_by_subject_df.loc[common_sbjs,col].values,df.loc[common_sbjs,'GoldMSI_Training_sum'].values)

# ## 1.10. Hospital Anxiety and Depression Scale (HADS). 
# The HADS measures the severity of depression- and anxiety-related symptoms for the week prior to completion and can be used to assess subclinical tendencies of depression and anxiety. 
# It consists of 14 items in total that can be scored on a four-pointLikert scale (e.g., 1 = “most of the time” to 4 = “never”).

survey = 'HADS'
df     = pd.read_csv(survey_data_paths[survey], sep='\t', index_col=0)

df     = df.loc[sbj_list]
pn.Row(pn.pane.DataFrame(df.loc[sbj_list].sample(10)), df.loc[sbj_list].hvplot.hist())

common_sbjs = list(df.dropna().index) #.corrwith(snyq_by_subject_df.loc[list(df.dropna().index)],axis=0)
print("++ Number of subjects with data: %d / %d" % (len(common_sbjs),snyq_by_subject_df.shape[0]))

for col in introspection_dimensions:
    snycq_R_with_behavs.loc['HADS-A_summary_sum',col], snycq_Rp_with_behavs.loc['HADS-A_summary_sum',col] = pearsonr(snyq_by_subject_df.loc[common_sbjs,col].values,df.loc[common_sbjs,'HADS-A_summary_sum'].values)
    snycq_S_with_behavs.loc['HADS-A_summary_sum',col], snycq_Sp_with_behavs.loc['HADS-A_summary_sum',col] = spearmanr(snyq_by_subject_df.loc[common_sbjs,col].values,df.loc[common_sbjs,'HADS-A_summary_sum'].values)
    
    snycq_R_with_behavs.loc['HADS-D_summary_sum',col], snycq_Rp_with_behavs.loc['HADS-D_summary_sum',col] = pearsonr(snyq_by_subject_df.loc[common_sbjs,col].values,df.loc[common_sbjs,'HADS-D_summary_sum'].values)
    snycq_S_with_behavs.loc['HADS-D_summary_sum',col], snycq_Sp_with_behavs.loc['HADS-D_summary_sum',col] = spearmanr(snyq_by_subject_df.loc[common_sbjs,col].values,df.loc[common_sbjs,'HADS-D_summary_sum'].values)

# ## 1.11. Internet Addiction Test (IAT). 
# The IAT assesses self-reported excessive use of the Internet. The testis comprised of 20 items that can be scored on a six-point Likert scale (0 = “does not apply” to5 = “always”). We used item three (i.e., “how often do you prefer the excitement of the Internet to intimacy with your partner?”) with a different scale compared to the original one. Therefore, this item was not included in the scoring of the scale..

survey = 'IAT'
df     = pd.read_csv(survey_data_paths[survey], sep='\t', index_col=0)

df     = pd.DataFrame(df.loc[sbj_list,'IAT_sum'])
pn.Row(pn.pane.DataFrame(df.loc[sbj_list].sample(10)), df.loc[sbj_list].hvplot.hist())

common_sbjs = list(df.dropna().index) #.corrwith(snyq_by_subject_df.loc[list(df.dropna().index)],axis=0)
print("++ Number of subjects with data: %d / %d" % (len(common_sbjs),snyq_by_subject_df.shape[0]))

for col in introspection_dimensions:
    snycq_R_with_behavs.loc['IAT_sum',col], snycq_Rp_with_behavs.loc['IAT_sum',col] = pearsonr(snyq_by_subject_df.loc[common_sbjs,col].values,df.loc[common_sbjs,'IAT_sum'].values)
    snycq_S_with_behavs.loc['IAT_sum',col], snycq_Sp_with_behavs.loc['IAT_sum',col] = spearmanr(snyq_by_subject_df.loc[common_sbjs,col].values,df.loc[common_sbjs,'IAT_sum'].values)

# ## 1.12. Involuntary Musical Imagery Scale (IMIS). 
# IMIS is a self-report inventory measuring phenomenological properties of the experiential tendency of having involuntary musical imagery, also known as“earworms”. 
# It measures four facets of involuntary musical imagery: the subjective evaluation of this phenomenon (negative valence), the embodied responses (movement), the personal contemplations(personal reflections), and the constructive properties (help). 
# It consists of 18 items that can be scored ondifferent scales: 14 items can be scored on a five-point Likert scale (1 = “never” to 5 = “always”); two items with different five-point Likert scales (e.g., 1 = “less than 5 seconds” to 5 = “more than a minute”); oneitem with a six-point Likert scale (1 = “never” to 6 = “almost continuously”). The English questionnaire consists of two parts (A and B) which were combined in the German version (see the respective ∗.txt filefor more details).

survey = 'IMIS'
df     = pd.read_csv(survey_data_paths[survey], sep='\t', index_col=0)

df     = df.loc[sbj_list]
pn.Row(pn.pane.DataFrame(df.loc[sbj_list].sample(10)), df.loc[sbj_list].hvplot.hist())

common_sbjs = list(df.dropna().index) #.corrwith(snyq_by_subject_df.loc[list(df.dropna().index)],axis=0)
print("++ Number of subjects with data: %d / %d" % (len(common_sbjs),snyq_by_subject_df.shape[0]))

for col in introspection_dimensions:
    for col_y in df.columns:
        snycq_R_with_behavs.loc[col_y,col], snycq_Rp_with_behavs.loc[col_y,col] = pearsonr(snyq_by_subject_df.loc[common_sbjs,col].values,df.loc[common_sbjs,col_y].values)
        snycq_S_with_behavs.loc[col_y,col], snycq_Sp_with_behavs.loc[col_y,col] = spearmanr(snyq_by_subject_df.loc[common_sbjs,col].values,df.loc[common_sbjs,col_y].values)

# ## 1.13. Mobile Phone Usage (MPU). 
# This in-house developed collection of items measures various patterns ofmobile phone usage, such as e-mail usage as well as the use of social network sites via smartphone. 
# It consists of 19 items with various answer formats.

survey = 'MPU'
df     = pd.read_csv(survey_data_paths[survey], sep='\t', index_col=0)

df.sample(5)

# ## 1.14. Multimedia Multitasking Index (MMI)
# The MMI measures the extent of simultaneous use of 12 different media types: computer-based streaming (video, music), non-music audio, computer games,voice calls, instant messaging, text messaging, email, web surfing, and other applications such as Word processing. 
# It consists of a total of 219 items, across the 12 media types, that can be scored on different Likert scales (e.g., 1 = “never” to 4 = “most of the time”; 1 = “more time” to 3 = “same amount of time”).

survey = 'MMI'
df     = pd.read_csv(survey_data_paths[survey], sep='\t', index_col=0)

df     = pd.DataFrame(df.loc[sbj_list,'MMI_score'])
pn.Row(pn.pane.DataFrame(df.loc[sbj_list].sample(10)), df.loc[sbj_list].hvplot.hist())

common_sbjs = list(df.dropna().index) #.corrwith(snyq_by_subject_df.loc[list(df.dropna().index)],axis=0)
print("++ Number of subjects with data: %d / %d" % (len(common_sbjs),snyq_by_subject_df.shape[0]))

for col in introspection_dimensions:
    snycq_R_with_behavs.loc['MMI_score',col], snycq_Rp_with_behavs.loc['MMI_score',col] = pearsonr(snyq_by_subject_df.loc[common_sbjs,col].values,df.loc[common_sbjs,'MMI_score'].values)
    snycq_S_with_behavs.loc['MMI_score',col], snycq_Sp_with_behavs.loc['MMI_score',col] = spearmanr(snyq_by_subject_df.loc[common_sbjs,col].values,df.loc[common_sbjs,'MMI_score'].values)

# ## 1.15. NEO Personality Inventory-Revised (NEO PI-R). 
# The NEO PI-R assesses the five personality traits:extraversion, agreeableness, conscientiousness, neuroticism, and openness to experience. 
# Moreover,the questionnaire also assesses six underlying facets for each of the five main factors. It consists of 241 items that can be scored on a five-point Likert scale.
# Due to a technical error, item 71 (i.e., “I am seldom sad or depressed”) was measured twice; one time instead of item 46 (i.e., “I seldom feel self-conscious when I’m around people”). Thus, item 46 was not taken into account for the summary score of subscale N3. Additionally, item 83 was missing and was therefore nottaken into account for creating subscale O5.

survey = 'NEO'
df     = pd.read_csv(survey_data_paths[survey], sep='\t', index_col=0)

df = df.loc[sbj_list,['NEO_N','NEO_E','NEO_O','NEO_A','NEO_C']]
pn.Row(pn.pane.DataFrame(df.loc[sbj_list].sample(10)), df.loc[sbj_list].hvplot.hist())

common_sbjs = list(df.dropna().index) #.corrwith(snyq_by_subject_df.loc[list(df.dropna().index)],axis=0)
print("++ Number of subjects with data: %d / %d" % (len(common_sbjs),snyq_by_subject_df.shape[0]))

for col in introspection_dimensions:
    for col_y in df.columns:
        snycq_R_with_behavs.loc[col_y,col], snycq_Rp_with_behavs.loc[col_y,col] = pearsonr(snyq_by_subject_df.loc[common_sbjs,col].values,df.loc[common_sbjs,col_y].values)
        snycq_S_with_behavs.loc[col_y,col], snycq_Sp_with_behavs.loc[col_y,col] = spearmanr(snyq_by_subject_df.loc[common_sbjs,col].values,df.loc[common_sbjs,col_y].values)

scatter = pd.DataFrame([snyq_by_subject_df.loc[common_sbjs,'Negative'].values,df.loc[common_sbjs,'NEO_N'].values], index=['Negative Thoughts','Neuroticism']).T.hvplot.scatter(x='Neuroticism',y='Negative Thoughts', aspect='square')
scatter * hv.Slope.from_scatter(scatter)

# ## 1.16. Personality Style and Disorder Inventory (PSSI). 
#
# The PSSI is a self-report measurement assessing 14 personality styles. These personality styles are conceptualized as non-pathologic, sub-clinical equivalents of personality disorders as described in diagnostic manuals such as the Diagnostic andStatistical Manual of Mental Disorders. The inventory consists of 140 items that can be scored on afour-point Likert scale (1 = “do not agree” to 4 = “highly agree”).
#
# PN = Paranoid, SZ = Schizophrenic, ST = Schyzotypal, BL = Borderline, HI = Histronic, NA = Narcisitic, SU = Avoidant, AB = Dependent, ZW = Obsessive-Compulsive, NT = Negativistic, DP = Depressivem SL = Altruistic, RH = Rhapsodic, AS = Assertive

survey = 'PSSI'
df     = pd.read_csv(survey_data_paths[survey], sep='\t', index_col=0)

df = df.loc[sbj_list]
pn.Row(pn.pane.DataFrame(df.loc[sbj_list].sample(10)), df.loc[sbj_list].hvplot.hist())

common_sbjs = list(df.dropna().index) #.corrwith(snyq_by_subject_df.loc[list(df.dropna().index)],axis=0)
print("++ Number of subjects with data: %d / %d" % (len(common_sbjs),snyq_by_subject_df.shape[0]))

for col in introspection_dimensions:
    for col_y in df.columns:
        snycq_R_with_behavs.loc[col_y,col], snycq_Rp_with_behavs.loc[col_y,col] = pearsonr(snyq_by_subject_df.loc[common_sbjs,col].values,df.loc[common_sbjs,col_y].values)
        snycq_S_with_behavs.loc[col_y,col], snycq_Sp_with_behavs.loc[col_y,col] = spearmanr(snyq_by_subject_df.loc[common_sbjs,col].values,df.loc[common_sbjs,col_y].values)

# ## 1.17 Self-Esteem Scale (SE)
# The SE is a self-report scale measuring global self-worth by assessing positive and negative feelings about the self. 
# It comprises eight items that can be scored on a six-point Likertscale (0 = “does not apply” to 5 = “applies to me”).

survey = 'SE'
df     = pd.read_csv(survey_data_paths[survey], sep='\t', index_col=0)
df.shape

df     = pd.DataFrame(df.loc[sbj_list,'SE_Mean_SelfEst'])
pn.Row(pn.pane.DataFrame(df.loc[sbj_list].sample(10)), df.loc[sbj_list].hvplot.hist())

common_sbjs = list(df.dropna().index) #.corrwith(snyq_by_subject_df.loc[list(df.dropna().index)],axis=0)
print("++ Number of subjects with data: %d / %d" % (len(common_sbjs),snyq_by_subject_df.shape[0]))

for col in introspection_dimensions:
    snycq_R_with_behavs.loc['SE_Mean_SelfEst',col], snycq_Rp_with_behavs.loc['SE_Mean_SelfEst',col] = pearsonr(snyq_by_subject_df.loc[common_sbjs,col].values,df.loc[common_sbjs,'SE_Mean_SelfEst'].values)
    snycq_S_with_behavs.loc['SE_Mean_SelfEst',col], snycq_Sp_with_behavs.loc['SE_Mean_SelfEst',col] = spearmanr(snyq_by_subject_df.loc[common_sbjs,col].values,df.loc[common_sbjs,'SE_Mean_SelfEst'].values)

# ## 1.18. Short Dark Triad (SD3). 
# The SD3 assesses the following personality traits: machiavellianism, narcissism, and psychopathy in their subclinical manifestations. 
# It consists of 27 items that can be scored on a five-point Likert scale (1 = “strongly disagree” to 5 = “strongly agree”). The questionnaire was retrieved from an online platform (http://www.midss.org/sites/default/files/d3.pdf) previous to its publication. Thus, item two of the used questionnaire (i.e., “Generally speaking, people won’t work hard unless they have to”) is different from the published version (i.e., “I like to use clever manipulation to get my way”). 
#
# > **NOTE**: I did a trick to match the IDs, but I am not sure. In fact when one looks at the size of the data, it feels like this is data for the other section of the LEMON dataset. Unless new information becomes available, I would suggest removing from the analysis

survey = 'SD3'
df     = pd.read_csv(survey_data_paths[survey], sep='\t', index_col=0)
df.index = ['sub-'+str(i).zfill(6) for i in df.index]
df.index.name = 'participant_id'
df.shape

df = df.loc[sbj_list]
pn.Row(pn.pane.DataFrame(df.loc[sbj_list].sample(10)), df.loc[sbj_list].hvplot.hist())

common_sbjs = list(df.dropna().index) #.corrwith(snyq_by_subject_df.loc[list(df.dropna().index)],axis=0)
print("++ Number of subjects with data: %d / %d" % (len(common_sbjs),snyq_by_subject_df.shape[0]))

for col in introspection_dimensions:
    for col_y in df.columns:
        snycq_R_with_behavs.loc[col_y,col], snycq_Rp_with_behavs.loc[col_y,col] = pearsonr(snyq_by_subject_df.loc[common_sbjs,col].values,df.loc[common_sbjs,col_y].values)
        snycq_S_with_behavs.loc[col_y,col], snycq_Sp_with_behavs.loc[col_y,col] = spearmanr(snyq_by_subject_df.loc[common_sbjs,col].values,df.loc[common_sbjs,col_y].values)

# ## 1.19. Social Desirability Scale-17 (SDS). 
#
# The SDS is a self-report questionnaire that assesses one’s tendency to seek social approval, and it can be used to control for biased answer’s tendencies due to social desirability. We used a German version of the scale 41 consisting of 17 items that can be scored on a five-point Likert scale (1 = “do not agree at all” to 5 = “completely agree”).

survey = 'SDS'
df     = pd.read_csv(survey_data_paths[survey], sep='\t', index_col=0)

df     = pd.DataFrame(df.loc[sbj_list,'SDS_sum'])
pn.Row(pn.pane.DataFrame(df.loc[sbj_list].sample(10)), df.loc[sbj_list].hvplot.hist())

common_sbjs = list(df.dropna().index) #.corrwith(snyq_by_subject_df.loc[list(df.dropna().index)],axis=0)
print("++ Number of subjects with data: %d / %d" % (len(common_sbjs),snyq_by_subject_df.shape[0]))

for col in introspection_dimensions:
    snycq_R_with_behavs.loc['SDS_sum',col], snycq_Rp_with_behavs.loc['SDS_sum',col] = pearsonr(snyq_by_subject_df.loc[common_sbjs,col].values,df.loc[common_sbjs,'SDS_sum'].values)
    snycq_S_with_behavs.loc['SDS_sum',col], snycq_Sp_with_behavs.loc['SDS_sum',col] = spearmanr(snyq_by_subject_df.loc[common_sbjs,col].values,df.loc[common_sbjs,'SDS_sum'].values)

scatter = pd.DataFrame([snyq_by_subject_df.loc[common_sbjs,'Positive'].values,df.loc[common_sbjs,'SDS_sum'].values], index=['Positive Thoughts','Social Desirability']).T.hvplot.scatter(x='Social Desirability',y='Positive Thoughts', aspect='square')
scatter * hv.Slope.from_scatter(scatter)

# ## 1.20. Tuckman Procrastination Scale (TPS). 
# The TPS assesses self-reports of procrastination in everyday life, which are related to the tendency to inappropriately delay pending tasks. It consists of 16 items that can be scored on a five-point Likert scale (1 = “does not apply to me at all” to 5 = “applies to me to a greatextent”). 

survey = 'TPS'
df     = pd.read_csv(survey_data_paths[survey], sep='\t', index_col=0)

df     = pd.DataFrame(df.loc[sbj_list,'TPS_D_sum'])
pn.Row(pn.pane.DataFrame(df.loc[sbj_list].sample(10)), df.loc[sbj_list].hvplot.hist())

common_sbjs = list(df.dropna().index) #.corrwith(snyq_by_subject_df.loc[list(df.dropna().index)],axis=0)
print("++ Number of subjects with data: %d / %d" % (len(common_sbjs),snyq_by_subject_df.shape[0]))

for col in introspection_dimensions:
    snycq_R_with_behavs.loc['TPS_D_sum',col], snycq_Rp_with_behavs.loc['TPS_D_sum',col] = pearsonr(snyq_by_subject_df.loc[common_sbjs,col].values,df.loc[common_sbjs,'TPS_D_sum'].values)
    snycq_S_with_behavs.loc['TPS_D_sum',col], snycq_Sp_with_behavs.loc['TPS_D_sum',col] = spearmanr(snyq_by_subject_df.loc[common_sbjs,col].values,df.loc[common_sbjs,'TPS_D_sum'].values)

# ## 1.21. UPPS-P Impulsive Behavior Scale (UPPS). 
# The UPPS-P is a self-report measure of different trait aspects of impulsive behavior. This revised scale quantifies five distinguishable facets of impulsivity:positive urgency, negative urgency, lack of premeditation, lack of perseverance, and sensation seeking. 
# It consists of 59 items that can be scored on a four-point Likert scale (1 = “strongly agree” to 4 = “stronglydisagree”).

survey = 'UPPS'
df     = pd.read_csv(survey_data_paths[survey], sep='\t', index_col=0)

df = df.loc[sbj_list]
pn.Row(pn.pane.DataFrame(df.loc[sbj_list].sample(10)), df.loc[sbj_list].hvplot.hist())

common_sbjs = list(df.dropna().index) #.corrwith(snyq_by_subject_df.loc[list(df.dropna().index)],axis=0)
print("++ Number of subjects with data: %d / %d" % (len(common_sbjs),snyq_by_subject_df.shape[0]))

for col in introspection_dimensions:
    for col_y in df.columns:
        snycq_R_with_behavs.loc[col_y,col], snycq_Rp_with_behavs.loc[col_y,col] = pearsonr(snyq_by_subject_df.loc[common_sbjs,col].values,df.loc[common_sbjs,col_y].values)
        snycq_S_with_behavs.loc[col_y,col], snycq_Sp_with_behavs.loc[col_y,col] = spearmanr(snyq_by_subject_df.loc[common_sbjs,col].values,df.loc[common_sbjs,col_y].values)

pBonf = 0.05 / (snycq_R_with_behavs.shape[0]*snycq_R_with_behavs.shape[1])
snycq_R_with_behavs = snycq_R_with_behavs.infer_objects()
snycq_R_with_behavs.round(2).hvplot.heatmap(height=600).opts(xrotation=90,clim=(-.5,.5), cmap='RdBu_r') + snycq_R_with_behavs[snycq_Rp_with_behavs<0.05].round(2).hvplot.heatmap(height=600).opts(xrotation=90,clim=(-.5,.5), cmap='RdBu_r')  + snycq_R_with_behavs[snycq_Rp_with_behavs<pBonf].round(2).hvplot.heatmap(height=600).opts(xrotation=90,clim=(-.5,.5), cmap='RdBu_r')

pBonf = 0.05 / (snycq_R_with_behavs.shape[0]*snycq_R_with_behavs.shape[1])
snycq_R_with_behavs = snycq_R_with_behavs.infer_objects()
snycq_R_with_behavs.round(2).hvplot.heatmap(height=600).opts(xrotation=90,clim=(-.5,.5), cmap='RdBu_r') + snycq_R_with_behavs[snycq_Rp_with_behavs<0.05].round(2).hvplot.heatmap(height=600).opts(xrotation=90,clim=(-.5,.5), cmap='RdBu_r')  + snycq_R_with_behavs[snycq_Rp_with_behavs<pBonf].round(2).hvplot.heatmap(height=600).opts(xrotation=90,clim=(-.5,.5), cmap='RdBu_r')


