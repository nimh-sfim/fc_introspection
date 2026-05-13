#!/usr/bin/env python
# coding: utf-8

# # Description: Visualiaze Subject-Level Cross-Validation CPM Results
# 
# This notebook loads the result of running 100 iterations of the CPM algorithm on real data, and also those of running 10,000 iterations with randomized labels (null distribution).
# 
# Using these data, the notebook then computes non-parametric p-values for each prediction.
# 
# Finally, the notebook generates summary figures for the ability of CPM to predict experiential variables.
# 
# > **NOTE:** Although the notebook loads and computes values for the three CPM models (pos, neg and glm), ultimately on the paper we only report results for the glm case.

# In[1]:


import pandas as pd
import os.path as osp
from utils.basics import RESOURCES_CPM_DIR
import hvplot.pandas
from tqdm import tqdm
import numpy as np
import pickle
from utils.basics import FB_400ROI_ATLAS_NAME
from cpm.plotting import plot_predictions
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
import xarray as xr
from statsmodels.stats.multitest import multipletests


# In[2]:


print('hvplot version: %s' % str(hvplot.__version__))
print('xr version: %s' % str(xr.__version__))
print('pandas version: %s' % str(pd.__version__))


# ***
# Set all configurations for CPM

# In[3]:


ACCURACY_METRIC      = 'pearson'
CORR_TYPE            = 'pearson'
E_SUMMARY_METRIC     = 'sum'
CONFOUNDS            = 'conf_residualized'
# How Behaviors are encoded so far
BEHAVIOR_LIST        = ['Factor1','Factor2','Vigilance','Images','Words','People','Myself','Positive','Negative','Surroundings','Intrusive','Future','Past','Specific']
# New labels for consistency with text descriptions
BEHAVIOR_LABELS_DICT = {'Factor1':'Thought Pattern 1','Factor2':'Thought Pattern 2','Vigilance':'Wakefulness',
                        'Images':'Images','Words':'Words','People':'People','Myself':'Myself','Positive':'Positive','Negative':'Negative','Surroundings':'Surroundings','Intrusive':'Intrusive','Future':'Future','Past':'Past','Specific':'Specific'}
BEHAVIOR_LABELS      = ['Thought Pattern 1','Thought Pattern 2','Wakefulness','Images','Words','People','Myself','Positive','Negative','Surroundings','Intrusive','Future','Past','Specific']
# Mode used for running CPM
SPLIT_MODE           = 'subject_aware'
# Atlas Selection
ATLAS                = FB_400ROI_ATLAS_NAME


# ***
# # 1. Load Observed and Predicted values over CPM attemps
# 
# # 1.1. Real Data
# 
# We load observed and predicted values for the 100 permutations on real data

# In[4]:


get_ipython().run_cell_magic('time', '', "real_results_path = osp.join(RESOURCES_CPM_DIR,f'real-{ATLAS}-{SPLIT_MODE}-{CONFOUNDS}-{CORR_TYPE}_{E_SUMMARY_METRIC}.pkl')\nprint('++ INFO: Loading real predictions from %s' % real_results_path)\nwith open(real_results_path,'rb') as f:\n     real_predictions_xr = pickle.load(f)\nNbehavs, Niters_real, Nscans, Nresults = real_predictions_xr.shape\nprint('++ INFO: Real predictions shape: Ntargets=%d, Niterations=%d, Nscans=%d, Nresults=%d' % (Nbehavs, Niters_real, Nscans, Nresults))\n")


# We update Behavior coordinate dimensions for proper labeling of final figures so that they agree with the main text

# In[5]:


real_predictions_xr = real_predictions_xr.assign_coords({'Behavior':[BEHAVIOR_LABELS_DICT[b] for b in real_predictions_xr.Behavior.values]})


# ## 1.2. Null Permutations 
# Next we load observed and predicted values for the 10,000 Randomized permations

# In[6]:


get_ipython().run_cell_magic('time', '', "null_results_path = osp.join(RESOURCES_CPM_DIR,f'null-{ATLAS}-{SPLIT_MODE}-{CONFOUNDS}-{CORR_TYPE}_{E_SUMMARY_METRIC}.pkl')\nprint('++ INFO: Loading null predictions from %s' % null_results_path)\n\nwith open(null_results_path,'rb') as f:\n     null_predictions_xr = pickle.load(f)\n_, Niters_null, _, _ = null_predictions_xr.shape\nprint('++ INFO: Null predictions shape: Ntargets=%d, Niterations=%d, Nscans=%d, Nresults=%d' % (Nbehavs, Niters_null, Nscans, Nresults))\n")


# In[7]:


null_predictions_xr = null_predictions_xr.assign_coords({'Behavior':[BEHAVIOR_LABELS_DICT[b] for b in null_predictions_xr.Behavior.values]})


# ***
# # 2. Compute Prediction Accuracies for all CPM attemps
# 
# ## 2.1. Real data
# 
# We first compute accuracies in terms of the 

# In[9]:


get_ipython().run_cell_magic('time', '', "accuracy_real          = {BEHAVIOR:pd.DataFrame(index=range(Niters_real), columns=['Accuracy']) for BEHAVIOR in BEHAVIOR_LABELS}\naccuracy_real_Spearman = {BEHAVIOR:pd.DataFrame(index=range(Niters_real), columns=['Accuracy']) for BEHAVIOR in BEHAVIOR_LABELS}\n\np_values          = pd.DataFrame(index=BEHAVIOR_LABELS,columns=['Non Parametric','Parametric'])\np_values_Spearman = pd.DataFrame(index=BEHAVIOR_LABELS,columns=['Non Parametric','Parametric'])\n\nfor BEHAVIOR in BEHAVIOR_LABELS:\n    for niter in tqdm(range(Niters_real), desc=BEHAVIOR):\n        observed  = pd.Series(real_predictions_xr.loc[BEHAVIOR,niter,:,'observed'].values)\n        if E_SUMMARY_METRIC == 'ridge':\n            predicted = pd.Series(real_predictions_xr.loc[BEHAVIOR,niter,:,'predicted (ridge)'].values)\n        else:\n            predicted = pd.Series(real_predictions_xr.loc[BEHAVIOR,niter,:,'predicted (glm)'].values)\n        accuracy_real[BEHAVIOR].loc[niter]  = observed.corr(predicted, method='pearson')\n        accuracy_real_Spearman[BEHAVIOR].loc[niter]  = observed.corr(predicted, method='spearman')\n        _,p_values.loc[BEHAVIOR,'Parametric'] = pearsonr(observed,predicted)\n        _,p_values_Spearman.loc[BEHAVIOR,'Parametric'] = spearmanr(observed,predicted)\n")


# Get median accuracies across all 100 real attempts

# In[11]:


median_accuracies = pd.DataFrame(columns=['Pearson R','Spearman R'], index=BEHAVIOR_LABELS)
for BEHAVIOR in BEHAVIOR_LABELS:
    median_accuracies.loc[BEHAVIOR,'Pearson R'] = accuracy_real[BEHAVIOR].median().values[0]
    median_accuracies.loc[BEHAVIOR,'Spearman R'] = accuracy_real_Spearman[BEHAVIOR].median().values[0]


# Write median accuracies to disk

# In[12]:


median_accuracies.to_csv(f'../resources/cpm/{SPLIT_MODE}_final_avg_accuracies.csv')
print(f'++ INFO: Median accuracies saved to ../resources/cpm/{SPLIT_MODE}_final_avg_accuracies.csv')


# Print accuracies reported in Table 4

# In[14]:


median_accuracies['Pearson R'].infer_objects().round(2)


# ## 2.2. Null data

# In[15]:


get_ipython().run_cell_magic('time', '', "accuracy_null = {BEHAVIOR:pd.DataFrame(index=range(Niters_null), columns=['Accuracy']) for BEHAVIOR in BEHAVIOR_LABELS}\nfor BEHAVIOR in BEHAVIOR_LABELS:\n    for niter in tqdm(range(Niters_null), desc=BEHAVIOR):\n        observed  = pd.Series(null_predictions_xr.loc[BEHAVIOR,niter,:,'observed'].values)\n        if E_SUMMARY_METRIC == 'ridge':\n            predicted = pd.Series(null_predictions_xr.loc[BEHAVIOR,niter,:,'predicted (ridge)'].values)\n        else:\n            predicted = pd.Series(null_predictions_xr.loc[BEHAVIOR,niter,:,'predicted (glm)'].values)\n        accuracy_null[BEHAVIOR].loc[niter]  = observed.corr(predicted, method=ACCURACY_METRIC)\n")


# ***
# 
# # 3. Compute non-parametric p-values associated with the accuracies
# 
# For this, we rely on the null distribution generated via label randomization. 
# 
# We use the formula on section 2.4.4 from Finn & Bandettini ["Movie-watching outperforms rest for functional connectivity-based prediction of behavior"](https://www.sciencedirect.com/science/article/pii/S1053811921002408) NeuroImage 2021

# In[16]:


p_values.columns.name = 'p-value'
for BEHAVIOR in BEHAVIOR_LABELS:
    p_values.loc[BEHAVIOR,'Non Parametric'] = (((accuracy_null[BEHAVIOR] > accuracy_real[BEHAVIOR].median()).sum() + 1) / (Niters_null+1)).values[0]


# Now we apply the FDRbh correction

# In[17]:


(reject_bonf, p_values['Non Parametric, FDRbh'], _, _ ) = multipletests(p_values['Non Parametric'],alpha=0.05,method='fdr_bh')


# Here we now show the remaining columns in Table 4

# In[20]:


p_values.infer_objects().round(3)[['Parametric','Non Parametric','Non Parametric, FDRbh']]


# ***
# # 4. Generate Prediction-reporting Figures
# 

# In[21]:


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# We create a new dataframe with the accuracy for all iterations in tidy form (ideal for plotting with sns functions)

# In[24]:


get_ipython().run_cell_magic('time', '', "rows = []\n\nfor BEHAVIOR in BEHAVIOR_LABELS:\n    for i in tqdm(range(Niters_null), desc=BEHAVIOR):\n        rows.append({\n            'Question': BEHAVIOR,\n            'Iteration': i,\n            'R': accuracy_null[BEHAVIOR].loc[i].values[0]\n        })\n\nnull_df = pd.DataFrame(rows, columns=['Question', 'Iteration', 'R'])\n")


# In[25]:


get_ipython().run_cell_magic('time', '', "rows = []\n\nfor BEHAVIOR in BEHAVIOR_LABELS:\n    for i in tqdm(range(Niters_real), desc=BEHAVIOR):\n        rows.append({\n            'Question': BEHAVIOR,\n            'Iteration': i,\n            'R': accuracy_real[BEHAVIOR].loc[i].values[0]\n        })\n\nreal_df = pd.DataFrame(rows, columns=['Question', 'Iteration', 'R'])\n")


# We will now save these summary views to disk, as we will need them on the next notebook that creates a dashboard that allows a comprehensive exploration of the CPM results.
# 
# > **NOTE:** This file is used in S18 Dashboard to load results. If you don't run this cell when new CPM results are available, the Dashboard will present outdated results.

# In[26]:


output_path = osp.join(RESOURCES_CPM_DIR,f'cpm_predictions_summary-{SPLIT_MODE}-{CONFOUNDS}-{CORR_TYPE}.pkl')
outputs     = {'real_df':real_df,'null_df':null_df, 'accuracy_real': accuracy_real, 'accuracy_null':accuracy_null, 'p_values':p_values, 'real_predictions_xr':real_predictions_xr, 'null_predictions_xr':null_predictions_xr}
with open(output_path ,'wb') as f:
    pickle.dump(outputs,f)
print('++ INFO: Data written to disk [%s]' % output_path)


# ## 4.1. Plot without statistical annotations
# 

# In[27]:


median_width = 0.4
sns.set(style='whitegrid')


# In[28]:


fig,ax = plt.subplots(1,1,figsize=(15,5))
sns.boxenplot(data=null_df,x='Question',y='R', color='lightgray', ax=ax) 
sns.stripplot(data=real_df,x='Question', y='R', alpha=.8, ax=ax)
plt.xticks(rotation=45);
for tick, text in zip(ax.get_xticks(), ax.get_xticklabels()):
    question   = text.get_text()
    median_val = accuracy_real[question].median().values[0]
    ax.plot([tick-median_width/2, tick+median_width/2],[median_val,median_val], lw=4, color='k')
ax.set_ylim(-.3,.4)
ax.set_ylabel('Prediction Accuracy: R(Observed,Predicted)');
ax.set_xlabel('SNYCQ Item')


# ## 4.2. Plot with Statistical Annotations

# In[29]:


fig,ax = plt.subplots(1,1,figsize=(15,5))
sns.boxenplot(data=null_df,x='Question',y='R', color='lightgray', ax=ax) 
sns.stripplot(data=real_df,x='Question', y='R', alpha=.5, ax=ax)
plt.xticks(rotation=45);
for tick, text in zip(ax.get_xticks(), ax.get_xticklabels()):
    # Add Black Line Signaling Median
    question   = text.get_text()
    median_val = accuracy_real[question].median().values[0]
    ax.plot([tick-median_width/2, tick+median_width/2],[median_val,median_val], lw=4, color='k')
    # Statistical Significant Information
    p = p_values.loc[question,'Non Parametric']
    if 5.00e-02 < p <= 1.00e+00:
        annot = '' 
    elif 1.00e-02 < p <= 5.00e-02:
        annot = '*'
    elif 1.00e-03 < p <= 1.00e-02:
        annot = '**'
    elif 1.00e-04 < p <= 1.00e-03:
        annot = '***'
    elif p <= 1.00e-04:
        annot = '****'
    max_val = real_df.set_index('Question').max()['R']
    ax.annotate(annot, xy=(tick, max_val+0.02), ha='center', fontsize=15)
    
ax.set_ylim(-.3,.4)
ax.set_ylabel('Prediction Accuracy: R(Observed,Predicted)');
ax.set_xlabel('SNYCQ Item')


# We will also plot the same results, but separated in three different panels. One for the wakefulness question, one for the two factors, and one for the 11 questions entering the Sparse Box-Constrained Non-Negative Matrix Factorization. This might come handy for presentations, yet it is exactly the same information as above.

# In[30]:


median_width = 0.4
sns.set(style='whitegrid')
fig,ax = plt.subplots(1,3 ,figsize=(20,5), gridspec_kw={'width_ratios': [1,2,14]})
# Vigilance
sns.boxenplot(data=null_df[null_df['Question']=='Wakefulness'],x='Question',y='R', color='lightgray', ax=ax[0]) 
sns.stripplot(data=real_df[real_df['Question']=='Wakefulness'],x='Question', y='R', alpha=.5, ax=ax[0])
for tick, text in zip(ax[0].get_xticks(), ax[0].get_xticklabels()):
    # Add Black Line Signaling Median
    question   = text.get_text()
    median_val = accuracy_real[question].median().values[0]
    print(text, '%.2f' % median_val)
    ax[0].plot([tick-median_width/2, tick+median_width/2],[median_val,median_val], lw=4, color='k')
    # Statistical Significant Information
    p = p_values.loc[question,'Non Parametric']
    if 5.00e-02 < p <= 1.00e+00:
        annot = '' 
    elif 1.00e-02 < p <= 5.00e-02:
        annot = '*'
    elif 1.00e-03 < p <= 1.00e-02:
        annot = '**'
    elif 1.00e-04 < p <= 1.00e-03:
        annot = '***'
    elif p <= 1.00e-04:
        annot = '****'
    max_val = real_df.set_index('Question').max()['R']
    ax[0].annotate(annot, xy=(tick, max_val+0.02), ha='center', fontsize=15)
    
ax[0].set_ylim(-.3,.4)
ax[0].set_ylabel('Prediction Accuracy: R(Observed,Predicted)');
ax[0].set_xlabel('')

# Factors
sns.boxenplot(data=null_df[(null_df['Question']=='Thought Pattern 1') | (null_df['Question']=='Thought Pattern 2')],x='Question', y='R', color='lightgray', ax=ax[1]) 
sns.stripplot(data=real_df[(real_df['Question']=='Thought Pattern 1') | (real_df['Question']=='Thought Pattern 2')],x='Question', y='R', alpha=.5, ax=ax[1])
for tick, text in zip(ax[1].get_xticks(), ax[1].get_xticklabels()):
    # Add Black Line Signaling Median
    question   = text.get_text()
    median_val = accuracy_real[question].median().values[0]
    print(text, '%.2f' % median_val)
    ax[1].plot([tick-median_width/2, tick+median_width/2],[median_val,median_val], lw=4, color='k')
    # Statistical Significant Information
    p = p_values.loc[question,'Non Parametric']
    if 5.00e-02 < p <= 1.00e+00:
        annot = '' 
    elif 1.00e-02 < p <= 5.00e-02:
        annot = '*'
    elif 1.00e-03 < p <= 1.00e-02:
        annot = '**'
    elif 1.00e-04 < p <= 1.00e-03:
        annot = '***'
    elif p <= 1.00e-04:
        annot = '****'
    max_val = real_df.set_index('Question').max()['R']
    ax[1].annotate(annot, xy=(tick, max_val+0.02), ha='center', fontsize=15)
ax[1].set_xticklabels(['Pattern 1','Pattern 2'])
ax[1].set_ylim(-.3,.4)
ax[1].set_ylabel('Prediction Accuracy: R(Observed,Predicted)');
ax[1].set_xlabel('Thought Patterns')

# Individual Iterms
sns.boxenplot(data=null_df[(null_df['Question']!='Thought Pattern 1') & (null_df['Question']!='Thought Pattern 2') & (null_df['Question']!='Wakefulness')],x='Question', y='R', color='lightgray', ax=ax[2]) 
sns.stripplot(data=real_df[(real_df['Question']!='Thought Pattern 1') & (real_df['Question']!='Thought Pattern 2') & (real_df['Question']!='Wakefulness')],x='Question', y='R', alpha=.5, ax=ax[2])
for tick, text in zip(ax[2].get_xticks(), ax[2].get_xticklabels()):
    # Add Black Line Signaling Median
    question   = text.get_text()
    median_val = accuracy_real[question].median().values[0]
    ax[2].plot([tick-median_width/2, tick+median_width/2],[median_val,median_val], lw=4, color='k')
    # Statistical Significant Information
    p = p_values.loc[question,'Non Parametric']
    if 5.00e-02 < p <= 1.00e+00:
        annot = '' 
    elif 1.00e-02 < p <= 5.00e-02:
        annot = '*'
        print(text, '%.2f' % median_val)
    elif 1.00e-03 < p <= 1.00e-02:
        annot = '**'
        print(text, '%.2f' % median_val)
    elif 1.00e-04 < p <= 1.00e-03:
        annot = '***'
        print(text, '%.2f' % median_val)
    elif p <= 1.00e-04:
        annot = '****'
        print(text, '%.2f' % median_val)
    max_val = real_df.set_index('Question').max()['R']
    ax[2].annotate(annot, xy=(tick, max_val+0.02), ha='center', fontsize=15)
    
ax[2].set_ylim(-.3,.4)
ax[2].set_ylabel('Prediction Accuracy: R(Observed,Predicted)');
ax[2].set_xlabel('SNYC Questionnaire: Form and Content of Thoughts')
plt.tight_layout()


# In[31]:


fig.savefig('./figures/Figure05_AC-CPMsubject-level-acc.png')


# In[32]:


median_width = 0.4
sns.set(style='whitegrid')
fig,ax = plt.subplots(1,3 ,figsize=(8,5), gridspec_kw={'width_ratios': [1,2,3]})
# Vigilance
sns.boxenplot(data=null_df[null_df['Question']=='Wakefulness'],x='Question',y='R', color='lightgray', ax=ax[0]) 
sns.stripplot(data=real_df[real_df['Question']=='Wakefulness'],x='Question', y='R', alpha=.5, ax=ax[0])
plt.xticks(rotation=45);
for tick, text in zip(ax[0].get_xticks(), ax[0].get_xticklabels()):
    # Add Black Line Signaling Median
    question   = text.get_text()
    median_val = accuracy_real[question].median().values[0]
    print(text, '%.2f' % median_val)
    ax[0].plot([tick-median_width/2, tick+median_width/2],[median_val,median_val], lw=4, color='k')
    # Statistical Significant Information
    p = p_values.loc[question,'Non Parametric']
    if 5.00e-02 < p <= 1.00e+00:
        annot = '' 
    elif 1.00e-02 < p <= 5.00e-02:
        annot = '*'
    elif 1.00e-03 < p <= 1.00e-02:
        annot = '**'
    elif 1.00e-04 < p <= 1.00e-03:
        annot = '***'
    elif p <= 1.00e-04:
        annot = '****'
    max_val = real_df.set_index('Question').max()['R']
    ax[0].annotate(annot, xy=(tick, max_val+0.02), ha='center', fontsize=15)
    
ax[0].set_ylim(-.3,.4)
ax[0].set_ylabel('R (Observed,Predicted)');
ax[0].set_xlabel('')

# Factors
sns.boxenplot(data=null_df[(null_df['Question']=='Thought Pattern 1') | (null_df['Question']=='Thought Pattern 2')],x='Question',y='R', color='lightgray', ax=ax[1]) 
sns.stripplot(data=real_df[(real_df['Question']=='Thought Pattern 1') | (real_df['Question']=='Thought Pattern 2')],x='Question', y='R', alpha=.5, ax=ax[1])
plt.xticks(rotation=45);
for tick, text in zip(ax[1].get_xticks(), ax[1].get_xticklabels()):
    # Add Black Line Signaling Median
    question   = text.get_text()
    median_val = accuracy_real[question].median().values[0]
    print(text, '%.2f' % median_val)
    ax[1].plot([tick-median_width/2, tick+median_width/2],[median_val,median_val], lw=4, color='k')
    # Statistical Significant Information
    p = p_values.loc[question,'Non Parametric']
    if 5.00e-02 < p <= 1.00e+00:
        annot = '' 
    elif 1.00e-02 < p <= 5.00e-02:
        annot = '*'
    elif 1.00e-03 < p <= 1.00e-02:
        annot = '**'
    elif 1.00e-04 < p <= 1.00e-03:
        annot = '***'
    elif p <= 1.00e-04:
        annot = '****'
    max_val = real_df.set_index('Question').max()['R']
    ax[1].annotate(annot, xy=(tick, max_val+0.02), ha='center', fontsize=15)

ax[1].set_ylim(-.3,.4)
ax[1].set_xticklabels(['Pattern 1','Pattern 2'])
ax[1].set_ylabel('R (Observed,Predicted)');
ax[1].set_xlabel('Thought Patterns')

# Significant Individual Iterms
sns.boxenplot(data=null_df[(null_df['Question']=='Images') | (null_df['Question']=='Surroundings') | (null_df['Question']=='Past')],x='Question',y='R', color='lightgray', ax=ax[2]) 
sns.stripplot(data=real_df[(real_df['Question']=='Images') | (real_df['Question']=='Surroundings') | (real_df['Question']=='Past')],x='Question', y='R', alpha=.5, ax=ax[2])
plt.xticks(rotation=45);
for tick, text in zip(ax[2].get_xticks(), ax[2].get_xticklabels()):
    # Add Black Line Signaling Median
    question   = text.get_text()
    median_val = accuracy_real[question].median().values[0]
    ax[2].plot([tick-median_width/2, tick+median_width/2],[median_val,median_val], lw=4, color='k')
    # Statistical Significant Information
    p = p_values.loc[question,'Non Parametric']
    if 5.00e-02 < p <= 1.00e+00:
        annot = '' 
    elif 1.00e-02 < p <= 5.00e-02:
        annot = '*'
        print(text, '%.2f' % median_val)
    elif 1.00e-03 < p <= 1.00e-02:
        annot = '**'
        print(text, '%.2f' % median_val)
    elif 1.00e-04 < p <= 1.00e-03:
        annot = '***'
        print(text, '%.2f' % median_val)
    elif p <= 1.00e-04:
        annot = '****'
        print(text, '%.2f' % median_val)
    max_val = real_df.set_index('Question').max()['R']
    ax[2].annotate(annot, xy=(tick, max_val+0.02), ha='center', fontsize=15)
    
ax[2].set_ylim(-.3,.4)
ax[2].set_ylabel('R (Observed,Predicted)');
ax[2].set_xlabel('SNYC Questionnaire Items')
plt.tight_layout()


# In[34]:


fig.savefig('./figures/CPMsubject-level-acc-fortalks.png')


# ## 4.3. Scatter Plots of Observed vs. Predicted Values

# In[35]:


N_sign_results = (p_values.loc[:,'Non Parametric'] < 0.05).sum()
print('++ INFO: Number of items predicted significantly: %d ' % N_sign_results)


# In[36]:


fig,ax = plt.subplots(2,int(N_sign_results/2),figsize=(16,10))
i = 0
for BEHAVIOR in BEHAVIOR_LABELS_DICT.values():
    p = p_values.loc[BEHAVIOR,'Non Parametric']
    if p <= 0.05:
        row,col        = np.unravel_index(i,(2,int(N_sign_results/2)))
        behav_obs_pred = pd.DataFrame(real_predictions_xr.median(dim='Iteration').loc[BEHAVIOR,:,['observed','predicted (glm)']], columns=['observed','predicted (glm)'])
        r,p = plot_predictions(behav_obs_pred, ax=ax[row,col], xlabel='Observed [%s]' % BEHAVIOR, ylabel='Predicted [%s]' % BEHAVIOR, font_scale=1,p_value=p_values.loc[BEHAVIOR,'Non Parametric'], 
                               ylim=(behav_obs_pred['predicted (glm)'].min(), behav_obs_pred['predicted (glm)'].max()),
                               xlim=(behav_obs_pred['observed'].min(), behav_obs_pred['observed'].max()))
        i= i + 1


# In[37]:


plt.tight_layout()
fig.savefig('./figures/CPMsubject-level-scatters-for-talks.png')

