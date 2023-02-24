# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: FC Introspection (Jan 2023)
#     language: python
#     name: fc_introspection
# ---

# # Description
#
# This notebook loads the result of running 100 iterations of the CPM algorithm on real data, and also those of running 10,000 iterations with randomized labels (null distribution).
#
# Using these data, the notebook then computes non-parametric p-values for each prediction.
#
# Finally, the notebook generates summary figures for the ability of CPM to predict experiential variables.
#
# > **NOTE:** Although the notebook loads and computes values for the three CPM models (pos, neg and glm), ultimately on the paper we only report results for the glm case.

import pandas as pd
import os.path as osp
from utils.basics import RESOURCES_CPM_DIR
import hvplot.pandas
from tqdm import tqdm
import holoviews as hv
import xarray as xr
import numpy as np
import pickle
from utils.basics import get_sbj_scan_list
from cpm.plotting import plot_predictions
import seaborn as sns
import panel as pn
import matplotlib.pyplot as plt

KFOLD_METHOD         = 'basic'     # basic or subject_aware
CORR_TYPE            = 'pearson'
E_SUMMARY_METRIC     = 'sum'
CPM_NITERATIONS      = 100
CPM_NULL_NITERATIONS = 10000
BEHAVIOR_LIST        = ['Images','Words','People','Myself','Positive','Negative','Surroundings','Intrusive','Vigilance','Future','Past','Specific']

# ## 1. Load CPM results for real models (100 iterations)
#
# For each behavior and iteration:
#
# 1. Load predicted and observed values.
# 2. Add those to an xr.DataArray for plotting later.
# 3. Compute the correlation between observed and predicted values and save it into a dataframe.
#
# The first two cells load results fresh from this. The third cell is an alternative that relies on a pre-compiled structure. You should only run cell 3 in case you are trying to save time while testing things

sbj_list, scan_list = get_sbj_scan_list(when='post_motion', return_snycq=False)
scan_idx = [sbj+'.'+run for (sbj,run) in scan_list]

# +
real_pred, real_pred_r    = {},{}
predictions_xr = xr.DataArray(dims   = ['Behavior','Iteration','Scan','Type'], 
                              coords = {'Behavior':BEHAVIOR_LIST, 'Iteration':range(CPM_NITERATIONS), 'Scan':scan_idx,'Type':['observed','predicted (glm)']})

for BEHAVIOR in BEHAVIOR_LIST:
    real_pred[BEHAVIOR] = pd.DataFrame(index=range(CPM_NITERATIONS),columns=['pos','neg','glm'])
    for r in tqdm(range(CPM_NITERATIONS), desc='Iteration [%s]' % BEHAVIOR):
        path = osp.join(RESOURCES_CPM_DIR,'results',BEHAVIOR,CORR_TYPE, E_SUMMARY_METRIC,'cpm_{b}_rep-{r}.pkl'.format(b=BEHAVIOR,r=str(r+1).zfill(5)))
        try:
            with open(path,'rb') as f:
                data = pickle.load(f)
        except:
            continue
        # Access the DataFrame with the observed and predicted values
        pred = data['behav_obs_pred']
        # Save all observed and predicted values
        predictions_xr.loc[BEHAVIOR,r,:,'observed']        = pred[BEHAVIOR+' observed'].values # Kind of redundant, but handy later when averaging across dimensions
        predictions_xr.loc[BEHAVIOR,r,:,'predicted (glm)'] = pred[BEHAVIOR+' predicted (glm)'].values
        # For each mode, compute the correlation between observed and predicted
        aux  = {}
        for tail in ['pos','neg','glm']:
            real_pred[BEHAVIOR].loc[r,tail] = pred[BEHAVIOR+' predicted ('+tail+')'].corr(pred[BEHAVIOR+' observed'])
        real_pred[BEHAVIOR]                 = real_pred[BEHAVIOR].infer_objects()
        # Summary correlation values between observed and predicted behavior using median as in in Finn et al. NI 2021
        real_pred_r[BEHAVIOR,'pos'] = real_pred[BEHAVIOR]['pos'].median()
        real_pred_r[BEHAVIOR,'neg'] = real_pred[BEHAVIOR]['neg'].median()
        real_pred_r[BEHAVIOR,'glm'] = real_pred[BEHAVIOR]['glm'].median()
        
# Save to disk as a single structure to save time        
out_path = '../resources/cpm/plot_tmp/real_predictions.pkl'
print('++ INFO: Saving generated data structures to disk [%s]' % out_path)
data_to_disk = {'real_pred_r':real_pred_r, 'predictions_xr':predictions_xr, 'real_pred':real_pred}
with open(out_path,'wb') as f:
    pickle.dump(data_to_disk,f)
# -

out_path       = '../resources/cpm/plot_tmp/real_predictions.pkl'
with open(out_path,'rb') as f:
    data_from_disk = pickle.load(f)
real_pred_r    = data_from_disk['real_pred_r']
predictions_xr = data_from_disk['predictions_xr']
real_pred      = data_from_disk['real_pred']

# ## 2. Load CPM results for the null distribution (10,000 iterations)
#
# Once those are in memory, we will compute summary non-parametric p-values.

# %%time
null_pred_r = {}
p_values    = {}
# For all behaviors
for BEHAVIOR in BEHAVIOR_LIST:
    null_pred_r[BEHAVIOR] = pd.DataFrame(index=range(CPM_NULL_NITERATIONS),columns=['pos','neg','glm'])
    for r in tqdm(range(CPM_NULL_NITERATIONS),desc='Iteration [%s]' % BEHAVIOR):
        # Generate path to results for a given iteration
        path = osp.join(RESOURCES_CPM_DIR,'null_distribution',BEHAVIOR,CORR_TYPE, E_SUMMARY_METRIC,'cpm_{b}_rep-{r}_NULL.pkl'.format(b=BEHAVIOR,r=str(r+1).zfill(5)))
        # Load results for such iteration
        with open(path,'rb') as f:
            data = pickle.load(f)
        # Extract the dataframe that contains observed and predicted values
        pred = data['behav_obs_pred']
        aux  = {}
        # For each modeling scenario:
        for tail in ['pos','neg','glm']:
            # Calculate the correlation between observed and predicted (under randomization conditions) and add the result to a dataframe
            null_pred_r[BEHAVIOR].loc[r,tail] = pred[BEHAVIOR+' predicted ('+tail+')'].corr(pred[BEHAVIOR+' observed'])
        null_pred_r[BEHAVIOR] = null_pred_r[BEHAVIOR].infer_objects()
    # Compute non-parametric p-values
    for tail in ['pos','neg','glm']:
        p_values[BEHAVIOR,tail] = ((null_pred_r[BEHAVIOR][tail] > real_pred_r[BEHAVIOR,tail]).sum() + 1) / (CPM_NULL_NITERATIONS + 1)
# Save everything to disk
out_path = '../resources/cpm/plot_tmp/null_pred_r.pkl'
data_to_disk = {'null_pred_r':null_pred_r,'p_values':p_values}
with open(out_path,'wb') as f:
    pickle.dump(data_to_disk,f)

out_path    = '../resources/cpm/plot_tmp/null_pred_r.pkl'
with open(out_path,'rb') as f:
    data_from_disk = pickle.load(f)
p_values    = data_from_disk['p_values']
null_pred_r = data_from_disk['null_pred_r']
del data_from_disk

# ***
#
# ## 3. Generate Prediction-reporting Figures

behav_select = pn.widgets.Select(name='Question', options=BEHAVIOR_LIST, value=BEHAVIOR_LIST[0])

null_df = pd.DataFrame(columns=['Question','Iteration','R'])
for BEHAVIOR in BEHAVIOR_LIST:
    for i in tqdm(range(CPM_NULL_NITERATIONS), desc='Iteration [%s]' % BEHAVIOR):
        null_df = null_df.append({'Question':BEHAVIOR,'Iteration':i,'R':null_pred_r[BEHAVIOR].loc[i,'glm']}, ignore_index=True)

real_df = pd.DataFrame(columns=['Question','Iteration','R'])
for BEHAVIOR in BEHAVIOR_LIST:
    for i in tqdm(range(CPM_NITERATIONS), desc='Iteration [%s]' % BEHAVIOR):
        real_df = real_df.append({'Question':BEHAVIOR,'Iteration':i,'R':real_pred[BEHAVIOR].loc[i,'glm']}, ignore_index=True)

median_width = 0.4
sns.set(style='whitegrid')
fig,ax = plt.subplots(1,1,figsize=(15,5))
sns.boxenplot(data=null_df,x='Question',y='R', color='lightgray', ax=ax) 
sns.stripplot(data=real_df,x='Question', y='R', alpha=.8, ax=ax)
plt.xticks(rotation=45);
for tick, text in zip(ax.get_xticks(), ax.get_xticklabels()):
    question   = text.get_text()
    median_val = real_pred_r[question,'glm']
    ax.plot([tick-median_width/2, tick+median_width/2],[median_val,median_val], lw=4, color='k')
ax.set_ylim(-.5,.5)
ax.set_ylabel('R (Observed,Predicted)');
ax.set_xlabel('SNYCQ Item')

fig,ax = plt.subplots(3,4,figsize=(20,12))
for i,BEHAVIOR in enumerate(BEHAVIOR_LIST):
    row,col        = np.unravel_index(i,(3,4))
    behav_obs_pred = pd.DataFrame(predictions_xr.mean(dim='Iteration').loc[BEHAVIOR].values, columns=['observed','Predicted (glm)'])
    r,p = plot_predictions(behav_obs_pred, ax=ax[row,col], xlabel='Observed [%s]' % BEHAVIOR,ylabel='Predicted [%s]' % BEHAVIOR, p_value=p_values[BEHAVIOR,'glm'], font_scale=1)
