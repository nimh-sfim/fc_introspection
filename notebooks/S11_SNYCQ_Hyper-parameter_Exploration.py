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

import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import hvplot.pandas
from utils.basics import PRJ_DIR
import holoviews as hv

result_list = glob.glob('{PRJ_DIR}/data/snycq/results/*.npy'.format(PRJ_DIR=PRJ_DIR))
print('++ Number of files available: %d' % len(result_list))

# %%time
cv_results = pd.DataFrame(columns=['Dimension','Sparsity','Repeat','CV','Error Type','Error'])
for (z, f) in enumerate(result_list):
    output = np.load(f)
    rep, cv, dim, sparsity,train_error,valid_error = int(output[0]), int(output[1]), int(output[2]), output[3], output[4], output[5]
    cv_results = cv_results.append({'Dimension':dim, 'Sparsity':sparsity, 'Repeat':rep, 'CV':cv, 'Error Type':'Training',   'Error':train_error}, ignore_index=True)
    cv_results = cv_results.append({'Dimension':dim, 'Sparsity':sparsity, 'Repeat':rep, 'CV':cv, 'Error Type':'Validation', 'Error':valid_error}, ignore_index=True)
cv_results = cv_results.infer_objects()

cv_valid_error = cv_results[cv_results['Error Type']=='Validation']
cv_train_error = cv_results[cv_results['Error Type']=='Training']

sparsity_to_explore = cv_valid_error['Sparsity'].unique()
sparsity_to_explore.sort()
print(['%.3f' % s for s in sparsity_to_explore])

# +
fig, ax = plt.subplots(2,1,figsize=(16,8))
sns.set(style='whitegrid')
for sparsity in sparsity_to_explore:
    aux = cv_valid_error[cv_valid_error['Sparsity']==sparsity]
    sns.lineplot(data=aux,x='Dimension', y='Error', style='Sparsity', ax=ax[0], label=str(sparsity), legend=False, markers=False, estimator='mean')
ax[0].set_ylabel('Frobenius norm (log-scale)')
ax[0].set_yscale('log')
ax[0].legend(prop={'size': 11}, loc='center', bbox_to_anchor=(0.5, 0.8), ncol=10, fancybox=True, shadow=True)
ax[0].set_title('Reconstruction Error', y=.9);

for sparsity in sparsity_to_explore:
    aux = cv_train_error[cv_train_error['Sparsity']==sparsity]
    sns.lineplot(data=aux,x='Dimension', y='Error', style='Sparsity', ax=ax[1], label=str(sparsity), legend=False, markers=False, estimator='mean')
ax[1].set_ylabel('Frobenius norm (log-scale)')
ax[1].set_yscale('log')
ax[1].legend(prop={'size': 11}, loc='center', bbox_to_anchor=(0.5, 0.1), ncol=10, fancybox=True, shadow=True)
ax[1].set_title('Training Error', y=.9);

# +
fig, ax = plt.subplots(1,1,figsize=(16,8))
sns.set(style='whitegrid')
for sparsity in sparsity_to_explore:
    aux = cv_valid_error[cv_valid_error['Sparsity']==sparsity]
    sns.lineplot(data=aux,x='Dimension', y='Error', style='Sparsity', ax=ax, label=str(sparsity), legend=False, markers=False, estimator='mean',c='k')
ax.set_ylabel('Frobenius norm (log-scale)')
ax.set_yscale('log')
#ax.legend(prop={'size': 11}, loc='center', bbox_to_anchor=(0.5, 0.8), ncol=10, fancybox=True, shadow=True)

for sparsity in sparsity_to_explore:
    aux = cv_train_error[cv_train_error['Sparsity']==sparsity]
    sns.lineplot(data=aux,x='Dimension', y='Error', style='Sparsity', ax=ax, label=str(sparsity), legend=False, markers=False, estimator='mean',c='r')
ax.set_ylabel('Frobenius norm (log-scale)')
ax.set_yscale('log')
#ax.legend(prop={'size': 11}, loc='center', bbox_to_anchor=(0.5, 0.1), ncol=10, fancybox=True, shadow=True)
ax.set_title('Training (Red) & Recon/Validation Error (Black)', y=.9);
# -

cv_valid_error.drop(['Repeat','CV','Error Type'], axis=1).groupby(['Dimension','Sparsity']).median().sort_values(by='Error')[0:10]

cv_train_error.drop(['Repeat','CV','Error Type'], axis=1).groupby(['Dimension','Sparsity']).median().sort_values(by='Error')[0:10]

(cv_results.hvplot.kde(y='Error',by='Error Type', title='All Values Explored for D & S').opts(legend_position='right') + \
cv_results.set_index('Dimension').loc[2].hvplot.kde(y='Error',by='Error Type',title='D=2 & All S').opts(legend_position='right') + \
cv_results.set_index('Sparsity').loc[1.0].hvplot.kde(y='Error',by='Error Type',title='All D & S=1.0').opts(legend_position='right') + \
cv_results.set_index(['Dimension','Sparsity']).loc[2,1.0].hvplot.kde(y='Error',by='Error Type',title='D=2 & S=1.0').opts(legend_position='right')).cols(2)

cv_valid_error[cv_valid_error['Error']==0].drop(['Repeat','CV','Error Type'],axis=1).value_counts()

cv_valid_error_D2 = cv_valid_error.set_index('Sparsity').loc[0].reset_index(drop=True)
opt_median_dim    = int(cv_valid_error_D2.groupby(by='Dimension').median().sort_values(by='Error').iloc[0].name)
opt_mean_dim    = int(cv_valid_error_D2.groupby(by='Dimension').mean().sort_values(by='Error').iloc[0].name)

fig, ax = plt.subplots(1,2,figsize=(12,4))
# Median
sns.lineplot(data = cv_valid_error_D2, x='Dimension', y='Error', estimator='median', ax=ax[0])
ax[0].set_title('Estimator = Median')
ax[0].annotate('Optimal Dim = {d}'.format(d=opt_median_dim),xy=(2,320000), bbox={'facecolor': 'lightblue', 'alpha': 0.5, 'pad': 10})
ax[0].set_ylim(150000, 350000)
# Mean
sns.lineplot(data = cv_valid_error_D2, x='Dimension', y='Error', estimator='mean', ax=ax[1])
ax[1].set_title('Estimator = Mean')
ax[1].annotate('Optimal Dim = {d}'.format(d=opt_mean_dim),xy=(2,320000), bbox={'facecolor': 'lightblue', 'alpha': 0.5, 'pad': 10})
ax[1].set_ylim(150000, 350000)

cv_valid_error_D2.groupby(by='Dimension').median().sort_values(by='Error').iloc[0].name

cv_valid_error = cv_valid_error.set_index(['Dimension','Sparsity'])

cv_valid_error = cv_valid_error[cv_valid_error['CV']!=9]

cv_valid_error

cv_valid_error.loc[2,1.0].hvplot.kde(y='Error', by='Repeat',title='Evalid | D=2 | S=1.0').opts(legend_position='bottom') + cv_valid_error.loc[2,1.0].hvplot.kde(y='Error', by='CV')

(cv_valid_error.loc[2,1.0].sort_values(by='Repeat').hvplot.kde(y='Error', by='Repeat',title='Validation Error | D=2 | S=1.0').opts(legend_position='left', height=600) + \
cv_valid_error.loc[2,1.0].sort_values(by='CV').hvplot.kde(y='Error', by='CV',title='Validation Error | D=2 | S=1.0').opts(legend_position='left', height=300) + \
cv_valid_error.loc[2,1.0].hvplot.kde(y='Error',title='Validation Error | D=2 | S=1.0')).cols(1)


