# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: FC Instrospection (2023 | 3.10)
#     language: python
#     name: fc_introspection_2023_py310
# ---

# +
import os
import numpy as np
import pandas as pd
import sys

from matplotlib import pyplot
from matplotlib import cm
import plotly.graph_objects as go
import plotly.figure_factory as ff
#from plotly.subplots import make_subplots
import seaborn as sns

from scipy.stats import f_oneway
import scikit_posthocs as sp
# -

import hvplot.pandas
from plotly.express import scatter_3d


# +
def load_errlist(csv_filepath):
    stats = pd.read_csv(csv_filepath)
    
    repeat_idx = np.unique(stats['repeat'].values)
    d_list = np.unique(stats['dimension'].values)
    W_beta_list = np.unique(stats['W_beta'].values)
    Q_beta_list = np.unique(stats['Q_beta'].values)

    train_err_list = []
    valid_err_list = []

    for d in d_list:
        for bw in W_beta_list:
            for bq in Q_beta_list:
                for r in repeat_idx:
                    per_stats = stats.loc[ (stats['repeat'] == r) &
                                           (stats['dimension'] == d) &
                                           (stats['W_beta'] == bw) &
                                           (stats['Q_beta'] == bq) ]

                    if per_stats.shape[0] > 0:
                        per_stats = per_stats.mean()
                        train_err_list.append([d, bw, bq, per_stats['train_error']]) 
                        valid_err_list.append([d, bw, bq, per_stats['valid_error']])
                    #else:
                    #    print(d, bw, bq, r)

    train_err_list = np.stack(train_err_list)
    valid_err_list = np.stack(valid_err_list)
    train_err_list = pd.DataFrame(train_err_list, columns=['dimension','bw','bq','error'])
    valid_err_list = pd.DataFrame(valid_err_list, columns=['dimension','bw','bq','error'])
    return stats, train_err_list, valid_err_list

def show_stats(train_err_list, valid_err_list):
    for bw in np.unique(valid_err_list['bw'].values):
        nq = len(np.unique(valid_err_list['bq'].values))
        fig, ax = pyplot.subplots(nrows=1, ncols=nq, figsize=(10, 2), sharey=True)

        for (idx,bq) in enumerate(np.unique(valid_err_list['bq'].values)):
            sns.lineplot(data=valid_err_list.loc[(valid_err_list['bw'] == bw) & (valid_err_list['bq'] == bq)],
                         x="dimension", y="error", hue='bq', palette="tab10", markers=True, ax=ax[idx], legend=False)
            sns.lineplot(data=train_err_list.loc[(train_err_list['bw'] == bw) & (train_err_list['bq'] == bq)],
                         x="dimension", y="error", hue='bq', palette=sns.color_palette("hls", 1), markers=True, ax=ax[idx], legend=False)
            if idx == 0:
                ax[idx].set_title('beta W={}\n beta Q={}'.format(bw, bq))
            else:
                ax[idx].set_title('beta Q={}'.format(bq))
            ax[idx].set_xticks(np.unique(valid_err_list['dimension'].values))
        pyplot.show()

        
def show_distribution(stats):
    
    repeat_idx = np.unique(stats['repeat'].values)
    d_list = np.unique(stats['dimension'].values)
    W_beta_list = np.unique(stats['W_beta'].values)
    Q_beta_list = np.unique(stats['Q_beta'].values)
    
    output_validation = []

    for d in d_list:
        train_err_list = []
        valid_err_list = []
        for r in repeat_idx:
            for bw in W_beta_list:
                for bq in Q_beta_list:
                    per_stats = stats.loc[ (stats['repeat'] == r) &
                                           (stats['dimension'] == d) &
                                           (stats['W_beta'] == bw) &
                                           (stats['Q_beta'] == bq) ]

                    if len(per_stats) > 0:
                        per_stats = per_stats.mean()
                        train_err_list.append([d, bw, bq, per_stats['train_error']])
                        valid_err_list.append([d, bw, bq, per_stats['valid_error']])

        train_err_list = np.stack(train_err_list)
        valid_err_list = np.stack(valid_err_list)

        for i in range(len(W_beta_list)):
            distribution_list = []
            setting_list = []

            fig, ax = pyplot.subplots(nrows=1, ncols=len(Q_beta_list), figsize=(10, 1), sharey=True)

            for j in range(len(Q_beta_list)):
                bw = W_beta_list[i]
                bq = Q_beta_list[j]
                idx = np.where( (valid_err_list[:,1] == bw) & (valid_err_list[:,2] == bq) )[0]

                sns.histplot(valid_err_list[idx,3], ax=ax[j], kde=True, bins=20)
                ax[j].set_title(str(bq)+', '+str(np.round(np.mean(valid_err_list[idx,3]),3)))
                if j == 0:
                    ax[j].set_ylabel('d={}\n beta W={}'.format(d, bw))
                ax[j].set_yticks([])
                
                # if d == 2 and j == 2:
                if d == 2:
                    output_validation.append(valid_err_list[idx,3])
                
            pyplot.show()
    return output_validation


# -

# # 1. Load Inputs
#
# ## 1.1. Load Questionnaire item labels

question_list = pd.read_csv('./mlt/data/snycq.csv').columns[2:]
print(question_list)
print(len(question_list))

# ## 1.2. Load Original Demographic information

demo_csv = pd.read_csv('./mlt/data/participants_post_motion_QA.csv')
demo_csv.head(5)

# ## 1.3. Load sNYCQ Answers

data_csv = pd.read_csv('./mlt/data/snycq.csv', index_col=['Subject','Run'])
print(data_csv.shape)
data_csv.head(3)

# # 2. Load Outputs (provided by the MLT group) - Confound Modeling Scenario
#
# MLT provided us with two solutions, one modeling demographic confounds, and one without. We decided to work with the one with confounds becuase it seems a more complete solution

data = np.load('./mlt/output/full_data.npz')
M        = data['M']          # Same as data_csv
nan_mask = data['nan_mask']   # All ones, as there are no missing values
confound = data['confound']   # Original encoding of Gender (1=Male, 0=Female) & Age (0.07 = 20-25 | 0.21 = 25-30 | 0.36 = 30-35 | 0.5 = 35-40 | 0.64 = 40-45 | 0.79 = 45-50 | 0.93 = 50-55 | 1 = 60-65) per scan

# Sanity check, that inputs agree

assert np.all(np.equal(M,data_csv.values)), "++ ERROR: The data I provided to MLT, and the data they used as input to the sbcNNMF algorithm differs"

# Load and show the demographic information

confound_df = pd.DataFrame(confound, index=data_csv.index, columns=['Age','Gender'])
confound_df.head(5)

# ### Search Space
# | Hyperparameter | Symbol | Space |
# |----------------|-----|---------|
# | Dimensionality | d | 1,2,3,4 |
# | Sparsity for W | $$\beta_{W}$$ | 0.0, 0.01, 0.1, 1, 2, 3 |
# | Sparsity for Q | $$\beta_{Q}$$ | 0.0, 0.01, 0.1, 1, 2, 3 |
# | Confound Combination | |  No Confound or Intercept modeling, Intercept modeling only, Confound & Intercept modeling |
#
# Number of iterations = 50
# Number of folds = 10

# # 3. Results for full confound modeling

stats, train_err_list, valid_err_list = load_errlist('./mlt/output/biowulf/stats_full_data.csv')

# Print number of missing --> For some combinations of (d, b_W, b_Q), in some iterations no folds completed. Why?
N_expected = 4 * 6 * 6 * 50 * 10
N_expected - stats.shape[0]

valid_err_list.head(5)

# # 4. Create Panels in Suppl. Fig. 3
#
# The goal of these plots is to show that we selected the best hyper-parameters for the sbcNNMF algorithm

# +
a = valid_err_list.copy()
a['bw'] = a['bw'].astype('str')
a['bq'] = a['bq'].astype('str')
a['dimension'] = a['dimension'].astype('str')

plot3d = scatter_3d(a,x='bw',y='bq',z='error', color='dimension', width=600, height=600, color_discrete_map={'1':'blue','2':'red','3':'green','4':'black'}, opacity=0.8)
plot3d.update_layout(scene = dict(
                    xaxis_title='Sparsity in W (βw)',
                    yaxis_title='Sparsity in Q (βq)',
                    zaxis_title='Validation Error'),
                    width=700,
                    margin=dict(r=20, b=10, l=10, t=10), legend_title='Number of Dimensions (d)')
plot3d
# -

fig, ax = pyplot.subplots(1,1,figsize=(5,5))
a = valid_err_list.copy()
a = a[a['dimension']==2]
a.drop('dimension',axis=1,inplace=True)
a['bw'] = a['bw'].astype('str')
a['bq'] = a['bq'].astype('str')
plot = sns.lineplot(data=a,y='error',x='bw', hue='bq', ax=ax)
ax.legend(loc='upper left',ncol=2, title=r"$\beta_{Q}$")
ax.set_ylabel('Validation Error')
ax.set_xlabel(r"$\beta_{W}$")


