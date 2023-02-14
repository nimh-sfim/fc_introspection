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

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import hvplot.pandas

import sys
sys.path.append('./mlt/')
from NMF_algorithm import NMF_model

from utils.basics import PROC_SNYCQ_DIR, get_sbj_scan_list
from utils.basics import SNYCQ_Questions, SNYCQ_Question_type
from utils.SNYCQ_NMF_Extra import plot_anwers_distribution_per_question, plot_Q_bars

SBJs, SCANs, SNYCQ = get_sbj_scan_list(when='post_motion', return_snycq=True)

SNYCQ = SNYCQ.drop('Vigilance',axis=1)
Nscans, Nquestions = SNYCQ.shape
print(SNYCQ.shape)

SNYCQ.to_csv('/data/SFIMJGC_Introspec/2023_fc_introspection/data/shared_with_mlt/snycq.csv')

PROC_SNYCQ_DIR

print(SNYCQ.isna().sum())

# ***
#
# # 1. Plot the data in different meaningful ways
#
# ## 1.1. Plot distribution of answers per question

# + tags=[]
plot_anwers_distribution_per_question(SNYCQ, figsize=(10,4), legend_fontsize=10)
# -

# ## 1.2. Plot the answers as a heatmap

data_to_plot              = pd.DataFrame(SNYCQ.values)
data_to_plot.index        = np.arange(SNYCQ.shape[0])
data_to_plot.index.name   = 'Questions'
data_to_plot.columns      = SNYCQ.columns
data_to_plot.columns.name = 'Scans'
x_ticks       = [0, 99, 199, 299, 399, Nscans-1]
x_tick_labels = ['1','100','200','300','400', str(Nscans)]
x_ticks_info  = list(tuple(zip(x_ticks,x_tick_labels)))
data_to_plot.hvplot.heatmap(width=400, height=800, cmap='viridis', yticks= x_ticks_info, fontscale=1.5, xlabel='Questions', ylabel='Scans', title='P').opts(xrotation=90, colorbar_opts={'title':'Response Value:'})

# ## 1.3. Explore the relationship between questions via simple correlation

snycq_corr = SNYCQ.corr().infer_objects()
snycq_corr.index.name='Questions'
snycq_corr.columns.name='Questions'

sns.set(font_scale=1.2)
clmap = sns.clustermap(snycq_corr.round(1), annot=True, cmap='RdBu_r', figsize=(8,8), vmin=-.5, vmax=.5, cbar_kws = {'orientation':'horizontal'}, cbar_pos=(0,0,1,.01), method='ward');
clmap.ax_row_dendrogram.set_visible(False)
clmap.ax_col_dendrogram.set_visible(False)
sorted_questions = list(SNYCQ.columns[clmap.dendrogram_col.reordered_ind])
print(sorted_questions)

g = sns.PairGrid(SNYCQ[sorted_questions], diag_sharey=False, height=1)
g.map_upper(sns.scatterplot, s=5)
g.map_lower(sns.kdeplot)
g.map_diag(sns.kdeplot, lw=1)

# ***
# # 2. Basic Attempts at decomposition

P = SNYCQ.values.astype('float')
print('Range of data matrix P:      [min={}, max={}]'.format(np.min(P), np.max(P)))
print('Dimensions of data matrix P: [#scans={}, #questions={}]'.format(P.shape[0],P.shape[1]))
assert np.sum(np.isnan(P)) == 0 # Make sure there are no missing values

Ws,Qs, models = {},{},{}
sparsity_to_explore = [0, 1e-3, 1e-2, 1e-1, .25, .5, .75, 1, 2, 5, 10]
sparsity_to_explore = [0, 1e-4, 1e-3, 1e-2, 2e-2, 3e-2, 4e-2, 5e-2, 6e-2, 7e-2, 8e-2, 9e-2, 1e-1, 2e-1, 3e-1,4e-1,5e-1,6e-1,7e-1,8e-1,9e-1, 1, 2, 5, 10]

# ## 2.1. Compute the model for DIM=2 and several values of Sparsity

for sparsity in sparsity_to_explore:
    models[sparsity] = NMF_model(P, np.ones_like(P), dimension=2, method='admm', Wbound=(True, 1.0), sparsity_parameter=sparsity)
    Ws[sparsity], Qs[sparsity] = models[sparsity].decomposition()

# ## 2.2. Generate Plots for Q

for sparsity in sparsity_to_explore:
    fig, axes = plt.subplot_mosaic("AAAAABBBBBCC", figsize=(10,4))
    g = sns.heatmap(Qs[sparsity], cmap='viridis', vmax=100, vmin=0, annot=True, ax=axes['C'], annot_kws={'size': 10})
    g.xaxis.set_ticks([]);
    g.yaxis.set_ticks([]);
    cbar = g.collections[0].colorbar
    cbar.ax.tick_params(labelsize=10)
    g.invert_yaxis()
    _ = plot_Q_bars(Qs[sparsity],SNYCQ, SNYCQ_Questions, SNYCQ_Question_type, figsize_x=8, fontsize=8, axes=axes)
    plt.close()
    fig.savefig('figures/Q_{s}.png'.format(s=str(sparsity)), bbox_inches='tight')

# ## 2.3 Generate Plots for W

for sparsity in sparsity_to_explore:
    fig,ax = plt.subplots(1,1,figsize=(5,5))
    aux = pd.DataFrame(Ws[sparsity], columns=['Factor 1','Factor 2'], index=SNYCQ.index)
    g = sns.scatterplot(data=aux,x='Factor 1', y='Factor 2', s=15)
    g.set_xlim(-.005,1.005)
    g.set_ylim(-.005,1.005)
    g.grid()
    plt.close()
    fig.savefig('./figures/W_scatter_{s}.png'.format(s=str(sparsity)), bbox_inches='tight')

rmse = pd.Series(index=sparsity_to_explore,name='RMSE', dtype='float')
for sparsity in sparsity_to_explore:
    this_W = Ws[sparsity]
    this_Q = Qs[sparsity]
    Precon = pd.DataFrame(np.dot(this_W,this_Q.T),index=SNYCQ.index, columns=SNYCQ.columns)
    Error  = np.abs(SNYCQ - Precon)
    fig,axs = plt.subplots(1,2,figsize=(16,2))
    # Plot Reconstructed Data
    plot = sns.heatmap(Precon.T, cmap='viridis', cbar=True, cbar_kws = dict(pad=0.01), vmin=0, vmax=100, ax=axs[0])
    axs[0].set_xticks([0, 100,200,300,400])
    axs[0].set_xticklabels([0, 100,200,300,400])
    axs[0].set_xlabel('Scans Number', fontsize=10)
    axs[0].set_ylabel('Questions', fontsize=10)
    axs[0].set_title('$P_{reconstructed}^T$', fontsize=10)
    axs[0].tick_params(axis='both', which='major', labelsize=10)
    #Plot Difference
    plot = sns.heatmap(Error.T, ax=axs[1], cmap='viridis', cbar=True, cbar_kws = dict(pad=0.01), vmin=0, vmax=100)
    axs[1].set_xticks([0, 100,200,300,400])
    axs[1].set_xticklabels([0, 100,200,300,400])
    axs[1].set_xlabel('Scans Number', fontsize=10)
    axs[1].set_ylabel('Questions', fontsize=10)
    axs[1].set_title('abs($P^T$ - $P_{reconstructed}^T$)', fontsize=10)
    axs[1].tick_params(axis='both', which='major', labelsize=10)
    plt.close()
    fig.savefig('./figures/Precon_{s}.png'.format(s=str(sparsity)), bbox_inches='tight')
    rmse[sparsity]=np.sqrt(np.power(SNYCQ - Precon,2).melt()['value'].mean())

rmse.hvplot.scatter(logx=True, xlabel='Sparsity', ylabel='RMSE', title='', xlim=(1e-4,10)).opts(show_legend=False) * rmse.hvplot(logx=True, xlim=(1e-4,10)).opts(show_legend=False)


