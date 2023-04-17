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
# This notebook performs an initial exploration of the self-reports associated with the resting-state scans. We will look at things like the distribution of responses across scans, the correlation between responses to the different questions and few similar things.
#

import pandas as pd
import numpy as np
import panel as pn
import seaborn as sns
import matplotlib.pyplot as plt
import hvplot.pandas
from IPython import display

from utils.basics import PROC_SNYCQ_DIR, get_sbj_scan_list
from utils.basics import SNYCQ_Questions, SNYCQ_Question_type
from utils.SNYCQ_NMF_Extra import plot_anwers_distribution_per_question, plot_Q_bars

# # 1. Load SNYCQ
#
# We will first load the self-reports for all scans that passed the QA

SBJs, SCANs, SNYCQ = get_sbj_scan_list(when='post_motion', return_snycq=True)

# We now remove the answer to the question about vigilance, as we do not use that question when looking for groups of scans with different experiences.

SNYCQ = SNYCQ.drop('Vigilance',axis=1)
Nscans, Nquestions = SNYCQ.shape
print(SNYCQ.shape)

# This next cell saves the data that will later become the input to the Non-Sparse Constrained Non-Negative Matrix Factorization algorithm.

SNYCQ.to_csv('/data/SFIMJGC_Introspec/2023_fc_introspection/data/shared_with_mlt/snycq.csv')

# Sanity check to make sure there are no missing values for any of the questions

print(SNYCQ.isna().sum())

# ***
#
# # 2. Plot the data in different meaningful ways
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
f = data_to_plot.hvplot.heatmap(width=400, height=800, cmap='viridis', yticks= x_ticks_info, fontscale=1.5, xlabel='Questions', ylabel='Scans', title='M').opts(xrotation=90, colorbar_opts={'title':'Response Value:'})
pn.Row(f).save('./figures/S10_M_matrix.png')

display.Image('./figures/S10_M_matrix.png')

# ## 1.3. Explore the relationship between questions via simple correlation
#
# We compute the correlation across the scan dimension, ensure the resulting dataframe has the correct data type and provide names to the index and columns.

snycq_corr = SNYCQ.corr().infer_objects()
snycq_corr.index.name='Questions'
snycq_corr.columns.name='Questions'

# We rely on seaborn ```clustermap``` function to sort question according to how correlated they are

sns.set(font_scale=1.2)
clmap = sns.clustermap(snycq_corr.round(1), annot=True, cmap='RdBu_r', figsize=(8,8), vmin=-.5, vmax=.5, cbar_kws = {'orientation':'horizontal'}, cbar_pos=(0,0,1,.01), method='ward');
clmap.ax_row_dendrogram.set_visible(False)
clmap.ax_col_dendrogram.set_visible(False)

# We automatically extract the order of the questions as grouped by ```clustermap``` above.

sorted_questions = list(SNYCQ.columns[clmap.dendrogram_col.reordered_ind])
print(sorted_questions)

# ## 1.4. Plot the distribution of answers per question

g = sns.PairGrid(SNYCQ[sorted_questions], diag_sharey=False, height=1)
g.map_upper(sns.scatterplot, s=5)
g.map_lower(sns.kdeplot)
g.map_diag(sns.kdeplot, lw=1)

layout = None
for q in SNYCQ.columns:
    plot = SNYCQ[q].reset_index(drop=True).hvplot.hist(bins=np.linspace(0,100,20), width=250, height=200, normed=True, ylabel='Density', fontsize=12) * SNYCQ[q].reset_index(drop=True).hvplot.kde()
    if layout is None:
        layout = plot
    else:
        layout = layout + plot
layout = layout.cols(3).opts(toolbar=None)
pn.Row(layout).save('./figures/S10_SNYCQ_AnswerDistribution.png')

display.Image('./figures/S10_SNYCQ_AnswerDistribution.png')
