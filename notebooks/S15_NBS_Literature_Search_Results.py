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

import pandas as pd
import numpy as np
import hvplot.pandas
import seaborn as sns
import matplotlib.pyplot as plt

nbs_lit_search_path = '../resources/nbs_literature_search/NBS_Papers_Top50.csv'

nbs_lit = pd.read_csv(nbs_lit_search_path)

nbs_lit = nbs_lit.replace('?',np.nan)
nbs_lit = nbs_lit.replace('NaN',np.nan)

nbs_lit = nbs_lit.infer_objects()

nbs_lit = nbs_lit[['% Sig. Edges', '% Sig. Nodes']].copy()

nbs_lit['% Sig. Edges'] = nbs_lit['% Sig. Edges'].str.rstrip('%').astype('float')
nbs_lit['% Sig. Nodes'] = nbs_lit['% Sig. Nodes'].str.rstrip('%').astype('float')

fig, axs = plt.subplots(1,2,figsize=(8,5))
# Percent Significant Connections in NBS analysis
sns.boxplot(data=nbs_lit,y='% Sig. Edges', color='lightblue', ax=axs[0])
sns.swarmplot(data=nbs_lit,y='% Sig. Edges',c ='b', s=2,ax=axs[0])
axs[0].set_ylabel('Percentage of Significant Edges')
axs[0].axhline(y=100*590/(380*379/2), color='r', linestyle='--', linewidth=1)
axs[0].text(-0.05, 1, 'a', transform=axs[0].transAxes,
      fontsize=14, fontweight='bold', va='top', ha='right')
# Percentage of Significant Nodes in NBS analysis
sns.boxplot(data=nbs_lit,y='% Sig. Nodes', color='lightblue', ax=axs[1])
sns.swarmplot(data=nbs_lit,y='% Sig. Nodes',c ='b', s=2, ax=axs[1])
axs[1].set_ylabel('Percentage of Significant Nodes')
axs[1].axhline(y=100*221/380, color='r', linestyle='--', linewidth=1)
fig.suptitle('Comparison to clinical studies using NBS', fontsize=14, fontweight='bold')
axs[1].text(-0.05, 1, 'b', transform=axs[1].transAxes,
      fontsize=14, fontweight='bold', va='top', ha='right')

fig, axs = plt.subplots(1,2,figsize=(8,5))
# Percent Significant Connections in NBS analysis
sns.boxplot(data=nbs_lit,y='% Sig. Edges', color='lightgray', ax=axs[0])
sns.swarmplot(data=nbs_lit,y='% Sig. Edges',c ='k', s=2,ax=axs[0])
axs[0].set_ylabel('Percentage of Significant Edges')
axs[0].axhline(y=100*590/(380*379/2), color='r', linestyle='--', linewidth=1)
# Percentage of Significant Nodes in NBS analysis
sns.boxplot(data=nbs_lit,y='% Sig. Nodes', color='lightgray', ax=axs[1])
sns.swarmplot(data=nbs_lit,y='% Sig. Nodes',c ='k', s=2, ax=axs[1])
axs[1].set_ylabel('Percentage of Nodes in Significant Edges')
axs[1].axhline(y=100*221/380, color='r', linestyle='--', linewidth=1)



