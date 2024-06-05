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
#     display_name: FC Introspection (Jan 2023)
#     language: python
#     name: fc_introspection
# ---

# # Description
#
# In this notebook we compare the accuracy achieved in this work when attempting to predict aspects of in-scanner experience with that reported in other applications of CPM to the prediction of behavior, cognitive traits and clinical metrics

import pandas as pd
import hvplot.pandas
import seaborn as sns
import matplotlib.pyplot as plt
import holoviews as hv
import panel as pn

df = pd.read_csv('../resources/cpm_literature_search/CPM_Lit_Search.csv')
df.columns

df.drop(['Link to Article','Effect Size','# Sig Edges Found'],axis=1)

df.hvplot.box(y='R-value', by='Category', width=1000, height=400, fontscale=2, ylabel='Accuracy') * hv.HLine(0.29)

df[['R-value','Category']].hvplot.box(by='Category')

plt.subplots(1,1,figsize=(4,5))
sns.set(font_scale=1)
sns.set_style('white')
sns.boxplot(data=df,y='R-value', saturation=0.5, color='gray')
sns.swarmplot(data=df,y='R-value', hue='Category', alpha=1)
plt.ylim((0.1,.6))
plt.xlim((-.5,2.5))

plt.subplots(1,1,figsize=(4,5))
sns.set(font_scale=1)
sns.set_style('white')
sns.boxplot(data=df,x='Category',y='R-value', saturation=0.5)
sns.swarmplot(data=df,x='Category',y='R-value', hue='Category', alpha=1)
plt.ylim((0.1,.6))
plt.xlim((-.5,2.5))

# +
plt.subplots(1,1,figsize=(4,5))
sns.set(font_scale=1)
sns.set_style('white')
sns.boxplot(data=df,x='Category',y='R-value', saturation=0.5)
sns.swarmplot(data=df,x='Category',y='R-value', hue='Category', alpha=1)
plt.hlines(0.28,-.5,2.5,'k', linestyles='dashed', label='Wakefulness', lw=1)
plt.hlines(0.15,-.5,2.5,'k', linestyles='dashed', label='Factor 1', lw=1)
plt.hlines(0.21,-.5,2.5,'k', linestyles='dashed', label='Factor 2', lw=1)
plt.hlines(0.13,-.5,2.5,'k', linestyles='dashed', label='Images', lw=1)
plt.hlines(0.17,-.5,2.5,'k', linestyles='dashed', label='Surroundings', lw=1)
plt.hlines(0.14,-.5,2.5,'k', linestyles='dashed', label='Past', lw=1)

plt.ylim((0.1,.6))
plt.xlim((-.5,2.5))

# +
plt.subplots(1,1,figsize=(4,5))
sns.set(font_scale=1)
sns.set_style('white')
sns.boxplot(data=df,y='R-value', saturation=0.5, color='gray')
sns.swarmplot(data=df,y='R-value', hue='Category', alpha=1)
plt.ylim((0.1,.6))
plt.xlim((-.5,2.5))
plt.hlines(0.28,-.5,2.5,'k', linestyles='dashed', label='Wakefulness', lw=1)
plt.hlines(0.15,-.5,2.5,'k', linestyles='dashed', label='Factor 1', lw=1)
plt.hlines(0.21,-.5,2.5,'k', linestyles='dashed', label='Factor 2', lw=1)
plt.hlines(0.13,-.5,2.5,'k', linestyles='dashed', label='Images', lw=1)
plt.hlines(0.17,-.5,2.5,'k', linestyles='dashed', label='Surroundings', lw=1)
plt.hlines(0.14,-.5,2.5,'k', linestyles='dashed', label='Past', lw=1)

plt.ylim((0.1,.6))
plt.xlim((-.5,2.5))
# -

aux = pd.DataFrame({'R': df['R-value'].astype(float),'Category': df['Category']}).dropna()
aux

aux.info()

clinical = aux[aux['Category']=='Clinical']
clinical

personality = aux[aux['Category']=='Personality']
personality

# +
personality['R-value'].hvplot.kde(color='green', label='Personality') * clinical['R-value'].hvplot.kde(color='blue', label='Clinical')






# -

aux.hvplot.box(y='R', by='Category')

y = aux.drop(labels=20)

aux


