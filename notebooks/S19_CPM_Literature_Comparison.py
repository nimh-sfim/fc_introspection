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
#     display_name: FC Instrospection py 3.10 | 2023b
#     language: python
#     name: fc_introspection_2023b_py310
# ---

# # Description
#
# This notebook creates the figure we use to compare how our results relate to prior CPM work

# +
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
# -

our_acc = pd.read_csv('../resources/cpm/final_avg_accuracies.csv', index_col=0)
our_acc.index.name='Target'
our_acc.columns.name='Accuracy'
our_acc.loc[['Thought Pattern 1','Thought Pattern 2','Wakefulness','Images','Surroundings','Past']]

# ### 1. Load the data

data=pd.read_excel('../resources/cpm_literature_search/CPM_Literature_Search.xlsx', na_filter=True, skiprows=2, sheet_name='FINAL')
data.columns=['Title','Year','# Subjects','Target','Category','Atlas','# ROIs','Pos Nw | R', 'Pos Nw | Rho', 'Neg Nw | R', 'Neg Nw | Rho','Both Nw | R','Both Nw | Rho','Cross Validation','Exclusion','Extra1']
data.drop(['Extra1'],axis=1, inplace=True)

scopus_num_studies = data['Title'].ffill().unique().shape[0]
print ("++ INFO: Number of papers found in scopus: %d" % scopus_num_studies)

# ### 2. Count number of excluded studies per reason
#
# Report the number of studies excluded from further analysis and the reason for exclusion

data['Exclusion'].value_counts()

print('++ INFO: Number of studies passing initial exclusion criteria: %d' % (scopus_num_studies - data['Exclusion'].value_counts().sum()))

# ### 3. Remove excluded studies

# Remove excluded studies
data = data[data['Exclusion'].isnull()]
data = data.drop(['Exclusion'],axis=1)
data = data.reset_index(drop=True)

data = data.ffill()

data.sample(10)

# +
data['Year']       = data['Year'].astype(int)
data['# Subjects'] = [int(str(i).split('-')[0]) for i in data['# Subjects']]

data = data.replace('N/R',np.nan)

for col in ['Pos Nw | R','Pos Nw | Rho','Neg Nw | R','Neg Nw | Rho','Both Nw | R','Both Nw | Rho']:
    data[col] = [i if ('NS' not in str(i)) else np.nan for i in data[col]]
# -

print('++ Number of reported models: %d' % data[['Pos Nw | R','Neg Nw | R','Both Nw | R','Pos Nw | Rho','Neg Nw | Rho','Both Nw | Rho']].melt().dropna().shape[0])

data['Category'].value_counts()

# ### Plot Pearson's R results

data2plot_R = None
for category in ['Personality/Well-being','Clinical','Cognition']:
    aux = data.set_index('Category').loc[category][['Pos Nw | R','Neg Nw | R','Both Nw | R']].melt().dropna()
    aux['Category'] = category
    aux.drop(['variable'],axis=1,inplace=True)
    aux.columns=['Pearson R','Category']
    if data2plot_R is None:
        data2plot_R = aux
    else:
        data2plot_R = pd.concat([data2plot_R,aux])
data2plot_R = data2plot_R.reset_index(drop=True)

data2plot_R['Category'].value_counts()

label_pos_extra = np.array([0,.02,0,-.02,-.045,-.07])
fig, ax = plt.subplots(1,1,figsize=(4,5))
sns.set(font_scale=1)
sns.set_style('white')
sns.boxplot(data=data2plot_R,x='Category',y='Pearson R',saturation=0.3)
sns.swarmplot(data=data2plot_R,x='Category',y='Pearson R', hue='Category',s=3)
for i, TARGET in enumerate(['Wakefulness','Thought Pattern 2','Surroundings','Thought Pattern 1','Images','Past']):
    ax.hlines(our_acc.loc[TARGET,'Pearson R'],-.5,2.5,'k', linestyles='dashed', label=TARGET, lw=1)
    ax.annotate(TARGET, xy=(2.5, our_acc.loc[TARGET,'Pearson R']), xytext=(2.8,our_acc.loc[TARGET,'Pearson R']+label_pos_extra[i]) ,
            arrowprops=dict(facecolor='black',width=2, headwidth=5), annotation_clip=False, fontsize=11, verticalalignment='center')
    #ax.text(2.6,our_acc.loc[TARGET,'Pearson R']*.96,TARGET,arrowprops=dict(facecolor='black', shrink=0.05))
ax.set_ylabel('Model Accuracy (Pearson`s R)')
plt.ylim((0.0,.7))
plt.xlim((-.5,2.5))
