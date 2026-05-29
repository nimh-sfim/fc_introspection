#!/usr/bin/env python
# coding: utf-8

# # Description
# 
# This notebook creates the figure we use to compare how our results relate to prior CPM work

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# In[2]:


our_acc = pd.read_csv('../resources/cpm/subject_aware_final_avg_accuracies.csv', index_col=0)
our_acc.index.name='Target'
our_acc.columns.name='Accuracy'
our_acc.loc[['Thought Pattern 1','Thought Pattern 2','Wakefulness','Images','Surroundings','Past']]


# ### 1. Load the data

# In[3]:


data=pd.read_excel('../resources/cpm_literature_search/CPM_Literature_Search.xlsx', na_filter=True, skiprows=2, sheet_name='FINAL')
data.columns=['Title','Year','# Subjects','Target','Category','Atlas','# ROIs','Pos Nw | R', 'Pos Nw | Rho', 'Neg Nw | R', 'Neg Nw | Rho','Both Nw | R','Both Nw | Rho','Cross Validation','Exclusion','Extra1']
data.drop(['Extra1'],axis=1, inplace=True)


# In[4]:


scopus_num_studies = data['Title'].ffill().unique().shape[0]
print ("++ INFO: Number of papers found in scopus: %d" % scopus_num_studies)


# ### 2. Count number of excluded studies per reason
# 
# Report the number of studies excluded from further analysis and the reason for exclusion

# In[5]:


data['Exclusion'].value_counts()


# In[6]:


print('++ INFO: Number of studies passing initial exclusion criteria: %d' % (scopus_num_studies - data['Exclusion'].value_counts().sum()))


# ### 3. Remove excluded studies

# In[7]:


# Remove excluded studies
data = data[data['Exclusion'].isnull()]
data = data.drop(['Exclusion'],axis=1)
data = data.reset_index(drop=True)


# In[8]:


data = data.ffill()


# In[9]:


data.sample(10)


# In[10]:


data['Year']       = data['Year'].astype(int)
data['# Subjects'] = [int(str(i).split('-')[0]) for i in data['# Subjects']]

data = data.replace('N/R',np.nan)

for col in ['Pos Nw | R','Pos Nw | Rho','Neg Nw | R','Neg Nw | Rho','Both Nw | R','Both Nw | Rho']:
    data[col] = [i if ('NS' not in str(i)) else np.nan for i in data[col]]


# In[11]:


print('++ Number of reported models: %d' % data[['Pos Nw | R','Neg Nw | R','Both Nw | R','Pos Nw | Rho','Neg Nw | Rho','Both Nw | Rho']].melt().dropna().shape[0])


# In[12]:


data['Category'].value_counts()


# ### Plot Pearson's R results

# In[13]:


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


# In[14]:


data2plot_R['Category'].value_counts()


# In[25]:


plt.rcParams['font.family'] = 'Arial'
label_pos_extra = np.array([0,.02,0,-.02,-.045,-.07])
fig, ax = plt.subplots(1,1,figsize=(4.4,5))
sns.set(font_scale=1)
sns.set_style('white')
sns.boxplot(data=data2plot_R,x='Category',y='Pearson R',saturation=0.3,hue='Category')
sns.swarmplot(data=data2plot_R,x='Category',y='Pearson R', hue='Category',s=3)
for i, TARGET in enumerate(['Wakefulness','Thought Pattern 2','Surroundings','Thought Pattern 1','Images','Past']):
    ax.hlines(our_acc.loc[TARGET,'Pearson R'],-.5,2.5,'k', linestyles='dashed', label=TARGET, lw=1)
    ax.annotate(TARGET, xy=(2.5, our_acc.loc[TARGET,'Pearson R']), xytext=(2.8,our_acc.loc[TARGET,'Pearson R']+label_pos_extra[i]) ,
            arrowprops=dict(facecolor='black',width=2, headwidth=5), annotation_clip=False, fontsize=11, verticalalignment='center')
ax.set_ylabel('Model Accuracy (Pearson`s R)')
plt.ylim((0.0,.7))
plt.xlim((-.5,2.5))


# ### Saving image and source data files

# In[26]:


import matplotlib as mpl
mpl.rcParams['svg.fonttype'] = 'none'
fig.savefig('./figures/Figure05_D.svg', format='svg', bbox_inches='tight')


# In[30]:


data2plot_R.to_csv('./source_data_files/figure_05_d_literature.csv', index=0)


# In[31]:


our_acc.loc[['Wakefulness','Thought Pattern 2','Surroundings','Thought Pattern 1','Images','Past'],'Pearson R'].to_csv('./source_data_files/figure_05_d_our_acc.csv', float_format='%.3f')


# In[16]:


fig.savefig('./figures/Figure05_D.png')

