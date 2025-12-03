# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: FC Introspection py 3.10 | 2023b | Cython
#     language: python
#     name: fc_introspection_2023b_py310_cython
# ---

# # Description
#
# This notebook will take in-scanner data for all scans deemed usable (469 at this point) and apply the ICQF algorithm to the data
#
# In the original submission, we relied on in-house implementation of ICQF. Since then, the machine learning group has released the algorithm publicly and has enhanced it with new functionality.
#
# We use that publicly available version now, which is available at: https://github.com/jefferykclam/ICQF

# Import ICQF version installed from github
import sys
sys.path.append('../../ICQF/')
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from src.data_class import matrix_class
from src.ICQF import ICQF

# +
import pandas as pd
import numpy as np
import hvplot.pandas
import pickle

import panel as pn
import seaborn as sns
import holoviews as hv
from holoviews import opts
import os.path as osp
from utils.basics import RESOURCES_SNYCQ_DIR, ORIG_DEMO_PATH
# -

# # 1. Load Experiential Data
#
# We load the data following the detection of outliers, meaning we will be working only with 469 scans

# Load Clustering Results
emb_plus  = pd.read_csv(osp.join(RESOURCES_SNYCQ_DIR, 'SNYCQ_tsne_embeddings_plus.csv'), index_col=[0,1])
emb_plus.drop(['TSNE1','TSNE2','TSNE3'],axis=1,inplace=True)
emb_plus.head(3)

# Next, we plot the data entering the ICQF analyses, this will become panel a in Supplementary Figure 3.

# +
SNYCQ_to_plot = emb_plus.drop(['Set Label','Group Probability'],axis=1)
SNYCQ_to_plot = SNYCQ_to_plot[['Future','Specific','Past','Positive','People','Images','Words','Negative','Surroundings','Myself','Intrusive']]
Nscans = SNYCQ_to_plot.shape[0]

data_to_plot              = pd.DataFrame(SNYCQ_to_plot.values)
data_to_plot.index        = np.arange(SNYCQ_to_plot.shape[0])
data_to_plot.index.name   = 'Questions'
data_to_plot.columns      = SNYCQ_to_plot.columns
data_to_plot.columns.name = 'Scans'
f_data = data_to_plot.hvplot.heatmap(width=375, height=900, cmap='viridis', fontscale=1.5, xlabel='Questions', title='', shared_axes=False).opts(xrotation=90, colorbar_opts={'title':'Response:'}, toolbar=None)
f_data
# -

# # 2. Load and Encode Basic Demographics

# Extract final list of scans and subjects
scan_list = emb_plus.index.to_list()
sbj_list  = list(emb_plus.index.get_level_values(level='Subject').unique()) 

# Extract SNYCQ items
SNYCQ = emb_plus[['People','Positive','Negative','Past','Future','Myself','Intrusive','Surroundings','Words','Images','Specific']]
SNYCQ_items = SNYCQ.columns

# Load demographic data
demographics = pd.read_csv(ORIG_DEMO_PATH, index_col=0,sep='\t')
demographics = demographics.loc[sbj_list]
demographics.index.name='Subject'
demographics.head(3)

# ## 2.1. Encode age and gender
#
# The age and gender are available in ```string``` form. They need to be converted to numerical for ICQF. The following cells do that and create the ```C``` or ```confounds``` matrix for ICQF

# Convert age classes into numerical labels
ageclass        = np.unique(demographics.loc[:,'age (5-year bins)'].values)
normal_ageclass = 0.5*(np.linspace(0,1,8)[1:] + np.linspace(0,1,8)[:-1])
age_translation_dict = {k:v for (k,v) in zip(ageclass,normal_ageclass)}
print(age_translation_dict)

# Convert Gender and Age into numerical labels
gender_list = []
age_list = []
for sbj,run in scan_list:
    gender = demographics.loc[sbj,'gender']
    age   = demographics.loc[sbj,'age (5-year bins)']
    # Gender string to 1,0 label
    if gender == 'M':
        gender_list.append(1.0)
    else:
        gender_list.append(0.0)
    # Age range to value between 0 and 1
    normal_age = age_translation_dict[age] #float(normal_ageclass[ np.where(ageclass == age)[0] ])
    age_list.append(normal_age)

# Now, that everything is available we create the ```C``` matrix, which contains an intercept (all ones), age and its decreasing counterpart, gender and its oposite counterpart.

confounds               = pd.DataFrame(index=SNYCQ.index,columns=['age','Gender (M)'])
confounds['age']        = age_list
confounds['Gender (M)']     = gender_list
confounds['1 - age']    = 1 -confounds['age']
confounds['Gender (F)'] = 1 -confounds['Gender (M)']
confounds['intercept']  = 1.0
confounds.head(5)

# # 3. Run ICQF
#
# ## 3.1. Prepare additional inputs (e.g., inputation mask and matrix_class data structure)

# No data inputation needed --> the mask contains all 1s
nan_mask = np.ones_like(SNYCQ.values)

# Create matrix_class with data, confounds and imputation mask
MF_data = matrix_class(M=SNYCQ.values, 
                       C=confounds.values,
                       nan_mask=nan_mask,
                       dataname='SNYCQ', itemlist=SNYCQ_items)

# Create ICQF object
clf = ICQF(None, regularizer=1, 
           W_upperbd=(True, 1.0),
           M_upperbd=(True, 100.0),
           Q_upperbd=(True, 100.0),
           verbose=True)


# ## 3.2. Estimate optimal hyper-parameters
#
# The ICQF package can estiamte, in a data-driven manner, the optimal values for number of dimensions (d), regularization of the W matrix (W_beta) and regularization of the Q matrix (Q_beta). The next cell will do that by exploring the following hyper-parameter space:
#
# |Hyper-parameter| Description| Range of Exploration|
# |:-------|:--------|:--------|
# | d | Number of Dimensions | 2,3,4,5 |
# | W_beta | Sparsity for W | 0.0, 0.01, 0.1, 0.2, 0.5, 1 |
# | Q_beta | Sparsity for Q | 0.0, 0.01, 0.1, 0.2, 0.5, 1 |

optimal_MF_data, optimal_stat, embed_stat_list = clf.detect_dimension(MF_data,
                    dimension_list=[2,3,4,5],
                    W_beta_list=[0.0,0.01,0.1,0.2,0.5,1],
                    Q_beta_list=[0.0,0.01,0.1,0.2,0.5,1],
                    repeat=5, nfold=10,
                    random_fold=False,
                    separate_beta=True,
                    detection='kneed')

# Print the selected hyper-parameter values:

# +

print('++ INFO [ICQF hyper-parameter optimization]: d=%d' % clf.n_components)
print('++ INFO:[ICQF hyper-parameter optimization]: W_beta=%0.2f' % clf.W_beta)
print('++ INFO:[ICQF hyper-parameter optimization]: W_beta=%0.2f' % clf.Q_beta)
# -

# Save the results of the optimization to disk

# +

data_to_save = {'optimal_MF_data': optimal_MF_data, 'optimal_stat':optimal_stat, 'embed_stat_list': embed_stat_list,'clf':clf}
with open('./results/icqf_results.pkl', 'wb') as file:
    pickle.dump(data_to_save, file)
# -

# ## 3.3. Extract results obtainecd with optimal hyper-parameters

# +
DIM = clf.n_components
Q = pd.DataFrame(optimal_MF_data.Q, index=SNYCQ_items, columns=['Factor {d}'.format(d=d+1) for d in range(DIM)])
W = pd.DataFrame(optimal_MF_data.W, index=SNYCQ.index, columns=['Factor {d}'.format(d=d+1) for d in range(DIM)])
C = pd.DataFrame(optimal_MF_data.C, index=SNYCQ.index, columns=['Age (elder)','Gender (M)','Age (younger)','Gender (F)','Intercept'])
C = C[['Intercept','Age (younger)','Age (elder)','Gender (M)','Gender (F)']] # Sorting to make it more interpretable

Qc = pd.DataFrame(optimal_MF_data.Qc, index=SNYCQ.columns, columns = ['Age (elder)','Gender (M)','Age (younger)','Gender (F)','Intercept'])
Qc = Qc[['Intercept','Age (younger)','Age (elder)','Gender (M)','Gender (F)']] # Sorting to make it more interpretable
# -

# ***

W_Supp_Fig = pd.concat([W, pd.DataFrame(index=W.index,columns=C.columns)],axis=1)
W_Supp_Fig.reset_index(drop=True).hvplot.heatmap(cmap='Greens', width=300, height=550, fontscale=1.2, clim=(0,1), shared_axes=False).opts( colorbar_opts={'title':'W Matrix'}, xrotation=90, toolbar=None)

C_Supp_Fig = pd.concat([pd.DataFrame(index=C.index,columns=W.columns), C],axis=1)
C_Supp_Fig.reset_index(drop=True).hvplot.heatmap(cmap='Purples', width=300, height=550, fontscale=1.2, clim=(0,1), shared_axes=False).opts( colorbar_opts={'title':'C Matrix'}, xrotation=90, toolbar=None)

Q_Supp_Fig = pd.concat([Q, pd.DataFrame(index=Q.index,columns=C.columns)],axis=1)
Q_Supp_Fig.hvplot.heatmap(cmap='Oranges', width=300, height=550, fontscale=1.2, clim=(0,100), shared_axes=False).opts( colorbar_opts={'title':'Q Matrix'}, xrotation=90, toolbar=None)

Qc_Supp_Fig = pd.concat([pd.DataFrame(index=Qc.index,columns=Q.columns),Qc],axis=1)
Qc_Supp_Fig.hvplot.heatmap(cmap='Oranges', width=300, height=550, fontscale=1.2, clim=(0,100), shared_axes=False).opts( colorbar_opts={'title':'Q Matrix'}, xrotation=90, toolbar=None)

# ***
#
# # 4. Plot scans in ICQF space 
#
# Next, we will generate a scatter plot where each scan is represented by a point in the 2D ICQF space. Scans will be colored according to Set Membership previously stablished by performing Gaussian Misture Modeling of the data in the original 11D space.

W_wSetLabels = pd.concat([W,emb_plus['Set Label']],axis=1)

W_wSetLabels.hvplot.scatter(
    x='Factor 1',
    y='Factor 2',
    aspect='square',
    color='Set Label',
    cmap=["#ff7f0e", "#ffffff", "#1f77b4"],
    legend='bottom_left',
    line_color='black',
    line_width=0.5,
    alpha=0.7
).opts(
        fontsize={
            'ticks': 12,
            'labels': 14,
            'legend': 14,
            'title': 16
        }
    ,toolbar=None)

# ## 4.1. Heatmap with relationship between Factor 1 (TP1) and original sNYCQ items 

F1_vals_to_plot = pd.DataFrame(Q.sort_values(by='Factor 1', ascending=True)['Factor 1'].round(0).astype(int)).T
f = F1_vals_to_plot.hvplot.heatmap(cmap='Bone', clim=(0,110), width=700, height=175,colorbar=False, yaxis=None).opts(toolbar=None, line_color='k', line_width=1)
f * hv.Labels(f).opts(opts.Labels(text_color='white', xrotation=45, fontsize={'labels':12,'xticks':14}))

# ## 4.2. Heatmap with relationship between Factor 2 (TP2) and original sNYCQ items

F2_vals_to_plot = pd.DataFrame(Q.sort_values(by='Factor 2', ascending=True)['Factor 2'].round(0).astype(int))
f = F2_vals_to_plot.hvplot.heatmap(cmap='Bone', clim=(0,110), width=200, height=700,colorbar=False, xaxis=None).opts(toolbar=None, line_color='k', line_width=1)
f * hv.Labels(f).opts(opts.Labels(text_color='white', xrotation=45, fontsize={'labels':12,'yticks':14}))

# ## 4.3 Plot examples of scans sitting on both corners of the embedding

sorted_q = Q.sort_values(by=['Factor 1','Factor 2'],ascending=False).index

top_left_scans = W[(W['Factor 1']<0.35) & (W['Factor 2']>0.9)].index
a = SNYCQ.loc[top_left_scans].reset_index(drop=True)
a[sorted_q].hvplot.heatmap(width=250, height=250, clim=(0,100), cmap='Viridis', ylabel='Scan', xlabel='Question', fontscale=1.2).opts(colorbar=False, xrotation=90, toolbar=None)

bot_right_scans = W[(W['Factor 1']>0.9) & (W['Factor 2']<0.1)].index
a = SNYCQ.loc[bot_right_scans].reset_index(drop=True)
a[sorted_q].hvplot.heatmap(width=250, height=250, clim=(0,100), cmap='Viridis', ylabel='Scan', xlabel='Question', fontscale=1.2).opts(colorbar=False, xrotation=90, toolbar=None)

# Save ICQF embedding to disk

W.to_csv('../resources/icqf/W.csv')
