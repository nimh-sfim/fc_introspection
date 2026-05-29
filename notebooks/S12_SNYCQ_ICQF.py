#!/usr/bin/env python
# coding: utf-8

# # Description : ICQF on SNYCQ Data
# 
# This notebook will take in-scanner data for all scans deemed usable (469 at this point) and apply the ICQF algorithm to the data
# 
# In the original submission, we relied on in-house implementation of ICQF. Since then, the [NIMH Machine Learning Core](https://cmn.nimh.nih.gov/mlt) has released the ICQF algorithm publicly and has enhanced it with new functionality.
# 
# We now use that publicly available version, which you can find at: https://github.com/jefferykclam/ICQF
# 
# > **NOTE:** ICQF requires cython, and the first time you try to import it, it will attempt some compilation. For that you need `gcc`
# 
# > **NOTE:** If you run into issues during compilation, do ```rm -rf ~/home/javiergc~/.pyxbld``` from a terminal, and try again.

# In[1]:


import sys
from pathlib import Path

icqf_root = Path("/data/SFIMJGC_Introspec/2023_fc_introspection/code/ICQF").resolve()
sys.path.insert(0, str(icqf_root))

from src.data_class import matrix_class
from src.ICQF import ICQF


# In[2]:


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# In[3]:


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


# # 1. Load Experiential Data
# 
# We load the data following the detection of outliers, meaning we will be working only with 469 scans

# In[4]:


# Load Clustering Results
emb_plus  = pd.read_csv(osp.join(RESOURCES_SNYCQ_DIR, 'SNYCQ_tsne_embeddings_plus.csv'), index_col=[0,1])
emb_plus.drop(['TSNE1','TSNE2','TSNE3'],axis=1,inplace=True)
print(emb_plus.shape)
emb_plus.head(3)


# Next, we plot the data entering the ICQF analyses, this will become panel a in Supplementary Figure 3.

# In[5]:


SNYCQ_to_plot = emb_plus.drop(['Set Label','Group Probability'],axis=1)
SNYCQ_to_plot = SNYCQ_to_plot[['Future','Specific','Past','Positive','People','Images','Words','Negative','Surroundings','Myself','Intrusive']]
Nscans = SNYCQ_to_plot.shape[0]

data_to_plot              = pd.DataFrame(SNYCQ_to_plot.values)
data_to_plot.index        = np.arange(SNYCQ_to_plot.shape[0])
data_to_plot.index.name   = 'Questions'
data_to_plot.columns      = SNYCQ_to_plot.columns
data_to_plot.columns.name = 'Scans'
f_data = data_to_plot.hvplot.heatmap(width=325, height=900, cmap='viridis', fontscale=1.25, xlabel='Questions', title='', shared_axes=False).opts(xrotation=90, colorbar_opts={'title':'Response:'}, toolbar=None)
f_data


# ![SNYCQ Heatmap](./figures/Figure01_A-SNYCQ_Heatmap.png)
# 
# ### Saving Publication Ready Panel and Source Data

# In[6]:


import holoviews as hv

from bokeh.io import save
from bokeh.models.plots import Plot
from bokeh.resources import INLINE

hv.extension("bokeh")

def svg_backend(hv_plot, element):
    hv_plot.state.output_backend = "svg"

svg_plot = f_data.opts(hooks=[svg_backend])

bokeh_obj = hv.render(svg_plot, backend="bokeh")

# Extra safety for layouts / overlays
if isinstance(bokeh_obj, Plot):
    bokeh_obj.output_backend = "svg"

for p in bokeh_obj.select({"type": Plot}):
    p.output_backend = "svg"

save(
    bokeh_obj,
    filename="./figures/Figure01_A-SNYCQ_Heatmap.html",
    resources=INLINE,
    title="Figure01_A",
)


# In[7]:


from utils.basics import get_sbj_scan_list
_, _, SNYCQ_wVigilance = get_sbj_scan_list(when='post_motion', return_snycq=True)
SNYCQ_to_plot_b=SNYCQ_wVigilance.loc[emb_plus.index,'Vigilance']
SNYCQ_to_plot_b.name='Wakefulness'
data_to_plot              = pd.DataFrame(SNYCQ_to_plot_b.values)
data_to_plot.index        = np.arange(SNYCQ_to_plot_b.shape[0])
data_to_plot.index.name   = 'Questions'
data_to_plot.columns      = ['Wakefulness']
data_to_plot.columns.name = 'Scans'
f_data = data_to_plot.hvplot.heatmap(width=70, height=900, cmap='viridis', fontscale=1.25, title='', shared_axes=False).opts(xrotation=90, toolbar=None, colorbar=False)
f_data


# In[8]:


def svg_backend(hv_plot, element):
    hv_plot.state.output_backend = "svg"

svg_plot = f_data.opts(hooks=[svg_backend])

bokeh_obj = hv.render(svg_plot, backend="bokeh")

# Extra safety for layouts / overlays
if isinstance(bokeh_obj, Plot):
    bokeh_obj.output_backend = "svg"

for p in bokeh_obj.select({"type": Plot}):
    p.output_backend = "svg"

save(
    bokeh_obj,
    filename="./figures/Figure01_A-SNYCQ_Heatmap_Top.html",
    resources=INLINE,
    title="Figure01_A_Top",
)


# In[9]:


pd.concat([SNYCQ_to_plot_b,SNYCQ_to_plot],axis=1).to_csv('./source_data_files/figure_01_a.csv', float_format='%.1f', index=None)


# # 2. Load and Encode Basic Demographics

# In[10]:


# Extract final list of scans and subjects
scan_list = emb_plus.index.to_list()
sbj_list  = list(emb_plus.index.get_level_values(level='Subject').unique()) 


# In[11]:


# Extract SNYCQ items
SNYCQ = emb_plus[['People','Positive','Negative','Past','Future','Myself','Intrusive','Surroundings','Words','Images','Specific']]
SNYCQ_items = SNYCQ.columns


# In[12]:


# Load demographic data
demographics = pd.read_csv(ORIG_DEMO_PATH, index_col=0,sep='\t')
demographics = demographics.loc[sbj_list]
demographics.index.name='Subject'
demographics.head(3)


# ## 2.1. Encode age and gender
# 
# The age and gender are available in ```string``` form. They need to be converted to numerical for ICQF. The following cells do that and create the ```C``` or ```confounds``` matrix for ICQF

# In[13]:


# Convert age classes into numerical labels
ageclass        = np.unique(demographics.loc[:,'age (5-year bins)'].values)
normal_ageclass = 0.5*(np.linspace(0,1,8)[1:] + np.linspace(0,1,8)[:-1])
age_translation_dict = {k:v for (k,v) in zip(ageclass,normal_ageclass)}
print(age_translation_dict)


# In[14]:


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

# In[15]:


confounds               = pd.DataFrame(index=SNYCQ.index,columns=['age','Gender (M)'])
confounds['age']        = age_list
confounds['Gender (M)']     = gender_list
confounds['1 - age']    = 1 -confounds['age']
confounds['Gender (F)'] = 1 -confounds['Gender (M)']
confounds['intercept']  = 1.0
confounds.head(5)


# ***
# # 3. Run ICQF
# 
# ## 3.1. Prepare additional inputs (e.g., inputation mask and matrix_class data structure)

# In[13]:


# No data inputation needed --> the mask contains all 1s
nan_mask = np.ones_like(SNYCQ.values)


# In[14]:


# Create matrix_class with data, confounds and imputation mask
MF_data = matrix_class(M=SNYCQ.values, 
                       C=confounds.values,
                       nan_mask=nan_mask,
                       dataname='SNYCQ', itemlist=SNYCQ_items)


# In[15]:


# Create ICQF object
clf = ICQF(None, regularizer=1, 
           W_upperbd=(True, 1.0),
           M_upperbd=(True, 100.0),
           Q_upperbd=(True, 100.0),
           verbose=True,
           random_state=24)


# ## 3.2. Estimate optimal hyper-parameters
# 
# The ICQF package can estiamte, in a data-driven manner, the optimal values for number of dimensions (d), regularization of the W matrix (W_beta) and regularization of the Q matrix (Q_beta). The next cell will do that by exploring the following hyper-parameter space:
# 
# |Hyper-parameter| Description| Range of Exploration|
# |:-------|:--------|:--------|
# | d | Number of Dimensions | 2,3,4,5 |
# | W_beta | Sparsity for W | 0.0, 0.01, 0.1, 0.2, 0.5, 1 |
# | Q_beta | Sparsity for Q | 0.0, 0.01, 0.1, 0.2, 0.5, 1 |

# In[16]:


optimal_MF_data, optimal_stat, embed_stat_list = clf.detect_dimension(MF_data,
                    dimension_list=[2,3,4,5],
                    W_beta_list=[0.0,0.01,0.1,0.2,0.5,1],
                    Q_beta_list=[0.0,0.01,0.1,0.2,0.5,1],
                    repeat=5, nfold=10,
                    random_fold=False,
                    separate_beta=True,
                    detection='kneed')


# Print the selected hyper-parameter values:

# In[17]:


print('++ INFO [ICQF hyper-parameter optimization]: d=%d' % clf.n_components)
print('++ INFO:[ICQF hyper-parameter optimization]: W_beta=%0.2f' % clf.W_beta)
print('++ INFO:[ICQF hyper-parameter optimization]: Q_beta=%0.2f' % clf.Q_beta)


# Save the results of the optimization to disk

# In[18]:


data_to_save = {'optimal_MF_data': optimal_MF_data, 'optimal_stat':optimal_stat, 'embed_stat_list': embed_stat_list,'clf':clf}
with open('../resources/snycq/SNYCQ_icqf_results.pkl', 'wb') as file:
    pickle.dump(data_to_save, file)


# In[16]:


import pickle

with open('../resources/snycq/SNYCQ_icqf_results.pkl', 'rb') as file:
    loaded_data = pickle.load(file)

optimal_MF_data = loaded_data['optimal_MF_data']
optimal_stat = loaded_data['optimal_stat']
embed_stat_list = loaded_data['embed_stat_list']
clf = loaded_data['clf']


# ## 3.3. Extract results obtainecd with optimal hyper-parameters

# In[17]:


DIM = clf.n_components
Q = pd.DataFrame(optimal_MF_data.Q, index=SNYCQ_items, columns=['Factor {d}'.format(d=d+1) for d in range(DIM)])
W = pd.DataFrame(optimal_MF_data.W, index=SNYCQ.index, columns=['Factor {d}'.format(d=d+1) for d in range(DIM)])
C = pd.DataFrame(optimal_MF_data.C, index=SNYCQ.index, columns=['Age (elder)','Gender (M)','Age (younger)','Gender (F)','Intercept'])
C = C[['Intercept','Age (younger)','Age (elder)','Gender (M)','Gender (F)']] # Sorting to make it more interpretable

Qc = pd.DataFrame(optimal_MF_data.Qc, index=SNYCQ.columns, columns = ['Age (elder)','Gender (M)','Age (younger)','Gender (F)','Intercept'])
Qc = Qc[['Intercept','Age (younger)','Age (elder)','Gender (M)','Gender (F)']] # Sorting to make it more interpretable


# ***
# Plot the W matrix

# In[18]:


W_Supp_Fig = pd.concat([W, pd.DataFrame(index=W.index,columns=C.columns)],axis=1)
W_plot = W_Supp_Fig.reset_index(drop=True).hvplot.heatmap(cmap='Greens', width=300, height=550, fontscale=1.2, clim=(0,1), shared_axes=False).opts( colorbar_opts={'title':'W Matrix'}, xrotation=90, toolbar=None)
hv.save(W_plot, osp.join('figures', 'Supplementary_Figure05W.html'), backend='bokeh')
W_plot


# ![W](./figures/Supplementary_Figure05_W.png)

# Plot the C matrix

# In[19]:


C_Supp_Fig = pd.concat([pd.DataFrame(index=C.index,columns=W.columns), C],axis=1)
C_plot = C_Supp_Fig.reset_index(drop=True).hvplot.heatmap(cmap='Purples', width=300, height=550, fontscale=1.2, clim=(0,1), shared_axes=False).opts( colorbar_opts={'title':'C Matrix'}, xrotation=90, toolbar=None)
hv.save(C_plot, osp.join('figures', 'Supplementary_Figure05_C.html'), backend='bokeh')
C_plot


# ![C](./figures/Supplementary_Figure05_C.png)

# Plot the Q matrix

# In[20]:


Q_Supp_Fig = pd.concat([Q, pd.DataFrame(index=Q.index,columns=C.columns)],axis=1)
Q_plot = Q_Supp_Fig.hvplot.heatmap(cmap='Oranges', width=300, height=550, fontscale=1.2, clim=(0,100), shared_axes=False).opts( colorbar_opts={'title':'Q Matrix'}, xrotation=90, toolbar=None)
hv.save(Q_plot, osp.join('figures', 'Supplementary_Figure05_Q.html'), backend='bokeh')
Q_plot


# ![Q](./figures/Supplementary_Figure05_Q.png)

# Plot Qc

# In[52]:


Qc_Supp_Fig = pd.concat([pd.DataFrame(index=Qc.index,columns=Q.columns),Qc],axis=1)
Qc_plot = Qc_Supp_Fig.hvplot.heatmap(cmap='Oranges', width=300, height=550, fontscale=1.2, clim=(0,100), shared_axes=False).opts( colorbar_opts={'title':'Q Matrix'}, xrotation=90, toolbar=None)
hv.save(Qc_plot, osp.join('figures', 'Supplementary_Figure05_Qc.html'), backend='bokeh')
Qc_plot


# ![Qc](./figures/Supplementary_Figure05_Qc.png)

# ***
# 
# # 4. Plot scans in ICQF space 
# 
# Next, we will generate a scatter plot where each scan is represented by a point in the 2D ICQF space. Scans will be colored according to Set Membership previously stablished by performing Gaussian Misture Modeling of the data in the original 11D space.

# In[21]:


W_wSetLabels = pd.concat([W,emb_plus['Set Label']],axis=1)


# In[22]:


import holoviews as hv

hv.extension("bokeh")

W_scat_hv = hv.Scatter(
    W_wSetLabels,
    kdims=["Factor 1"],
    vdims=["Factor 2", "Set Label"],
).opts(
    aspect="square",
    color="Set Label",
    cmap=["#ff7f0e", "#ffffff", "#1f77b4"],
    legend_position="bottom_left",
    line_color="black",
    line_width=0.5,
    alpha=0.7,
    size=6,          # adjust to match hvplot's default marker size
    toolbar=None,
    fontsize={
        "ticks": 12,
        "labels": 14,
        "legend": 14,
        "title": 16,
    },
)
W_scat_hv


# ![W Scatter Plot](./figures/Figure01_C-SNYCQ_ICQF_Scatter.png)
# 
# ### Save Publication Ready Panel and Data Souce File

# In[23]:


svg_plot = W_scat_hv.opts(hooks=[svg_backend])

bokeh_obj = hv.render(svg_plot, backend="bokeh")

if isinstance(bokeh_obj, Plot):
    bokeh_obj.output_backend = "svg"

for p in bokeh_obj.select({"type": Plot}):
    p.output_backend = "svg"

save(
    bokeh_obj,
    filename="./figures/Figure01_C-SNYCQ_ICQF_Scatter_main.html",
    resources=INLINE,
    title="Figure01_C-SNYCQ_ICQF_Scatter_main",
)


# In[24]:


W_wSetLabels.to_csv('./source_data_files/figure_01_c_main.csv', float_format='%.2f', index=None)


# ## 4.1. Heatmap with relationship between Factor 1 (TP1) and original sNYCQ items 

# In[25]:


F1_vals_to_plot = pd.DataFrame(Q.sort_values(by='Factor 1', ascending=True)['Factor 1'].round(0).astype(int)).T
f = F1_vals_to_plot.hvplot.heatmap(cmap='Bone', clim=(0,110), width=700, height=100,colorbar=False, yaxis=None).opts(toolbar=None, line_color='k', line_width=1)
TP1_vector = f * hv.Labels(f).opts(opts.Labels(text_color='white', xrotation=0, fontsize={ 'labels':12,'xticks':14}))
#hv.save(TP1_vector, osp.join('figures', 'Figure01_D-SNYCQ_ICQF_TP1_Values.html'), backend='bokeh')
TP1_vector


# ![TP1 Values](./figures/Figure01_D-SNYCQ_ICQF_TP1_Values.png)
# 
# ### Save Punblication Ready and Source Data

# In[26]:


svg_plot = TP1_vector.opts(hooks=[svg_backend])

bokeh_obj = hv.render(svg_plot, backend="bokeh")

if isinstance(bokeh_obj, Plot):
    bokeh_obj.output_backend = "svg"

for p in bokeh_obj.select({"type": Plot}):
    p.output_backend = "svg"

save(
    bokeh_obj,
    filename="./figures/Figure01_D-SNYCQ_ICQF_TP1_Values.html",
    resources=INLINE,
    title="Figure01_D-SNYCQ_ICQF_TP1_Values",
)


# In[29]:


F1_vals_to_plot.to_csv('./source_data_files/figure_01_d.csv', float_format='%.0f', index=None)


# ## 4.2. Heatmap with relationship between Factor 2 (TP2) and original sNYCQ items

# In[ ]:


F2_vals_to_plot = pd.DataFrame(Q.sort_values(by='Factor 2', ascending=True)['Factor 2'].round(0).astype(int))
f = F2_vals_to_plot.hvplot.heatmap(cmap='Bone', clim=(0,110), width=200, height=700,colorbar=False, xaxis=None).opts(toolbar=None, line_color='k', line_width=1)
TP2_vector = f * hv.Labels(f).opts(opts.Labels(text_color='white', xrotation=90, fontsize={'labels':12,'yticks':14}))
#hv.save(TP2_vector, osp.join('figures', 'Figure01_E-SNYCQ_ICQF_TP2_Values.html'), backend='bokeh')
TP2_vector


# ![TP2 Values](./figures/Figure01_D-SNYCQ_ICQF_TP2_Values.png)
# 
# ### Save Publication Ready and source data

# In[31]:


svg_plot = TP2_vector.opts(hooks=[svg_backend])

bokeh_obj = hv.render(svg_plot, backend="bokeh")

if isinstance(bokeh_obj, Plot):
    bokeh_obj.output_backend = "svg"

for p in bokeh_obj.select({"type": Plot}):
    p.output_backend = "svg"

save(
    bokeh_obj,
    filename="./figures/Figure01_E-SNYCQ_ICQF_TP2_Values.html",
    resources=INLINE,
    title="Figure01_E-SNYCQ_ICQF_TP2_Values",
)


# In[33]:


F2_vals_to_plot.to_csv('./source_data_files/figure_01_e.csv', float_format='%.0f')


# ## 4.3 Plot examples of scans sitting on both corners of the embedding

# In[34]:


sorted_q = Q.sort_values(by=['Factor 1','Factor 2'],ascending=False).index


# In[35]:


top_left_scans = W[(W['Factor 1']<0.35) & (W['Factor 2']>0.9)].index
a = SNYCQ.loc[top_left_scans].reset_index(drop=True)
top_left_scans_plot = a[sorted_q].hvplot.heatmap(width=250, height=250, clim=(0,100), cmap='Viridis', ylabel='Scan', xlabel='Question', fontscale=1.2).opts(colorbar=False, xrotation=90, toolbar=None)
hv.save(top_left_scans_plot, osp.join('figures', 'Figure01_C-SNYCQ_ICQF_TopLeftScans.html'), backend='bokeh')
top_left_scans_plot


# ![Top Left Scans](./figures/Figure01_C-SNYCQ_ICQF_TopLeftScans.png)
# 
# ### Saving SVG and Data source

# In[42]:


svg_plot = top_left_scans_plot.opts(hooks=[svg_backend])

bokeh_obj = hv.render(svg_plot, backend="bokeh")

if isinstance(bokeh_obj, Plot):
    bokeh_obj.output_backend = "svg"

for p in bokeh_obj.select({"type": Plot}):
    p.output_backend = "svg"

save(
    bokeh_obj,
    filename="./figures/Figure01_C-SNYCQ_ICQF_TopLeftScans.html",
    resources=INLINE,
    title="Figure01_C-SNYCQ_ICQF_TopLeftScans",
)


# In[39]:


a[sorted_q].to_csv('./source_data_files/figure_01_c_top_left_scans.csv', float_format='%.0f', index=None)


# Let's now show a few representaive scans from the bottom right corner

# In[44]:


bot_right_scans = W[(W['Factor 1']>0.9) & (W['Factor 2']<0.1)].index
a = SNYCQ.loc[bot_right_scans].reset_index(drop=True)
bot_right_scans_plot = a[sorted_q].hvplot.heatmap(width=250, height=250, clim=(0,100), cmap='Viridis', ylabel='Scan', xlabel='Question', fontscale=1.2).opts(colorbar=False, xrotation=90, toolbar=None)
hv.save(bot_right_scans_plot, osp.join('figures', 'Figure01_C-SNYCQ_ICQF_BotRightScans.html'), backend='bokeh')
bot_right_scans_plot


# ![Bottom Right Scans](./figures/Figure01_C-SNYCQ_ICQF_BotRightScans.png)
# 
# ### Save a SVG and Data Source

# In[43]:


svg_plot = bot_right_scans_plot.opts(hooks=[svg_backend])

bokeh_obj = hv.render(svg_plot, backend="bokeh")

if isinstance(bokeh_obj, Plot):
    bokeh_obj.output_backend = "svg"

for p in bokeh_obj.select({"type": Plot}):
    p.output_backend = "svg"

save(
    bokeh_obj,
    filename="./figures/Figure01_C-SNYCQ_ICQF_BotRightScans.html",
    resources=INLINE,
    title="Figure01_C-SNYCQ_ICQF_BotRightScans",
)


# In[45]:


a[sorted_q].to_csv('./source_data_files/figure_01_c_bot_right_scans.csv', float_format='%.0f', index=None)


# Save ICQF embedding to disk

# In[32]:


W.to_csv('../resources/snycq/SNYCQ_W.csv')

