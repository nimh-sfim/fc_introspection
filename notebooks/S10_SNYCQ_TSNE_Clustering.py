#!/usr/bin/env python
# coding: utf-8

# # Description: SNYCQ Initial Exploration, Clustering and TSNE
# 
# This notebook contains the following analytical steps associated with the in-scanner experience data
# 
# 1. Data scaling: this is accomplished using skicit-learn [```RobustScaler```](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html)
# 
# 2. Outlier detection
# 
# 3. Creates figure looking a potential correlations between sNYCQ items
# 
# 4. Dimensionality reduction with T-SNE
# 
# 5. Clustering analysis in original 11D space 

# In[1]:


from utils.basics import get_sbj_scan_list, RESOURCES_DINFO_DIR, RESOURCES_SNYCQ_DIR, ORIG_DEMO_PATH
from utils.plotting import show_correlations_with_statistics
import os.path as osp
from scipy.stats import ttest_ind, mannwhitneyu, wilcoxon
import hvplot.pandas
import holoviews as hv
import seaborn as sns
from tqdm.notebook import  tqdm

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import panel as pn
from scipy.stats import pearsonr

from sklearn.preprocessing import RobustScaler
from sklearn.covariance import MinCovDet
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.manifold import TSNE, trustworthiness
from sklearn.linear_model import LinearRegression


# Configurations for outlier detection, ambigous clusters and random seed

# In[3]:


OUTLIER_Q = 0.997          # robust cutoff on squared Mahalanobis distances
AMBIG_THRESH = 0.8         # ambiguous if max(prob) < 0.8
RANDOM_STATE = 42


# ***
# # 1. Load In-scanner Experience Data (SNYCQ)
# 
# We load this data only for the scans that have passed our QA for the imaging data

# In[4]:


SBJs, SCANs, SNYCQ_wVigilance = get_sbj_scan_list(when='post_motion', return_snycq=True)
SNYCQ              = SNYCQ_wVigilance.drop('Vigilance',axis=1)
Nscans, Nquestions = SNYCQ.shape
print(SNYCQ.shape)
SNYCQ_items        = SNYCQ.columns


# ***
# # 2. Data Scaling
# Answers to all questions, except wakefulness, were scaled using scikit-learn ```RobustScaler```. This scaler object removes the median and scales the data by the inter-quantile range. This form of scaling was performed to avoid excessive influence of outliers in the scaling process. 
# 

# In[5]:


X_raw_df    = SNYCQ[SNYCQ_items].replace([np.inf, -np.inf], np.nan).dropna(axis=0)
idx         = X_raw_df.index
X_scaled    = RobustScaler().fit_transform(X_raw_df.values)
X_scaled_df = pd.DataFrame(X_scaled,index=idx, columns=X_raw_df.columns)


# We look at the distributions of the sNYCQ data before and after scaling

# In[6]:


plot_dist = X_raw_df.hvplot.hist(title='SNYC-Q pre scaling') + X_scaled_df.hvplot.hist(title='SNYC-Q post scaling', shared_axes=False)
hv.save(plot_dist, osp.join('figures', 'FigureXX-SNYCQ_histograms_pre_post_scaling.html'))
plot_dist


# ![Distributions of SNYQC items before and after scaling](./figures/FigureXX-SNYCQ_histograms_pre_post_scaling.png)

# ***
# # 3. Outlier Detection
# 
# We relied on the Mahalanobis distance of each scan to the sample's mean in order to detect outlier scans. We set the threshold to the top 3% quantile

# In[7]:


mcd = MinCovDet().fit(X_scaled_df.values)
# squared Mahalanobis distances for the training set
md2 = mcd.mahalanobis(X_scaled_df.values) if hasattr(mcd, "mahalanobis") else mcd.dist_

# Threshold
thr = np.quantile(md2, OUTLIER_Q)
keep = md2 < thr


# In[8]:


md2_df = pd.DataFrame(md2, index=idx)
md2_df.hvplot(hover_cols=['Subject','Run'], title='Mahalanobis ditance',ylabel='Mahalanobis distance') *hv.HLine(thr).opts(line_width=0.5, line_dash='dashed', line_color='k')


# Now that we have identified two outlier scans, we create a new version of the data where those scans have been removed (e.g., ```X_$$$$_kept```)

# In[9]:


idx_kept          = idx[keep] 
X_scaled_kept_df  = X_scaled_df.loc[idx_kept]   # Scaled data for scans not marked as outliers
X_raw_kept_df     = X_raw_df.loc[idx_kept]      # Original data for scans not marked as outliers
print(f"[Outliers] Flagged {(~keep).sum()} / {len(md2)}; keeping {keep.sum()}")


# # 4. Correlation between SNYCQ items
# 
# To explore the structure of in-scanner experience reports, we first computed the Pearson’s correlation between the 11 in-scanner experience items.

# In[10]:


# Compute correlation matrix
X_raw_kept_corr_df = X_raw_kept_df.corr()


# In[11]:


# Estimate  P-value matrix (Pearson)
cols    = X_raw_kept_df.columns
pval_df = pd.DataFrame(np.zeros((len(cols), len(cols))),
                       index=cols, columns=cols)

for i, c1 in enumerate(cols):
    for j, c2 in enumerate(cols):
        if i <= j:
            r, p = pearsonr(X_raw_kept_df[c1], X_raw_kept_df[c2])
            pval_df.loc[c1, c2] = p
            pval_df.loc[c2, c1] = p


# In[12]:


# Get clustering order from seaborn
clustergrid = sns.clustermap(X_raw_kept_corr_df)
plt.close()

row_order = clustergrid.dendrogram_row.reordered_ind
ordered   = X_raw_kept_corr_df.index[row_order]

corr_ord = X_raw_kept_corr_df.round(2).loc[ordered, ordered]
pval_ord = pval_df.loc[ordered, ordered]


# In[13]:


# Make sure names exist (used by show_results)
corr_ord.index.name   = 'index'
corr_ord.columns.name = 'col'
corr_ord.name         = 'corr'

pval_ord.index.name   = 'index'
pval_ord.columns.name = 'col'
pval_ord.name         = 'pval'


# In[14]:


# Calcualte the number of unique entries to do Bonferroni correction
n_comps = corr_ord.shape[0]*(corr_ord.shape[0]-1)/2


# In[ ]:


# Plot: correlation heatmap with bold black outline for pBonf < 0.05
plot = show_correlations_with_statistics(
    data_val=corr_ord,
    data_pval=pval_ord,
    pval_thr=0.05/n_comps,
    clabel='Pearson correlation',
    height=700,
    width=800,
    cmap='RdBu_r',
    fontscale=1.5,
    clim=(-0.7, 0.7)
)


# ![Figure 01 - Panel B](./figures/Figure01_B-SNYCQcorr.png)
# 
# ### Saving Publication ready format data and figure panel
# 
# We will save now the panel as HTML in a way that we can then export to PDF via the "Print..." option of the browser

# In[16]:


import holoviews as hv

from bokeh.io import save
from bokeh.models.plots import Plot
from bokeh.resources import INLINE

hv.extension("bokeh")

def svg_backend(hv_plot, element):
    hv_plot.state.output_backend = "svg"

svg_plot = plot.opts(hooks=[svg_backend])

bokeh_obj = hv.render(svg_plot, backend="bokeh")

# Extra safety for layouts / overlays
if isinstance(bokeh_obj, Plot):
    bokeh_obj.output_backend = "svg"

for p in bokeh_obj.select({"type": Plot}):
    p.output_backend = "svg"

save(
    bokeh_obj,
    filename="./figures/Figure01_B-SNYCQcorr.html",
    resources=INLINE,
    title="Figure01_B",
)


# In[17]:


corr_ord.to_csv('./source_data_files/figure_01_b.csv',float_format='%.2f')
print('Saving source data for Figure 01B: SNYCQ correlation matrix with significant correlations outlined in black [./source_data_files/figure_01_b.csv]')


# ***
# # 4. Dimensionality Reduction with T-SNE
# ## 4.1. Hyper-parameter Optimization
# Two key hyper-parameters of the T-SNE algorithm are dimensionality and perplexity. We relied on [trustworthiness](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.trustworthiness.html)—an estimate of how well a given embedding preserves local distances—to select these two hyper-parameters on a data-driven manner. The explored hyper-parameter space was: 
# 
# | Hyper-parameter | Space Explored |
# |:----------------|:---------------|
# |dimensionality|  {1,2,3}|
# | perplexity | {5,9,10,13,15,18,20,30,40,50} |

# In[18]:


def choose_tsne_perplexity(X, random_state=RANDOM_STATE,n_components=[1,2,3]):
    n = X.shape[0]
    # Theoretical upper bound: perplexity < (n - 1); practical upper ~ n/3
    max_perp = max(5, min(60, (n - 1) // 3))
    base = min(max_perp, max(5, int(round(n / 50))))  # ~ n/50 baseline, clamped [5, 60]
    # Candidate set around 'base' plus some classics
    cands = sorted(set([
        5, 10, 15, 20, 30, 40, 50,
        base, int(base*1.5), int(base*2)
    ]))
    cands = [p for p in cands if 5 <= p <= max_perp and p < (n - 1)]
    scores = []
    for nc in tqdm(n_components, position=0, desc='Num Components'):
        for p in tqdm(cands, position=1, desc='Perplexity', leave=False):
            tsne_tmp = TSNE(n_components=nc, perplexity=p, learning_rate='auto', init='pca',
                            n_iter_without_progress=600, early_exaggeration=12.0, random_state=random_state)
            Ytmp = tsne_tmp.fit_transform(X)
            tw = trustworthiness(X, Ytmp, n_neighbors=10, metric='euclidean')
            scores.append((nc,p, tw))
    # pick the perplexity with max trustworthiness; break ties by preferring mid-range
    scores.sort(key=lambda t: (-t[2], abs(t[0] - 30)))
    return scores[0][0], scores[0][1], pd.DataFrame(scores, columns=["num_components","perplexity", "trustworthiness"])

best_nc, best_perp, tw_table = choose_tsne_perplexity(X_scaled_kept_df.values, RANDOM_STATE)


# Now we show what were the values that maximized the trustworthiness of the T-SNE embeddings

# In[19]:


print(f"[t-SNE tuning] Selected perplexity = {best_perp}")
print(f"[t-SNE tuning] Selected dimensionaity = {best_nc}")
print(f"[t-SNE tuning] Trustworthiness = {tw_table['trustworthiness'].max()}")


# ## 4.2. Compute T-SNE with optimal dimensionality and perplexity

# In[20]:


tsne = TSNE(
    n_components=best_nc, perplexity=best_perp, learning_rate='auto', init='pca',
    n_iter_without_progress=1000, early_exaggeration=12.0, random_state=RANDOM_STATE
)
Y = tsne.fit_transform(X_scaled_kept_df.values)
if best_nc == 2:
    emb = pd.DataFrame(Y, index=idx_kept, columns=["TSNE1", "TSNE2"])
else:
    emb = pd.DataFrame(Y, index=idx_kept, columns=["TSNE1", "TSNE2","TSNE3"])


# ## 4.3. Compute Biplot Arrows inidicating directions of maximal variance for each SNYCQ item
# 
# To aid with the interpretation of how T-SNE dimensions relate to the original items in the SNYC survey, we modeled each of the SNYCQ items as a linear function of the three T-SNE coordinates. We then used these coefficients to draw biplot arrows depicting the directions of maximal change for each SNYCQ item in the 3D T-NSE embedding space.

# In[21]:


def compute_biplot_arrows(emb_df, features_df, feature_names):
    """
    For each feature f: fit f ~ a*X + b*Y on the 2D embedding; return direction (a,b) and R^2.
    features_df should be standardized/robust-scaled (we use Xr_kept).
    """
    Xy = emb_df.values  # shape (n,2)
    n_components = Xy.shape[1]
    arrows = []
    reg = LinearRegression()
    for j, f in enumerate(feature_names):
        y = features_df.iloc[:, j].values
        reg.fit(Xy, y)
        if n_components == 2:
            a, b = reg.coef_
            r2 = reg.score(Xy, y)
            arrows.append((f, a, b, r2))
        elif n_components == 3:
            a,b,c = reg.coef_
            r2 = reg.score(Xy,y)
            arrows.append((f,a,b,c,r2))
        else:
            print('++ ERROR: This function only works with 2D and 3D embeddings')
            return None
    if n_components==2:
        arrows_df = pd.DataFrame(arrows, columns=["feature", "beta_x", "beta_y", "R2"]).sort_values("R2", ascending=False)
    elif n_components == 3:
        arrows_df = pd.DataFrame(arrows, columns=["feature", "beta_x", "beta_y", "beta_z","R2"]).sort_values("R2", ascending=False)        
    return arrows_df


# In[22]:


tsne_arrows = compute_biplot_arrows(emb,X_raw_kept_df,SNYCQ_items)
# Scaling so that they are clearly visible in the plot
tsne_arrows['beta_x'] = tsne_arrows['beta_x'] * 5 
tsne_arrows['beta_y'] = tsne_arrows['beta_y'] * 5
tsne_arrows['beta_z'] = tsne_arrows['beta_z'] * 5


# ## 4.4. Plot the T-SNE embedding colored by one SNYCQ item

# In[23]:


# Create a new DF with both th original values and the dimensions in T-SNE (for plotting purposes)
emb_plus = pd.concat([emb, X_raw_kept_df], axis=1)


# In[24]:


tsne_camera_object = dict(
    center=dict(x=6.661338147750939e-16, y=-1.942890293094024e-16, z=8.881784197001252e-16),
    eye=dict(x=0.4505864572659788, y=1.5176226342505497, z=-1.6428294069570146),
    up=dict(x=-0.7264163680411282, y=0.05191238226958195, z=0.6852914451596732)
)
# Create the scatter trace
scatter_trace = go.Scatter3d(
    x=emb_plus['TSNE1'],
    y=emb_plus['TSNE2'],
    z=emb_plus['TSNE3'],
    mode='markers',
    marker=dict(
        size=5,                         # Fixed small size
        color=emb_plus['People'],      # Color mapped to 'People'
        colorscale='viridis',          # Viridis colormap
        opacity=0.8,
        colorbar=dict(title='People')  # Optional colorbar
    ),
    text=[
        "<br>".join(f"{col}: {row[col]}" for col in emb_plus.columns if col not in ['TSNE1', 'TSNE2', 'TSNE3'])
        for _, row in emb_plus.iterrows()
    ],
    hoverinfo='text'
)

# === 2. Arrows from (0, 0, 0) to (beta_x, beta_y, beta_z) ===

arrow_traces = []

for i, row in tsne_arrows.iterrows():
    # Line (arrow body)
    if row['feature'] == 'People':
        arrow_color = 'red'
    else:
        arrow_color = 'black'
    arrow_trace = go.Scatter3d(
        x=[0, row['beta_x']],
        y=[0, row['beta_y']],
        z=[0, row['beta_z']],
        mode='lines',
        line=dict(
            color=arrow_color,
            width=max(1, row['R2'] * 10),  # Scale width by R2
        ),
        showlegend=False
    )
    
    # Text label at arrow tip
    label_trace = go.Scatter3d(
        x=[row['beta_x']],
        y=[row['beta_y']],
        z=[row['beta_z']],
        mode='text',
        text=[row['feature']],
        textposition='top center',
        showlegend=False
    )
    
    arrow_traces.extend([arrow_trace, label_trace])


# === 3. Combine all traces ===

fig = go.Figure(data=[scatter_trace] + arrow_traces)

# === 4. Update layout ===

fig.update_layout(
    scene=dict(
        xaxis_title='TSNE1',
        yaxis_title='TSNE2',
        zaxis_title='TSNE3',
        aspectmode='data',
        camera=tsne_camera_object

    ),
    title='3D t-SNE Plot with Feature Arrows',
    margin=dict(l=0, r=0, b=0, t=40),
    height=700,
    showlegend=False
)

fig.show()


# In[25]:


fig.write_html(osp.join('figures', 'FigureXX-SNYCQtsne_colorby_People.html'))


# ![TSNE Map colored by the People question](./figures/FigureXX-SNYCQtsne_colorby_People.png)

# *** 
# # 5. Clustering Analyses
# 
# ## 5.1 Apply K-means and Gaussian Mixture Modeling to the data (K=2)
# 
# To separate scans into two sets with well-differentiated inner-experience, we decided to apply two different clustering algorithms to the in-experience data (11 questions) following robust scaling. 
# 
# >NOTE: To be clear, clustering was performed in the original 11D space, not the low dimensional space generated by T-SNE. 
# 
# Working with two different clustering methods allows to evaluate the robustness of results against clustering technique.
# 
# The two selected methods were [K-Means](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html) and [Gaussian Mixture Modeling](https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html); both as implemented in the python library scikit-learn. 
# 
# K-Means was chosen as a representative hard-clustering method often used in the neuroimaging literature. 
# 
# Gaussian Mixture Modeling was chosen because of its soft-clustering nature (i.e., it provides membership probabilities) and because it allows non-spherical clusters. In both instances, we set k=2.
# 
# The next cell computes the GMM clustering for K = 2

# In[26]:


K          = 2
gm         = GaussianMixture(n_components=K, covariance_type="spherical", random_state=RANDOM_STATE).fit(X_scaled_kept_df.values)
proba      = gm.predict_proba(X_scaled_kept_df.values)
labels_gmm = proba.argmax(axis=1)


# Now we do the same using KMeans for comparison

# In[27]:


km = KMeans(n_clusters=K, random_state=RANDOM_STATE, n_init=10).fit(X_scaled_kept_df.values)
labels_km = km.predict(X_scaled_kept_df.values)


# ## 5.2. Cluster method comparison with ARI and Silhouette Index
# 
# We now check for the consistency of the clustering results across both methods using the Asjusted Rand Index (ARI), and also for their quality, separately, using the Silhouette Index (SI)
# 
# 1. Compute the SI for each clustering result

# In[28]:


sil_gmm = silhouette_score(X_scaled_kept_df.values, labels_gmm)
sil_km  = silhouette_score(X_scaled_kept_df.values, labels_km)


# 2. Compute the ARI comparing both methods

# In[29]:


ari_km_gmm = adjusted_rand_score(labels_km, labels_gmm)


# 3. Print the computed statistics

# In[30]:


print(f"[GMM k=2 sph] silhouette={sil_gmm:.3f}")
print(f"[K-M k=2    ] silhouette={sil_km:.3f}")

print(f"[Agreement] ARI(KMeans vs GMM) = {ari_km_gmm:.3f}")
print(f"[Sizes] GMM clusters: {np.bincount(labels_gmm)}")


# Based on the SI, we decided to move forward with the GMM solustion, which we explore in further detail.

# ## 5.3. Bootstraing Analysis for GMM

# In[31]:


def subsample_labels(model, X, frac=0.8, seed=0):
    rng = np.random.RandomState(seed)
    n = X.shape[0]
    take = np.sort(rng.choice(n, int(frac*n), replace=False))  # sort for convenience
    # clone model with same hyperparams
    m2 = type(model)(**model.get_params())
    if hasattr(m2, "random_state"):
        m2.random_state = seed
    # (KMeans has fit_predict; GMM needs fit + predict)
    if hasattr(m2, "fit_predict"):
        labels = m2.fit_predict(X[take])
    else:
        m2.fit(X[take]); labels = m2.predict(X[take])
    return take, labels

def bootstrap_ari(model, X, n_runs=20, frac=0.8, base_seed=0):
    aris = []
    for s in range(n_runs):
        i1, l1 = subsample_labels(model, X, frac=frac, seed=base_seed + 2*s)
        i2, l2 = subsample_labels(model, X, frac=frac, seed=base_seed + 2*s + 1)
        # align to the same samples, same order
        common = np.intersect1d(i1, i2)
        # positions of the common indices in each subsample
        pos1 = np.searchsorted(i1, common)
        pos2 = np.searchsorted(i2, common)
        aris.append(adjusted_rand_score(l1[pos1], l2[pos2]))
    return float(np.mean(aris)), float(np.std(aris))

gm_mean_ari, gm_sd_ari = bootstrap_ari(gm, X_scaled_kept_df.values, n_runs=20, frac=0.8, base_seed=100)
print("Bootstrap ARI (GMM) mean±sd:    %.2f +/- %.2f" %(gm_mean_ari, gm_sd_ari))


# ***
# ## 5.4. Detection of scans with ambigous cluster membership
# 
# One additional bonus of GMM over K-means is that GMM not being a hard clustering algorithm, it outputs cluster membershup probabilities. We will use these to detect scans with ambiguous membership. Such scans will not be included in the population differences analyses later on.

# In[32]:


proba_df = pd.DataFrame(proba, index=idx_kept, columns=['P(c1)','P(c2)'])
proba_df.hvplot.hist('P(c1)', title='Probability Distribution for Membership in Cluster 1')


# In[33]:


# store probabilities & an ambiguity flag
p1        = proba[:, 1]
ambiguity = np.maximum(1 - p1, p1) < (AMBIG_THRESH)


# Save final cluster/sets labels in a new pandas dataframe: ```group_info_df```

# In[34]:


# Store that information with clear labels in a pandas Dataframe
group_info_df = pd.DataFrame(index=idx_kept, columns=['Set Label', 'Group Probability'])
for i,scan in enumerate(idx_kept):
    if ambiguity[i]:
        group_info_df.loc[scan,"Set Label"] = "Ambiguous"
        group_info_df.loc[scan,"Group Probability"] = np.max([p1[i],1-p1[i]])
    else:
        if p1[i] > 1 - p1[i]:
            group_info_df.loc[scan,"Set Label"] = "Set B"
            group_info_df.loc[scan,"Group Probability"] = p1[i]
        else:
            group_info_df.loc[scan,"Set Label"] = "Set A"
            group_info_df.loc[scan,"Group Probability"] = 1 - p1[i]
group_info_df['Set Label'].value_counts()


# Add cluster membership information to the TSNE embedding information, and save two versions to disk: one with the original SNYCQ items, one with their scaled values.

# In[ ]:


emb_plus = pd.concat([emb_plus, group_info_df], axis=1)
emb_plus.to_csv(osp.join(RESOURCES_SNYCQ_DIR, 'SNYCQ_tsne_embeddings_plus.csv'))
print('++ INFO: Saved t-SNE embeddings with original features and group info to CSV: %s' % osp.join(RESOURCES_SNYCQ_DIR, 'SNYCQ_tsne_embeddings_plus.csv'))
emb_plus_scaled = pd.concat([emb, X_scaled_kept_df, group_info_df], axis=1)
emb_plus_scaled.to_csv(osp.join(RESOURCES_SNYCQ_DIR, 'SNYCQ_tsne_embeddings_plus_scaled.csv'))
print('++ INFO: Saved t-SNE embeddings with scaled features and group info to CSV: %s' % osp.join(RESOURCES_SNYCQ_DIR, 'SNYCQ_tsne_embeddings_plus_scaled.csv'))


# ## 5.5. Plot the T-SNE embedding again, but this time with scans colored according to set membership

# In[36]:


tsne_camera_object = dict(
    center=dict(x=0.0, y=0.0, z=0.0),
    eye=dict(x=1.25, y=-1.25, z=1.25),
    up=dict(x=0.0, y=0.0, z=0.0)
)
scatter_traces = []

group_colors = {
    "Set A": "#1f77b4",       # light blue
    "Set B": "#ff7f0e",       # orange
    "Ambiguous": "#ffffff"    # white fill
}

for group, color in group_colors.items():
    group_data = emb_plus[emb_plus["Set Label"] == group]
    scatter_traces.append(
        go.Scatter3d(
            x=group_data["TSNE1"],
            y=group_data["TSNE2"],
            z=group_data["TSNE3"],
            mode='markers',
            name=group,
            marker=dict(
                size=5,
                color=color,
                opacity=0.9,
                line=dict(
                    color='black' if group == "Ambiguous" else color,
                    width=2 if group == "Ambiguous" else 0
                )
            ),
            text=[
                "<br>".join(
                    f"{col}: {row[col]}"
                    for col in emb_plus.columns if col not in ['TSNE1', 'TSNE2', 'TSNE3']
                )
                for _, row in group_data.iterrows()
            ],
            hoverinfo='text'
        )
    )
# === 3. Combine all traces ===
fig = go.Figure(data=scatter_traces + arrow_traces)

fig.update_layout(
    scene=dict(
        xaxis_title='TSNE1',
        yaxis_title='TSNE2',
        zaxis_title='TSNE3',
        aspectmode='data',
        camera=tsne_camera_object
    ),
    legend=dict(
        title='Group:',
        itemsizing='constant',
        x=0.55,   # move horizontally (1.0 is far right, 0.5 is middle)
        y=0.9,   # move vertically (1.0 is top)
        xanchor='left',
        yanchor='top',
        bgcolor='rgba(255,255,255,0.7)',  # optional background for readability
        bordercolor='rgba(0,0,0,0.2)',
        borderwidth=1
    ),
    title='3D t-SNE Plot with Feature Arrows',
    margin=dict(l=0, r=0, b=0, t=40),
    height=700,
    showlegend=True
)

fig.show()


# In[37]:


fig.write_html(osp.join('figures', 'Figure01_K_SNYCQtsneWclusters.html'))


# ![TSNE with clusters](./figures/Figure01_K-SNYCQtnseWclusters.png)

# ### Save Publication Ready Figure Panel and Data source

# In[38]:


fig.write_image(osp.join('figures', 'Figure01_K_SNYCQtsneWclusters.svg'))


# In[39]:


emb_plus[['TSNE1', 'TSNE2', 'TSNE3','Set Label']].to_csv('./source_data_files/figure_01_k.csv', float_format='%.3f', index=False)


# ***
# ## 6. Explore how Sets A and B differ in terms of SNYCQ items, vigilance, head motion and basic demographics
# 
# We will seek for potential differences across groups in the following variables:
# 
# * All entries in the SNYCQ, including wakefulness
# * Mean head motion
# * Age distribution
# * Gender distribution
# 
# We will do this using Cohen's d, MannWhitney tests and Wilconxon tests

# ## 6.1. Examination of differences in age distribution

# In[43]:


# Load Demographic Data
demographics = pd.read_csv(ORIG_DEMO_PATH, index_col=0,sep='\t')
demographics = demographics.loc[list(SBJs)]
# Load Demographic Data
#demographics = pd.read_csv(osp.join(RESOURCES_SNYCQ_DIR,'participants_post_motion_QA.csv'), index_col=0)
# Extract information about age
age_per_scan = pd.DataFrame(index=idx_kept, columns=['Set Label','Age (5-year bins)'])
for sbj,run in tqdm(idx_kept):
    age_range = demographics.loc[sbj,'age (5-year bins)']
    group_label = emb_plus.loc[(sbj,run),'Set Label']
    age_per_scan.loc[(sbj,run),'Age (5-year bins)'] = age_range
    age_per_scan.loc[(sbj,run),'Set Label'] = group_label
# Remove entries for ambiguous scans
age_per_scan = age_per_scan[age_per_scan['Set Label']!='Ambiguous']
# Get counts of scas in each age range 
age_counts_per_group = age_per_scan.groupby('Set Label').value_counts()
age_counts_per_group = age_counts_per_group.infer_objects()

# Prepare Dataframe for plotting with hvplot
age_counts_per_group = age_counts_per_group.reset_index()
age_counts_per_group.columns = ['Group','Age Range','# Scans']
age_counts_per_group.replace({'Set A':'A', 'Set B':'B'}, inplace=True)
age_counts_per_group['color'] = '#ffffff'
age_counts_per_group.loc[age_counts_per_group['Group']=='A','color'] = group_colors['Set A']
age_counts_per_group.loc[age_counts_per_group['Group']=='B','color'] = group_colors['Set B']
age_counts_per_group = age_counts_per_group.infer_objects()
age_counts_per_group = age_counts_per_group.sort_values(by='Age Range', ascending=True)


# In[44]:


A = age_counts_per_group.set_index(['Group','Age Range']).loc['A',:]['# Scans']
B = age_counts_per_group.set_index(['Group','Age Range']).loc['B',:]['# Scans']
W, w_p = wilcoxon(A,B, alternative='two-sided', method='exact')
print('++ AGE ACROSS SETS: Wilcoxon = %.2f (p = %.2f)' % (W,w_p))


# In[ ]:


# Generate graph that will get later added to a Grid with information about all variables
age_bar_plot = age_counts_per_group.hvplot.bar(x='Age Range',by='Group', alpha=0.5, xlabel='Age',cmap=["#1f77b4","#ff7f0e"]).opts(toolbar=None, xrotation=90, width=250, height=200, fontscale=1)


# ![Age per set](./figures/Figure01_I-AgePerSet.png)

# ### Save publication ready and source data for Figure 01I

# In[48]:


age_counts_per_group.to_csv('./source_data_files/figure_01_i.csv', index=False)


# In[50]:


svg_plot = age_bar_plot.opts(hooks=[svg_backend])

bokeh_obj = hv.render(svg_plot, backend="bokeh")

# Extra safety for layouts / overlays
if isinstance(bokeh_obj, Plot):
    bokeh_obj.output_backend = "svg"

for p in bokeh_obj.select({"type": Plot}):
    p.output_backend = "svg"

save(
    bokeh_obj,
    filename="./figures/Figure01_I-AgePerSet.html",
    resources=INLINE,
    title="Figure01_I",
)


# ## 6.2. Examination of differneces in gender distribution

# In[51]:


# Extract information about age
sex_per_scan = pd.DataFrame(index=idx_kept, columns=['Set Label','Sex'])
for sbj,run in tqdm(idx_kept):
    sex         = demographics.loc[sbj,'gender']
    if sex == 'M': 
        sex = 'Male'
    else:
        sex = 'Female'
    group_label = emb_plus.loc[(sbj,run),'Set Label']
    sex_per_scan.loc[(sbj,run),'Sex']         = sex
    sex_per_scan.loc[(sbj,run),'Set Label'] = group_label
# Remove entries for ambiguous scans
sex_per_scan = sex_per_scan[sex_per_scan['Set Label']!='Ambiguous']
# Get counts of scas in each age range 
sex_counts_per_group = sex_per_scan.groupby('Set Label').value_counts()
sex_counts_per_group = sex_counts_per_group.infer_objects()
sex_counts_per_group


# In[52]:


sex_bar_plot = sex_counts_per_group.hvplot.bar(stacked=True, xlabel='', legend='top_left', title='', ylabel='# Scans', color=['white','gray']).opts(toolbar=None, width=250, height=200, fontscale=1)
#hv.save(sex_bar_plot, osp.join('figures', 'Figure01_J-SexPerSet.html'))


# ![Sex per Set](./figures/Figure01_J-SexPerSet.png)
# 
# ### Save Publication Ready and Data Source

# In[56]:


svg_plot = sex_bar_plot.opts(hooks=[svg_backend])

bokeh_obj = hv.render(svg_plot, backend="bokeh")

# Extra safety for layouts / overlays
if isinstance(bokeh_obj, Plot):
    bokeh_obj.output_backend = "svg"

for p in bokeh_obj.select({"type": Plot}):
    p.output_backend = "svg"

save(
    bokeh_obj,
    filename="./figures/Figure01_J-SexPerSet.html",
    resources=INLINE,
    title="Figure01_J",
)


# In[55]:


sex_counts_per_group.to_csv('./source_data_files/figure_01_j.csv')


# ## 6.3. Examination of diffrences in head motion

# In[57]:


# Load motion information for each scan
mot_info   = pd.read_csv(osp.join(RESOURCES_DINFO_DIR,'motion_confounds.csv'),index_col=['Subject','Run'])
scans_in_A = emb_plus[emb_plus['Set Label'] == 'Set A'].index
scans_in_B = emb_plus[emb_plus['Set Label'] == 'Set B'].index
mot_A      = mot_info.loc[scans_in_A,'Mean Rel Motion'].values
mot_B      = mot_info.loc[scans_in_B,'Mean Rel Motion'].values

U, u_p     = mannwhitneyu(mot_A,mot_B,alternative='two-sided')
print('++ AGE ACROSS SETS: Mann-Whiteney U    = %.2f (p = %.2f)' % (U,u_p))


# In[ ]:


mot_A = mot_info.loc[scans_in_A,'Mean Rel Motion']
mot_B = mot_info.loc[scans_in_B,'Mean Rel Motion']
overlay = mot_A.hvplot.hist(label='Set A', c=group_colors['Set A'], title='', width=250, height=200, alpha=0.5, shared_axes=False, bins=20, normed=True).opts(toolbar=None) * \
          mot_B.hvplot.hist(label='Set B', c=group_colors['Set B'], alpha=0.5, shared_axes=False, bins=20, normed=True) * \
          mot_A.hvplot.kde(label='Set A', c=group_colors['Set A'],  alpha=0.5, shared_axes=False) * \
          mot_B.hvplot.kde(label='Set B', c=group_colors['Set B'], alpha=0.5, shared_axes=False)


# ![Motion per set](./figures/Figure01_H-MotionPerSet.png)
# 
# ### Save Publication ready figure and Source Data

# In[59]:


svg_plot = overlay.opts(hooks=[svg_backend])

bokeh_obj = hv.render(svg_plot, backend="bokeh")

# Extra safety for layouts / overlays
if isinstance(bokeh_obj, Plot):
    bokeh_obj.output_backend = "svg"

for p in bokeh_obj.select({"type": Plot}):
    p.output_backend = "svg"

save(
    bokeh_obj,
    filename="./figures/Figure01_H-MotionPerSet.html",
    resources=INLINE,
    title="Figure01_H",
)


# In[64]:


pd.concat([mot_A.reset_index(drop=True), mot_B.reset_index(drop=True)], axis=1, keys=['Set A','Set B']).to_csv('./source_data_files/figure_01_h.csv', index=False)


# ## 6.4 Examination of Vigilance
# 

# In[65]:


vigilance = SNYCQ_wVigilance['Vigilance']
vigilance.name = 'Wakefulness'
scans_in_A = emb_plus[emb_plus['Set Label'] == 'Set A'].index
scans_in_B = emb_plus[emb_plus['Set Label'] == 'Set B'].index
vigilance_A      = vigilance.loc[scans_in_A].values
vigilance_B      = vigilance.loc[scans_in_B].values

U, u_p      = mannwhitneyu(vigilance_A,vigilance_B,alternative='two-sided')
print('++ VIGILANCE ACROSS SETS: Mann-Whiteney U    = %.2f (p = %.2f)' % (U,u_p))


# In[66]:


vigilance_A      = vigilance.loc[scans_in_A]
vigilance_B      = vigilance.loc[scans_in_B]
overlay = vigilance_A.hvplot.hist(label='Set A', c=group_colors['Set A'], title='', width=250, height=200, alpha=0.5, shared_axes=False, bins=[0,10,20,30,40,50,60,70,80,90,100], normed=True, legend=False).opts(toolbar=None) * \
          vigilance_B.hvplot.hist(label='Set B', c=group_colors['Set B'], alpha=0.5, shared_axes=False, bins=[0,10,20,30,40,50,60,70,80,90,100], normed=True) * \
          vigilance_A.hvplot.kde(label='Set A', c=group_colors['Set A'],  alpha=0.5, shared_axes=False) * \
          vigilance_B.hvplot.kde(label='Set B', c=group_colors['Set B'], alpha=0.5, shared_axes=False)
#hv.save(overlay, osp.join('figures', 'Figure01_G-VigilancePerSet.html'))


# ![Wakefulness per set](./figures/Figure01_G-VigilancePerSet.png)
# ### Save Publication Ready Panel and Source Data

# In[67]:


svg_plot = overlay.opts(hooks=[svg_backend])

bokeh_obj = hv.render(svg_plot, backend="bokeh")

# Extra safety for layouts / overlays
if isinstance(bokeh_obj, Plot):
    bokeh_obj.output_backend = "svg"

for p in bokeh_obj.select({"type": Plot}):
    p.output_backend = "svg"

save(
    bokeh_obj,
    filename="./figures/Figure01_G-VigilancePerSet.html",
    resources=INLINE,
    title="Figure01_G",
)


# In[68]:


pd.concat([vigilance_A.reset_index(drop=True), vigilance_B.reset_index(drop=True)], axis=1, keys=['Set A','Set B']).to_csv('./source_data_files/figure_01_g.csv', index=False)


# ## 6.5. Examination of differences in SNYCQ items

# In[69]:


def cohens_d(x0, x1):
    m0, m1 = np.nanmean(x0), np.nanmean(x1)
    s0, s1 = np.nanstd(x0, ddof=1), np.nanstd(x1, ddof=1)
    n0, n1 = np.sum(~np.isnan(x0)), np.sum(~np.isnan(x1))
    sp = np.sqrt(((n0-1)*s0**2 + (n1-1)*s1**2) / (n0 + n1 - 2))
    return (m1 - m0) / sp if sp > 0 else np.nan

def calculate_stats_per_set(df, items, label_col='Set Label', method="bootstrap",
                            B=2000,              # bootstrap reps
                            ci=95,random_state=123):
    
    rng = np.random.default_rng(random_state)
    alpha_low = (100 - ci) / 2.0
    alpha_high = 100 - alpha_low

    out = []
    for f in items:
        x0 = df.loc[df[label_col] == "Set A", f].dropna().values
        x1 = df.loc[df[label_col] == "Set B", f].dropna().values
        m0 = float(np.mean(x0)) if x0.size else np.nan
        m1 = float(np.mean(x1)) if x1.size else np.nan
        d  = cohens_d(x0,x1) 
        u, p = mannwhitneyu(x0,x1,alternative='two-sided')
        if method == "bootstrap":
            # nonparametric bootstrap of the mean per cluster
            if x0.size >= 2:
                boots0 = [np.mean(x0[rng.integers(0, x0.size, x0.size)]) for _ in range(B)]
                lo0, hi0 = np.percentile(boots0, [alpha_low, alpha_high])
            else:
                lo0 = hi0 = np.nan

            if x1.size >= 2:
                boots1 = [np.mean(x1[rng.integers(0, x1.size, x1.size)]) for _ in range(B)]
                lo1, hi1 = np.percentile(boots1, [alpha_low, alpha_high])
            else:
                lo1 = hi1 = np.nan

        elif method == "analytic":
            # mean ± z * SE; SE = s / sqrt(n)
            z = 1.96 if ci == 95 else None
            if z is None:
                from scipy.stats import norm
                z = float(norm.ppf(0.5 + ci/200.0))
            if x0.size >= 2:
                se0 = float(np.std(x0, ddof=1) / np.sqrt(x0.size))
                lo0, hi0 = m0 - z*se0, m0 + z*se0
            else:
                lo0 = hi0 = np.nan
            if x1.size >= 2:
                se1 = float(np.std(x1, ddof=1) / np.sqrt(x1.size))
                lo1, hi1 = m1 - z*se1, m1 + z*se1
            else:
                lo1 = hi1 = np.nan
        else:
            raise ValueError("method must be 'bootstrap' or 'analytic'.")

        out.append((f, d, u,p, m0, lo0, hi0, m1, lo1, hi1))

    out_df = pd.DataFrame(out, columns=["Item", "d(Set B - Set A)", "MW (U)","MW (p)","Set A (mean)", "lo (Set A)", "hi (Set A)", "Set B (mean)", "lo (Set B)", "hi (Set B)"]).set_index("Item")
    return out_df


# In[70]:


non_ambiguous_scans = emb_plus[emb_plus['Set Label']!='Ambiguous'].index
data_items          = SNYCQ_wVigilance.loc[non_ambiguous_scans,:]
data_labels         = emb_plus.loc[non_ambiguous_scans,'Set Label']
data                = pd.concat([data_items,data_labels],axis=1)
stats_per_set       = calculate_stats_per_set(data,[c for c in data_items.columns if c !='Vigilance'],'Set Label')
table_02 = stats_per_set.sort_values(by="d(Set B - Set A)", ascending=False).round(2)[['Set A (mean)','Set B (mean)','d(Set B - Set A)','MW (U)','MW (p)']]
table_02


# In[71]:


table_02.to_csv('./source_data_files/table_02.csv', float_format='%.2f')


# In[72]:


layout                = pn.GridBox(ncols=4)
items_in_descending_d = stats_per_set.sort_values(by="d(Set B - Set A)", ascending=False).index
set_a_scans = emb_plus[emb_plus['Set Label']=='Set A'].index
set_b_scans = emb_plus[emb_plus['Set Label']=='Set B'].index
for item in items_in_descending_d:
    bins = [0,10,20,30,40,50,60,70,80,90,100]
    this_pval = stats_per_set.loc[item,"MW (p)"]
    if this_pval >= 0.01:
        title = "Cohen d=%.2f | n.s." % stats_per_set.loc[item]["d(Set B - Set A)"]
    else:
        title = "Cohen d=%.2f | p < 0.01" % stats_per_set.loc[item]["d(Set B - Set A)"] 
    overlay = SNYCQ.loc[set_a_scans,item].hvplot.hist(label='Set A', c=group_colors['Set A'], title=title, width=225, height=200, alpha=0.5, shared_axes=False, bins=bins, normed=True) * \
              SNYCQ.loc[set_b_scans,item].hvplot.hist(label='Set B', c=group_colors['Set B'], alpha=0.5, shared_axes=False, bins=bins, normed=True) * \
              SNYCQ.loc[set_a_scans,item].hvplot.kde(label='Set A', c=group_colors['Set A'], title=title, width=300, height=200, alpha=0.5, shared_axes=False) * \
              SNYCQ.loc[set_b_scans,item].hvplot.kde(label='Set B', c=group_colors['Set B'], alpha=0.5, shared_axes=False)
    overlay = overlay.opts(show_legend=False,shared_axes=False, toolbar=None)
    layout.append(overlay)
layout.save( osp.join('figures', 'Supplementary_Figure02.html'))


# ![Supplementary Figure 02](./figures/Supplementary_Figure02.png)

# Provide the same information in more concise manner in the form of a radar plot

# In[79]:


def plot_radar_means_with_ci(
    stats_df,               # output of cluster_means_ci (indexed by feature)
    order=None,             # optional ordering of features
    title="Per-cluster item means (with 95% CI)",
    ci_label="95% CI"
):
    """
    Draws radar with Cluster 0 & 1 mean lines and shaded CI bands.
    """
    # order features
    if order is None:
        features = list(stats_df.index)
    else:
        features = list(order)
        stats_df = stats_df.loc[features]

    # angles for axes + close the loop
    angles = np.linspace(0, 2*np.pi, len(features), endpoint=False).tolist()
    angles += angles[:1]

    # extract arrays and close loops
    m0 = stats_df["Set A (mean)"].to_numpy().tolist(); m0 += m0[:1]
    m1 = stats_df["Set B (mean)"].to_numpy().tolist(); m1 += m1[:1]
    lo0 = stats_df["lo (Set A)"].to_numpy().tolist(); lo0 += lo0[:1]
    hi0 = stats_df["hi (Set A)"].to_numpy().tolist(); hi0 += hi0[:1]
    lo1 = stats_df["lo (Set B)"].to_numpy().tolist(); lo1 += lo1[:1]
    hi1 = stats_df["hi (Set B)"].to_numpy().tolist(); hi1 += hi1[:1]

    # limits based on CI envelopes
    all_vals = np.array((lo0 + hi0 + lo1 + hi1), dtype=float)
    finite_vals = all_vals[np.isfinite(all_vals)]
    if finite_vals.size:
        rmin, rmax = float(np.nanmin(finite_vals)), float(np.nanmax(finite_vals))
    else:
        rmin, rmax = 0.0, 1.0
    pad = max(1.0, 0.05 * (rmax - rmin))

    # plot
    fig = plt.figure(figsize=(7, 7))
    ax = plt.subplot(111, polar=True)

    # mean lines
    l0, = ax.plot(angles, m0, linewidth=2, label="Set A")
    l1, = ax.plot(angles, m1, linewidth=2, label="Set B")

    # shaded CI polygons (build as upper path + reversed lower path)
    # cluster 0
    ang_np = np.array(angles)
    poly0_ang = np.concatenate([ang_np, ang_np[::-1]])
    poly0_rad = np.concatenate([np.array(hi0), np.array(lo0)[::-1]])
    s0 = ax.fill(poly0_ang, poly0_rad, alpha=0.15, label=f"Set A {ci_label}")

    # cluster 1
    poly1_ang = np.concatenate([ang_np, ang_np[::-1]])
    poly1_rad = np.concatenate([np.array(hi1), np.array(lo1)[::-1]])
    s1 = ax.fill(poly1_ang, poly1_rad, alpha=0.15, label=f"Set B {ci_label}")

    # axes & labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(features)
    ax.set_ylim(rmin - pad, rmax + pad)
    ax.set_title(title)
    ax.legend(loc="upper right", bbox_to_anchor=(0.1, 1.0))

    plt.tight_layout()
    plt.show()
    return fig


# In[80]:


# Plot (keep your preferred order of spokes)
plot = plot_radar_means_with_ci(stats_per_set, order=list(items_in_descending_d),
                          title="SNYCQ per-cluster means with 95% CI")


# ### Save Publication Ready Panel and Source Data for the Radar Plot

# In[88]:


stats_per_set.to_csv('./source_data_files/figure_01_f.csv', float_format='%.2f')


# In[85]:


plot.savefig('./figures/Figure01_F.svg', format='svg', bbox_inches='tight')


# ***
# 
# # Distribution of SNYCQ values (Supplementary Figure 1)

# In[86]:


layout = None
for q in SNYCQ.columns:
    plot = SNYCQ[q].reset_index(drop=True).hvplot.hist(bins=np.linspace(0,100,20), width=250, height=200, normed=True, ylabel='Density', fontsize=12) * SNYCQ[q].reset_index(drop=True).hvplot.kde()
    if layout is None:
        layout = plot
    else:
        layout = layout + plot
layout = layout.cols(3).opts(toolbar=None)
hv.save(layout, osp.join('figures', 'Supplementary_Figure01.html'))


# ![Supplementary Figure 01](./figures/Supplementary_Figure01.png)

# 
