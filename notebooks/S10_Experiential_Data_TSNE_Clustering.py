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
#     display_name: FC Instrospection py 3.10 | 2023b
#     language: python
#     name: fc_introspection_2023b_py310
# ---

from utils.basics import get_sbj_scan_list, DATA_DIR, ORIG_SNYCQ_PATH, RESOURCES_DINFO_DIR, RESOURCES_SNYCQ_DIR, ORIG_DEMO_PATH
from textwrap import wrap
import os.path as osp
from scipy.stats import ttest_ind, mannwhitneyu, ttest_rel, wilcoxon
import hvplot.pandas
import holoviews as hv
import seaborn as sns
from tqdm.notebook import  tqdm


# +
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import panel as pn

from sklearn.preprocessing import RobustScaler
from sklearn.covariance import MinCovDet
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, adjusted_rand_score
from sklearn.manifold import TSNE, trustworthiness
from sklearn.linear_model import LinearRegression
# -

# Configurations for outlier detection, ambigous clusters and random seed

OUTLIER_Q = 0.997          # robust cutoff on squared Mahalanobis distances
AMBIG_THRESH = 0.8         # ambiguous if max(prob) < 0.8
RANDOM_STATE = 42

# # 1. Load the SNYCQ data

SBJs, SCANs, SNYCQ_wVigilance = get_sbj_scan_list(when='post_motion', return_snycq=True)
SNYCQ              = SNYCQ_wVigilance.drop('Vigilance',axis=1)
Nscans, Nquestions = SNYCQ.shape
print(SNYCQ.shape)
SNYCQ_items        = SNYCQ.columns

# # 2. Data Scaling
# Answers to all questions, except wakefulness, were scaled using scikit-learn ```RobustScaler```. This scaler object removes the median and scales the data by the inter-quantile range. This form of scaling was performed to avoid excessive influence of outliers in the scaling process. 
#

X_raw_df    = SNYCQ[SNYCQ_items].replace([np.inf, -np.inf], np.nan).dropna(axis=0)
idx         = X_raw_df.index
X_scaled    = RobustScaler().fit_transform(X_raw_df.values)
X_scaled_df = pd.DataFrame(X_scaled,index=idx, columns=X_raw_df.columns)

X_raw_df.hvplot.hist(title='SNYC-Q pre scaling') + X_scaled_df.hvplot.hist(title='SNYC-Q post scaling', shared_axes=False)

# # 3. Outlier Detection
#
# We relied on the Mahalanobis distance of each scan to the sample's mean in order to detect outlier scans. We set the threshold to the top 3% quantile

# +
mcd = MinCovDet().fit(X_scaled_df.values)
# squared Mahalanobis distances for the training set
md2 = mcd.mahalanobis(X_scaled_df.values) if hasattr(mcd, "mahalanobis") else mcd.dist_

# Threshold
thr = np.quantile(md2, OUTLIER_Q)
keep = md2 < thr
# -

md2_df = pd.DataFrame(md2, index=idx)
md2_df.hvplot(hover_cols=['Subject','Run'], title='Mahalanobis ditance',ylabel='Mahalanobis distance') *hv.HLine(thr).opts(line_width=0.5, line_dash='dashed', line_color='k')

# Now that we now who the outlier scans are, we create a new version of the data where those scans have been removed

idx_kept          = idx[keep] 
X_scaled_kept_df  = X_scaled_df.loc[idx_kept]   # Scaled data for scans not marked as outliers
X_raw_kept_df     = X_raw_df.loc[idx_kept]      # Original data for scans not marked as outliers
print(f"[Outliers] Flagged {(~keep).sum()} / {len(md2)}; keeping {keep.sum()}")

# # 3. Correlation between SNYCQ items
#
# To explore the structure of in-scanner experience reports, we first computed the Pearson’s correlation between the 11 in-scanner experience items.

# +
X_raw_kept_corr_df = X_raw_kept_df.corr()
clustergrid        = sns.clustermap(X_raw_kept_corr_df)
plt.close()

# Get the reordered row indices
row_order          = clustergrid.dendrogram_row.reordered_ind
index_order        = X_raw_kept_corr_df.index[row_order]

# Create interactive heatmap usin the sorting that seaborn did for us
heatmap = X_raw_kept_corr_df.loc[index_order,index_order].round(1).hvplot.heatmap(clim=(-.7,.7),cmap='RdBu_r',aspect='square', frame_width=500,fontscale=1.5).opts(xrotation=45, clabel='Pearson Correation')
myplot  = heatmap * hv.Labels(heatmap).opts(text_color='k')
myplot


# -

# # 4. Dimensionality Reduction with T-SNE
# ## 4.1. Hyper-parameter Optimization
# Two key hyper-parameters of the T-SNE algorithm are dimensionality and perplexity. We relied on trustworthiness [REF]—an estimate of how well a given embedding preserves local distances—to select these two hyper-parameters on a data-driven manner. The explored hyper-parameter space was: 
#
# | Hyper-parameter | Space Explored |
# |:----------------|:---------------|
# |dimensionality|  {1,2,3}|
# | perplexity | {5,9,10,13,15,18,20,30,40,50} |

# +
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
# -

#print("\n[t-SNE tuning] Candidate perplexities & trustworthiness:")
#print(tw_table.to_string(index=False, float_format=lambda v: f"{v:0.3f}"))
print(f"[t-SNE tuning] Selected perplexity = {best_perp}")
print(f"[t-SNE tuning] Selected dimensionaity = {best_nc}")
print(f"[t-SNE tuning] Trustworthiness = {tw_table['trustworthiness'].max()}")

# ## 4.2. Compute T-SNE with optimal dimensionality and perplexity

tsne = TSNE(
    n_components=best_nc, perplexity=best_perp, learning_rate='auto', init='pca',
    n_iter_without_progress=1000, early_exaggeration=12.0, random_state=RANDOM_STATE
)
Y = tsne.fit_transform(X_scaled_kept_df.values)
if best_nc == 2:
    emb = pd.DataFrame(Y, index=idx_kept, columns=["TSNE1", "TSNE2"])
else:
    emb = pd.DataFrame(Y, index=idx_kept, columns=["TSNE1", "TSNE2","TSNE3"])


# ## 4.3. Compute Bipolar Arrows
#
# To aid with the interpretation of how T-SNE dimensions relate to the original items in the SNYC survey, we modeled each of the SNYCQ items as a linear function of the three T-SNE coordinates. We then used these coefficients to draw biplot arrows depicting the directions of maximal change for each SNYCQ item in the 3D T-NSE embedding space.

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


tsne_arrows = compute_biplot_arrows(emb,X_raw_kept_df,SNYCQ_items)
# Scaling so that they are clearly visible in the plot
tsne_arrows['beta_x'] = tsne_arrows['beta_x'] * 5 
tsne_arrows['beta_y'] = tsne_arrows['beta_y'] * 5
tsne_arrows['beta_z'] = tsne_arrows['beta_z'] * 5

# ## 4.4. Plot the T-SNE embedding colored by one SNYCQ item

# Create a new DF with both th original values and the dimensions in T-SNE (for plotting purposes)
emb_plus = pd.concat([emb, X_raw_kept_df], axis=1)
print(emb_plus.shape)

# +
tsne_camera_object = dict(center=dict( x=6.661338147750939e-16, y=-1.942890293094024e-16, z=8.881784197001252e-16 ),
                    eye=dict( x=-1.5176226342505497, y=0.4505864572659788, z=-1.6428294069570146),
                    up=dict(x= -0.7264163680411282, y= 0.05191238226958195, z= 0.6852914451596732))

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
    height=700
)

fig.show()

# -

# # 5. Clustering
#
# To separate scans into two sets with well-differentiated inner-experience, we decided to apply two different clustering algorithms to the in-experience data (11 questions) following robust scaling. To be clear, clustering was performed in the original 11D space, not the low dimensional space generated by T-SNE. Working with two different clustering methods allows to evaluate the robustness of results against clustering technique.
#
# The two selected methods were K-Means and Gaussian Mixture Modeling; both as implemented in the python library scikit-learn. K-Means was chosen as a representative hard-clustering method often used in the neuroimaging literature. Gaussian Mixture Modeling was chosen because of its soft-clustering nature (i.e., it provides membership probabilities) and because it allows non-spherical clusters. In both instances, we set k=2.

K          = 2
gm         = GaussianMixture(n_components=K, covariance_type="spherical", random_state=RANDOM_STATE).fit(X_scaled_kept_df.values)
proba      = gm.predict_proba(X_scaled_kept_df.values)
labels_gmm = proba.argmax(axis=1)


# +

km = KMeans(n_clusters=K, random_state=RANDOM_STATE, n_init=10).fit(X_scaled_kept_df.values)
labels_km = km.predict(X_scaled_kept_df.values)
# -

# ## 5.1. Cluster method comparison with ARI and Silhouette Index

# +
sil_gmm = silhouette_score(X_scaled_kept_df.values, labels_gmm)
sil_km  = silhouette_score(X_scaled_kept_df.values, labels_km)

ari_km_gmm = adjusted_rand_score(labels_km, labels_gmm)

print(f"[GMM k=2 sph] silhouette={sil_gmm:.3f}")
print(f"[K-M k=2    ] silhouette={sil_km:.3f}")

print(f"[Agreement] ARI(KMeans vs GMM) = {ari_km_gmm:.3f}")
print(f"[Sizes] GMM clusters: {np.bincount(labels_gmm)}")

# -

# ## 5.2. Bootstraing Analysis for GMM

# +
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
# -

# ## 5.3. Detection of scans with ambigous cluster membership

proba_df = pd.DataFrame(proba, index=idx_kept, columns=['P(c1)','P(c2)'])
proba_df.hvplot.hist('P(c1)')

# store probabilities & an ambiguity flag
p1        = proba[:, 1]
ambiguity = np.maximum(1 - p1, p1) < (AMBIG_THRESH)

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

emb_plus = pd.concat([emb_plus, group_info_df], axis=1)

# +
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
    height=700
)

fig.show()

# -

#
# ## 6. See what's difference or not across groups
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

# +
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

# Add zero for bins with zero counts (needed for plotting)
age_bins_avail = list(demographics['age (5-year bins)'].unique())
for abin in age_bins_avail:
    if ("Set A",abin) not in age_counts_per_group.index:
        age_counts_per_group.loc[("Set A",abin)] = 0
    if ("Set B",abin) not in age_counts_per_group.index:
        age_counts_per_group.loc[("Set B",abin)] = 0


# Prepare Dataframe for plotting with hvplot
age_counts_per_group = age_counts_per_group.reset_index()
age_counts_per_group.columns = ['Group','Age Range','# Scans']
age_counts_per_group.replace({'Set A':'A', 'Set B':'B'}, inplace=True)
age_counts_per_group['color'] = '#ffffff'
age_counts_per_group.loc[age_counts_per_group['Group']=='A','color'] = group_colors['Set A']
age_counts_per_group.loc[age_counts_per_group['Group']=='B','color'] = group_colors['Set B']
age_counts_per_group = age_counts_per_group.infer_objects()
age_counts_per_group = age_counts_per_group.sort_values(by='Age Range', ascending=True)
# -

# Compute statistics to chedk for significant differences across groups
W, p = wilcoxon(age_counts_per_group.set_index(['Group','Age Range']).loc['A',:]['# Scans'],age_counts_per_group.set_index(['Group','Age Range']).loc['B',:]['# Scans'], alternative='two-sided', method='exact')
if p < 0.05:
    sig_string = 'p < 0.05'
else:
    sig_string = 'n.s.'

# Generate graph that will get later added to a Grid with information about all variables
age_bar_plot = age_counts_per_group.hvplot.bar(x='Age Range',by='Group', color='color', alpha=0.5, xlabel='Age',title=f'Wilcoxon = %.1f | {sig_string}' % W).opts(toolbar=None, xrotation=90, width=225, height=200, fontscale=.8)

# ## 6.2. Examination of differneces in gender distribution

# Extract information about age
sex_per_scan = pd.DataFrame(index=idx_kept, columns=['Set Label','Sex'])
for sbj,run in tqdm(idx_kept):
    sex         = demographics.loc[sbj,'gender']
    group_label = emb_plus.loc[(sbj,run),'Set Label']
    sex_per_scan.loc[(sbj,run),'Sex']         = sex
    sex_per_scan.loc[(sbj,run),'Set Label'] = group_label
# Remove entries for ambiguous scans
sex_per_scan = sex_per_scan[sex_per_scan['Set Label']!='Ambiguous']
# Get counts of scas in each age range 
sex_counts_per_group = sex_per_scan.groupby('Set Label').value_counts()
sex_counts_per_group = sex_counts_per_group.infer_objects()
sex_counts_per_group

sex_bar_plot = sex_counts_per_group.hvplot.bar(stacked=True, xlabel='', legend='top_left', title='Sex Distribution', ylabel='# Scans').opts(toolbar=None, width=225, height=200)
sex_bar_plot

# ## 6.3. Examination of diffrences in head motion

# Load motion information for each scan
mot_info = pd.read_csv(osp.join(RESOURCES_DINFO_DIR,'motion_confounds.csv'),index_col=['Subject','Run'])

# ## 6.4. Examination of differences in SNYCQ items

# +
# 4) Interpretability: Cohen's d per feature
df      = pd.concat([SNYCQ_wVigilance.loc[idx_kept], mot_info.loc[idx_kept]], axis=1)
X       = df.values
X_items = df.columns
g0 = labels_gmm == 0
g1 = labels_gmm == 1

def cohens_d(x0, x1):
    m0, m1 = np.nanmean(x0), np.nanmean(x1)
    s0, s1 = np.nanstd(x0, ddof=1), np.nanstd(x1, ddof=1)
    n0, n1 = np.sum(~np.isnan(x0)), np.sum(~np.isnan(x1))
    sp = np.sqrt(((n0-1)*s0**2 + (n1-1)*s1**2) / (n0 + n1 - 2))
    return (m1 - m0) / sp if sp > 0 else np.nan

d_vals = []
for j, f in enumerate(X_items):
    d = cohens_d(X[g0, j], X[g1, j])
    u, p = mannwhitneyu(X[g0, j], X[g1, j],alternative='two-sided')
    m0, m1 = np.nanmean(X[g0, j]), np.nanmean(X[g1, j])
    d_vals.append((f, d, u, p, m0, m1))

d_df = pd.DataFrame(d_vals, columns=["feature", "cohens_d (1-0)", "MannWhiney U", "p_value", "men_set_a", "men_set_b"]) \
       .sort_values("cohens_d (1-0)", key=lambda s: s.abs(), ascending=False)

print("\nTop features by |Cohen's d| (group separation):")
print(d_df.to_string(index=False, float_format=lambda v: f"{v:0.3f}"))

# +
group_a_scans = group_info_df.reset_index().set_index('Set Label').loc['Set A'].set_index(['Subject','Run']).index
group_b_scans = group_info_df.reset_index().set_index('Set Label').loc['Set B'].set_index(['Subject','Run']).index

items = list(d_df.set_index(["feature"]).abs().sort_values(by="cohens_d (1-0)", ascending=False).index)
# -

layout = pn.GridBox(ncols=3)
for item in items:
    this_pval = d_df.set_index('feature').loc[item]["p_value"]
    if this_pval > 0.05:
        title = "Cohen d=%.2f | n.s." % d_df.set_index('feature').loc[item]["cohens_d (1-0)"]
    else:
        title = "Cohen d=%.2f | p < 0.05" % d_df.set_index('feature').loc[item]["cohens_d (1-0)"] 
    if item != "Mean Rel Motion":
        bins = [0,10,20,30,40,50,60,70,80,90,100]
    else:
        bins = 20
    overlay = df.loc[group_a_scans,item].hvplot.hist(label='Set A', c=group_colors['Set A'], title=title, width=225, height=200, alpha=0.5, shared_axes=False, bins=bins, normed=True) * \
              df.loc[group_b_scans,item].hvplot.hist(label='Set B', c=group_colors['Set B'], alpha=0.5, shared_axes=False, bins=bins, normed=True) * \
              df.loc[group_a_scans,item].hvplot.kde(label='Set A', c=group_colors['Set A'], title=title, width=300, height=200, alpha=0.5, shared_axes=False) * \
              df.loc[group_b_scans,item].hvplot.kde(label='Set B', c=group_colors['Set B'], alpha=0.5, shared_axes=False)
    overlay = overlay.opts(show_legend=False,shared_axes=False, toolbar=None)
    layout.append(overlay)
layout.append(age_bar_plot)
layout.append(sex_bar_plot)
layout

# +
from scipy.stats import norm
# --------- 1) Build Cohen's d summary table from SNYCQ ----------
def cohens_d_ci_boot(x0, x1, B=2000, random_state=123):
    """Bootstrap 95% CI for Cohen's d (cluster1 - cluster0)."""
    rng = np.random.default_rng(random_state)
    x0 = np.asarray(x0); x1 = np.asarray(x1)
    x0 = x0[np.isfinite(x0)]; x1 = x1[np.isfinite(x1)]
    n0, n1 = len(x0), len(x1)
    if n0 < 2 or n1 < 2:
        return np.nan, np.nan
    d_boot = []
    for _ in range(B):
        b0 = x0[rng.integers(0, n0, n0)]
        b1 = x1[rng.integers(0, n1, n1)]
        m0, m1 = b0.mean(), b1.mean()
        s0, s1 = b0.std(ddof=1), b1.std(ddof=1)
        sp = np.sqrt(((n0-1)*s0**2 + (n1-1)*s1**2) / max(n0+n1-2, 1))
        d_boot.append((m1 - m0) / sp if sp > 0 else np.nan)
    d_boot = np.array([v for v in d_boot if np.isfinite(v)])
    if len(d_boot) == 0:
        return np.nan, np.nan
    lo, hi = np.percentile(d_boot, [2.5, 97.5])
    return float(lo), float(hi)

def make_cohens_d_summary(SNYCQ, SNYCQ_items, label_col="Set Label", B=2000, random_state=123):
    """Return a DataFrame with d, CI, r, AUC, and per-cluster means for each feature."""
    # keep rows with binary labels {0,1}
    df = SNYCQ[SNYCQ['Set Label']!='Ambiguous']
    assert set(df[label_col].unique()) <= {'Set A','Set B'}, "Labels must be binary (Set A/Set B)."

    rows = []
    for f in SNYCQ_items:
        x0 = df.loc[df[label_col] == 'Set A', f].values
        x1 = df.loc[df[label_col] == 'Set B', f].values
        m0, m1 = np.nanmean(x0), np.nanmean(x1)
        s0, s1 = np.nanstd(x0, ddof=1), np.nanstd(x1, ddof=1)
        n0, n1 = np.sum(~np.isnan(x0)), np.sum(~np.isnan(x1))
        sp = np.sqrt(((n0-1)*s0**2 + (n1-1)*s1**2) / max(n0+n1-2, 1))
        d = (m1 - m0) / sp if sp > 0 else np.nan
        # Translations
        r = d / np.sqrt(d**2 + 4) if np.isfinite(d) else np.nan
        auc = norm.cdf(d / np.sqrt(2)) if np.isfinite(d) else np.nan
        # Bootstrap CI
        lo, hi = cohens_d_ci_boot(x0, x1, B=B, random_state=random_state)
        rows.append((f, d, lo, hi, r, auc, m0, m1))

    d_plus = pd.DataFrame(rows, columns=[
        "feature", "d (1-0)", "d_CI_low", "d_CI_high",
        "r_approx", "AUC_approx", "mean_cluster0", "mean_cluster1"
    ])
    return d_plus

# --------- 3) Radar chart of per-cluster means ----------
def plot_radar_means(d_plus, order=None, savepath=None):
    df = d_plus.copy()
    df.set_index(['feature'], inplace=True)
    features = order if order is not None else df["feature"].tolist()
    df = df.loc[features]

    angles = np.linspace(0, 2*np.pi, len(features), endpoint=False).tolist()
    angles += angles[:1]
    vals0 = df["mean_cluster0"].tolist(); vals0 += vals0[:1]
    vals1 = df["mean_cluster1"].tolist(); vals1 += vals1[:1]

    plt.figure(figsize=(7, 7))
    ax = plt.subplot(111, polar=True)
    ax.plot(angles, vals0, linewidth=2, label="Cluster 0")
    ax.plot(angles, vals1, linewidth=2, label="Cluster 1")
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(features)

    all_vals = np.array(df[["mean_cluster0","mean_cluster1"]]).flatten()
    rmin, rmax = float(np.min(all_vals)), float(np.max(all_vals))
    pad = max(1.0, 0.05*(rmax - rmin))
    ax.set_ylim(rmin - pad, rmax + pad)

    ax.set_title("Per-cluster item means (radar)")
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.show()



# -

list(d_plus.sort_values(by="d (1-0)", ascending=False)['feature'])

d_plus = make_cohens_d_summary(emb_plus, SNYCQ_items)
plot_radar_means(d_plus, order=list(d_plus.sort_values(by="d (1-0)", ascending=False)['feature']))

# ***
# # 8. How often scans fall in the same cluster
#
# First, we count how many scans we have per subject and keep that information on a pandas Series object

outlier_scans = [i for i in SNYCQ.index if i not in idx_kept]
print(outlier_scans)

SNYCQ = SNYCQ.drop(outlier_scans)
SCANs = SNYCQ.index
SNYCQ.shape

SBJs        = list(idx_kept.get_level_values('Subject').unique())
N_MIN_SCANS = 2
scans_per_subject = pd.Series(index=SBJs, dtype=int)
for sbj in scans_per_subject.index:
    aux                    = SNYCQ.loc[sbj,:]
    scans_per_subject[sbj] = aux.shape[0]
sbjs_sel_scans = list(scans_per_subject[scans_per_subject > N_MIN_SCANS].index)
assert scans_per_subject.sum() == len(SCANs)

Nsbjs_total       = len(SNYCQ.index.get_level_values('Subject').unique())
Nsbjs_sel_scans   = len(sbjs_sel_scans)
print('++ INFO: Number of subjects in these analyses    : %d subjects' % Nsbjs_total)
print('++ INFO: Number of subjects with 3 or more scans : %d subjects' % Nsbjs_sel_scans)


# Now, for each participant with 3 or more scans, we check for three possible configurations:
#
# 1. All scans in the same set
# 2. All scans, except 1, in the same set.
# 3. Any other combination across the three sets

def count_scans_per_group(sbjs,cluster_info):
    # Extract from cluster_info structure the entries for scans with 2 or more scans
    df = pd.DataFrame(0, index=sbjs, columns=['All scans in same set','All except one scan in same set','Other configurations'], dtype=int)
    aux = cluster_info.loc[sbjs,'Set Label']
    n_scans = aux.shape[0]
    for sbj in aux.index.get_level_values('Subject'):
        auxx = aux.loc[sbj,:]
        auxx_max = auxx.value_counts().max()
        if auxx_max == 4:
            df.loc[sbj,'All scans in same set'] = 1
        elif auxx_max == (auxx.shape[0]-1):
            df.loc[sbj,'All except one scan in same set'] = 1
        else:
            df.loc[sbj,'Other configurations'] = 1
    return df.sum()


import seaborn as sns
fig,axs = plt.subplots(1,1,figsize=(7,7))
final_counts = count_scans_per_group(sbjs_sel_scans, emb_plus)
labels       = final_counts.index
labels       = [ '\n'.join(wrap(l, 15)) for l in labels ]
f            = axs.pie(final_counts, colors=sns.color_palette("ch:start=.2,rot=-.3"), labels=labels,autopct='%.0f%%');
plt.tight_layout()
