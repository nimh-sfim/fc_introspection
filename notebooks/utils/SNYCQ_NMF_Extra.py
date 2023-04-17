from textwrap import wrap
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns

from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.cluster.xmeans import xmeans
from sklearn.cluster import AgglomerativeClustering

def plot_anwers_distribution_per_question(data,figsize=(16,6),xtick_fontsize=12, legend_fontsize=12):
    P = data.values.astype('float')
    Nscans, Nquestions = P.shape
    fig, ax = plt.subplots(figsize=figsize)
    counts = []
    for i in range(Nquestions):
        counts.append([ np.where( (np.minimum(P[:,i],99) >= j*10) * (np.minimum(P[:,i],99) < (j+1)*10) )[0].shape[0] for j in range(10)])
    counts = np.stack(counts)
    label_list = []
    for i in range(9):
        label_list.append('Score [{}, {})'.format(i*10, (i+1)*10))
    label_list.append('Score [{}, {}]'.format(90, 100))

    for i in range(10):
        ax.bar(np.linspace(1,P.shape[1],P.shape[1]),np.sum(counts[:,:(10-i)], axis=1), label=label_list[i])

    ax.set_xticks(np.linspace(1,P.shape[1],P.shape[1]))
    ax.set_xticklabels(data.columns.values, fontsize=xtick_fontsize, rotation=45)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.3),
          fancybox=True, shadow=True, ncol=5, fontsize=legend_fontsize)
    plt.tight_layout()
    ax.plot([0,Nquestions+1],[Nscans,Nscans],'k--')
    plt.close()
    return fig

  
def plot_Q_bars(Q, SNYCQ_data, SNYCQ_Questions, SNYCQ_Question_type, ylabel_type='short', figsize_x=16, fontsize=12, xlim=(0,100),axes=None):
    if isinstance(Q,pd.DataFrame):
       Q = Q.values
    nQ = Q.shape[0]
    k = Q.shape[1]
    if ylabel_type == 'short':
     question_list = SNYCQ_data.columns.values
    else:
     question_list = [SNYCQ_Questions[label] for label in SNYCQ_data.columns.values]
    type_list = [SNYCQ_Question_type[label] for label in SNYCQ_data.columns.values]
    wrap_question_list = [ '\n'.join(wrap(l, 100)) for l in question_list ]
    

    fig = plt.figure(figsize=(figsize_x,0.45*nQ))
    gs = matplotlib.gridspec.GridSpec(1,k,figure=fig)

    color_code = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 
                  'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

    upper_quantile = 0.7
    
    for d in range(k):
        factor = Q[:,d]

        factor_load = []
        factor_question = []
        factor_type = []
        factor_count = []
        factor_color = []
        type_color = []
        
        for (i, question) in enumerate(wrap_question_list[:nQ]):
            factor_load.append(factor[i])
            factor_type.append(type_list[i])
            factor_question.append(question)
            if factor[i] > np.quantile(factor[factor>0], upper_quantile):
                factor_color.append('tab:red')
            else:
                factor_color.append('tab:gray')
            if type_list[i] == 'Form':
                type_color.append('tab:brown')
            elif type_list[i] == 'Content':
                type_color.append('tab:green')
            elif type_list[i] == 'Misc':
                type_color.append('k')
        if axes is None:        
           ax = fig.add_subplot(gs[0,d])
        else:
           ax = axes[list(axes.keys())[d]]
        ax.barh(np.linspace(0,nQ-1,nQ), np.asarray(factor_load),
                color=factor_color, alpha=0.5,
                tick_label=factor_question)
        
        if d == 0:
            for t in range(len(factor_question)):
                ax.get_yticklabels()[t].set_color(type_color[t])
        else:
#             ax.set_yticks([])
            ax.set_yticklabels([])
            
        ax.grid(alpha=0.5)
        ax.invert_yaxis()

        ax.set_xticks([0, Q[:,d].max()])
        ax.set_xticklabels([0, int(Q[:,d].max())])
        ax.set_xlim(xlim)
        ax.tick_params(axis='y', labelsize=fontsize)
        ax.set_title('Factor {}'.format(d+1), fontsize=fontsize)
        ax.set_ylim([0-0.5, nQ-0.5])
        ax.vlines(np.quantile(factor[factor>0], upper_quantile), -0.5, nQ-0.5, colors='gray', linestyles='dashed',linewidth=1)

    fig.suptitle('', fontsize=fontsize, y=0.92)
    plt.tight_layout()
    plt.close()
    return fig
  
#def plot_W(W, label):
#    k = W.shape[1]
#    jump = np.where(np.sort(label)[1:]-np.sort(label)[:-1] > 0)[0]
#    center = np.append(jump,label.shape[0])
#    diff = (center[1:] - center[:-1])/2
#    center = center - np.append(center[0]/2, diff)
#    
#    fig, ax = plt.subplots(figsize=(16,0.5*k))
#    sns.heatmap((W[np.argsort(label),:]).T, cmap='Blues', ax=ax, cbar=False)
#    ax.set_yticks(np.arange(W.shape[1])+0.5)
#    ax.set_yticklabels(['Dim {}'.format(i+1) for i in range(k)], rotation=0, fontsize=14)
#    ax.set_xticks([])
#    ax.set_xlabel('Sorted scans', fontsize=16)
#    for (i, loc) in enumerate(jump):
#        ax.vlines(loc+1,0-0.5,W.shape[1], color='r')
#    for (i, loc) in enumerate(center):
#        ax.text(loc, -0.5, 'Group {}'.format(i+1), ha='left', rotation=45, fontsize=14)
#    bottom, top = ax.get_ylim()
#    ax.set_ylim(bottom + 0.5, top - 0.5)
#    plt.close()
#    return fig

def plot_W(W,cluster_labels=None):
    if isinstance(W,pd.DataFrame):
        W = W.values
    Nscans, Ndims = W.shape
    fig, ax = plt.subplots(figsize=(16,0.5*Ndims))
    if cluster_labels is None:
        scan_sorting = np.arange(Nscans)
        xlabel = 'Unsorted Scans'
    else:
        jump = np.where(np.sort(cluster_labels)[1:]-np.sort(cluster_labels)[:-1] > 0)[0]
        center = np.append(jump,cluster_labels.shape[0])
        diff = (center[1:] - center[:-1])/2
        center = center - np.append(center[0]/2, diff)
        scan_sorting = np.argsort(cluster_labels)
        xlabel = 'Sorted Scans'
    sns.heatmap((W[scan_sorting,:]).T, cmap='Blues', ax=ax, cbar=False)
    ax.set_yticks(np.arange(W.shape[1])+0.5)
    ax.set_yticklabels(['Dim {}'.format(i+1) for i in range(Ndims)], rotation=0, fontsize=14)
    ax.set_xticks([])
    ax.set_xlabel(xlabel, fontsize=16)
    if not (cluster_labels is None):
        for (i, loc) in enumerate(jump):
            ax.vlines(loc+1,0-0.5,W.shape[1], color='r')
        for (i, loc) in enumerate(center):
            ax.text(loc, -0.5, 'Group {}'.format(i+1), ha='left', rotation=45, fontsize=14)
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    plt.close()
    return fig
   
def group_questions(Q, SNYCQ_Questions, kinit = 2, kmax = 10):    
    k = Q.shape[1]
    initial_centers = kmeans_plusplus_initializer(Q, kinit).initialize()
    xmeans_instance = xmeans(Q, initial_centers, kmax)
    xmeans_instance.process()
    clusters = xmeans_instance.get_clusters()
    label = np.zeros(Q.shape[0])
    for k in range(len(clusters)): label[clusters[k]] = k
    k = len(clusters)
    return label, clusters
   
def cluster_scans(W,n_clusters=None, kmin=1, kmax=10):
    if n_clusters is None:
        print('++ INFO [cluster_scans]: Data-driven selection of k --> X-Kmeans')
        # Get initial set of clusters
        initial_centers = kmeans_plusplus_initializer(W, kmin).initialize()
        # Run the X-mean algorithm
        xmeans_instance = xmeans(W, initial_centers, kmax)
        xmeans_instance.process()
        # Get the clusters
        clusters = xmeans_instance.get_clusters()
        # Create cluster label array
        label = np.zeros(W.shape[0])
        for k in range(len(clusters)): label[clusters[k]] = k
    else:
        print('++ INFO [cluster_scans]: Doing agglomerative clustering for provied k = %d' % n_clusters)
        clustering = AgglomerativeClustering(n_clusters=n_clusters).fit(W)
        label = clustering.labels_
    return label

# DEPRECATED# def plot_W_scatter(W,figsize=(6,6), plot_hist=False, plot_kde=False, cluster_labels=None, cluster_palette='tab10', marker_size=10):
# DEPRECATED#     Nscans,Ndims = W.shape
# DEPRECATED#     assert Ndims == 2
# DEPRECATED#     f, ax = plt.subplots(figsize=figsize)
# DEPRECATED#     if plot_hist:
# DEPRECATED#         sns.histplot(data=W,    x='Factor 1', y='Factor 2', bins=10, pthresh=-1, cmap="viridis", alpha=.5)
# DEPRECATED#     if plot_kde:
# DEPRECATED#         sns.kdeplot(data=W,     x='Factor 1', y='Factor 2', levels=10, color="k", linewidths=1)
# DEPRECATED#     if cluster_labels is None:
# DEPRECATED#         sns.scatterplot(data=W, x='Factor 1', y='Factor 2', s=10, color='k')
# DEPRECATED#     else:
# DEPRECATED#         Wplot = W.copy()
# DEPRECATED#         Wplot['Cluster'] = cluster_labels
# DEPRECATED#         sns.scatterplot(data=Wplot, x='Factor 1', y='Factor 2', s=marker_size, hue='Cluster', palette=cluster_palette)
# DEPRECATED#     ax.set_xlim(-0.05,1.05)
# DEPRECATED#     ax.set_ylim(-0.05,1.05)
# DEPRECATED#     plt.close()
# DEPRECATED#     return f

def plot_W_scatter(W,figsize=(6,6), plot_hist=False, plot_kde=False, cluster_palette='tab10', marker_size=10, clusters_info=None):
    Nscans,Ndims = W.shape
    assert Ndims == 2
    fig, ax = plt.subplots(figsize=figsize)
    if plot_hist:
        sns.histplot(data=W,    x='Factor 1', y='Factor 2', bins=10, pthresh=-1, cmap="viridis", alpha=.5)
    if plot_kde:
        sns.kdeplot(data=W,     x='Factor 1', y='Factor 2', levels=10, color="k", linewidths=1)
    if not(clusters_info is None):
        cl_id2label = clusters_info.drop_duplicates().set_index('Cluster ID').to_dict()['Cluster Label']
        N_clusters = len(clusters_info['Cluster ID'].unique())
        aux = pd.concat([W,clusters_info],axis=1)
        sns.scatterplot(data=aux, x='Factor 1', y='Factor 2', s=marker_size, hue='Cluster ID', palette=cluster_palette)
        handles, labels  =  ax.get_legend_handles_labels()
        ax.legend(handles, [cl_id2label[c+1] for c in range(N_clusters)], loc='lower left')
    else:
        sns.scatterplot(data=W, x='Factor 1', y='Factor 2', s=marker_size, color='k', edgecolor='w', linewidth=1)
    ax.set_xlim(-0.005,1.005)
    ax.set_ylim(-0.005,1.005)
    plt.close()
    return fig
   
def plot_P(P,figsize=(16,4),question_order=None, scan_order=None, clusters_info=None, transpose=True):
    assert isinstance(P,pd.DataFrame)
    Nscans,Nquestions=P.shape
    fig, ax = plt.subplots(figsize=figsize)
    if not (question_order is None):
        P = P[question_order]
    if not (scan_order is None):
        P = P.loc[scan_order]
    if transpose:
        P = P.T
        title = 'Data Matrix $P^T$'
    else:
        title = 'Data Matrix $P$'
    sns.heatmap(data=P, cmap='viridis')
    ax.set_xticks([1,100,200,300,Nscans]);
    ax.set_xticklabels([1,100,200,300,Nscans]);
    ax.set_xlabel('Scans', fontsize=16)
    ax.set_ylabel('Questions', fontsize=16)
    ax.set_title(title, fontsize=18)
    if not (clusters_info is None):
        aux = pd.concat([P,clusters_info],axis=1)
        if not(scan_order is None):
            aux = aux.loc[scan_order]
        aux = aux.reset_index()
        aux['Cluster Transitions'] = aux['Cluster ID'].diff()
        aux_filtered = aux[aux['Cluster Transitions'] != 0]
        jump   = np.array(aux_filtered.index[1::])
        center = np.append(jump,Nscans)
        diff   = (center[1:] - center[:-1])/2
        center = center - np.append(center[0]/2, diff)
        aux_cl_labels = list(aux_filtered['Cluster Label'].values)
        for (i, loc) in enumerate(jump):
            ax.vlines(loc+1,0-0.5,P.shape[1], color='r', lw=2)
        for (i, loc) in enumerate(center):
            ax.text(loc, -0.5, aux_cl_labels[i], ha='left', rotation=45, fontsize=14)
        
    plt.close()
    return fig
   
def plot_W_heatmap(W,clusters_info=None,scan_order=None, cmap='Blues'):
    Nscans,Ndims = W.shape
    assert Ndims == 2
    fig, ax = plt.subplots(figsize=(16,0.5*Ndims))
    if clusters_info is None:
        if scan_order is None:
            xlabel = 'Unsorted Scans'
            sns.heatmap(W.T, cmap=cmap, ax=ax, cbar=True)
        else:
            xlabel = 'Sorted Scans'
            sns.heatmap((W[scan_order,:]).T, cmap=cmap, ax=ax, cbar=True)
    else:
        aux = pd.concat([W,clusters_info],axis=1)
        if not(scan_order is None):
            aux = aux.loc[scan_order]
        aux = aux.reset_index()
        aux['Cluster Transitions'] = aux['Cluster ID'].diff()
        aux_filtered = aux[aux['Cluster Transitions'] != 0]
        jump   = np.array(aux_filtered.index[1::])
        center = np.append(jump,Nscans)
        diff   = (center[1:] - center[:-1])/2
        center = center - np.append(center[0]/2, diff)
        sns.heatmap((aux[['Factor 1','Factor 2']]).T, cmap=cmap, ax=ax, cbar=True)
        aux_cl_labels = list(aux_filtered['Cluster Label'].values)
        for (i, loc) in enumerate(jump):
            ax.vlines(loc+1,0-0.5,W.shape[1], color='r')
        for (i, loc) in enumerate(center):
            ax.text(loc, -0.5, aux_cl_labels[i], ha='left', rotation=45, fontsize=14)
        xlabel='Sorted Scans'
        
    ax.set_yticks(np.arange(W.shape[1])+0.5)
    ax.set_yticklabels(['Factor {}'.format(i+1) for i in range(Ndims)], rotation=0, fontsize=14)
    ax.set_xticks([])
    ax.set_xlabel(xlabel, fontsize=16)
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    plt.close()
    return fig
   