import numpy as np
from matplotlib import pyplot
from matplotlib.pyplot import *

import seaborn as sns
import copy

from scipy.spatial.distance import cosine

def simulation_Ex1(nrow, ncol, ndict, overlap, density=0.3,
                   upperbd=100, lowerbd=0,
                   noise=True, SNR=0.95, confounder=False):
    
    D = np.zeros((nrow, ndict))
    for k in range(ndict):
        for j in range(nrow):
            if (j >= (k)*( (nrow//ndict) )) and (j <= (nrow//ndict) + overlap + (k)*((nrow//ndict))):
#                 D[j,k] = 1 * np.random.uniform()
#                 D[j,k] = 1.0
                D[j,k] += np.random.uniform(0.5, 1.0)*np.random.binomial(1, p=0.9)
#                 D[j,k] += np.random.binomial(1, p=0.9)
#                 D[j,k] = np.random.binomial(1, p=0.9)
            else:
                D[j,k] = 0
                
    A = np.random.binomial(size=(ndict,ncol), n=1, p=density)
    A = 1.0*(A > 0)
    S = np.random.uniform(low=lowerbd, high=upperbd, size=(ndict,ncol))
    A = A*S
    
    D[D > 1] = 1
    
    if confounder:
        C = np.zeros((nrow, 2))
        C[:nrow//2,0] = 1
        C[nrow//2:,1] = 1
    
#         D[:,0] += C[:,0]*0.3*np.random.binomial(size=(nrow), n=1, p=0.99)
#         D[:,1] += C[:,1]*0.7*np.random.binomial(size=(nrow), n=1, p=0.99)

#         W2 = C
        W2 = np.vstack((C[:,0]*np.random.binomial(size=(nrow), n=1, p=0.9),
                        C[:,1]*np.random.binomial(size=(nrow), n=1, p=0.9))).T
    
#         A21 = np.ones(ncol)*np.random.uniform(low=lowerbd, high=upperbd*0.5)*np.random.binomial(size=(ncol), n=1, p=0.3)
#         A22 = np.ones(ncol)*np.random.uniform(low=lowerbd, high=upperbd*0.5)*np.random.binomial(size=(ncol), n=1, p=0.3)

        A21 = np.ones(ncol)*np.random.binomial(size=(ncol), n=1, p=0.3)
        A22 = np.ones(ncol)*np.random.binomial(size=(ncol), n=1, p=0.3)

        A2 = np.vstack((A21, A22))
    
        P = D@A + W2@A2
        
        true_W = np.hstack((D, W2))
        true_Q = np.vstack((A, A2)).T

    else:
        C = None
        P = D@A
        
        true_W = D
        true_Q = A
    
        
    P_clean = copy.deepcopy(P)
    
    if noise:
#         noise_matrix = np.random.choice((0, 1, 2), size=(nrow, ncol), p=[SNR, (1 - SNR) / 2., (1 - SNR) / 2.])
#         P[noise_matrix == 1] = 1
#         P[noise_matrix == 2] = 2

        noise_indicator = np.random.choice((0, 1), size=(nrow, ncol), p=[SNR, (1.0 - SNR)])
        noise_matrix = (noise_indicator == 1) * np.random.uniform(-np.max(P), np.max(P), size=(nrow, ncol))
        P +=noise_matrix
        P[P < 0] = 0
        P[P > np.max(P)] = np.max(P)

    fig, ax = pyplot.subplots(figsize=(12,3))
    sns.heatmap(P_clean.T, ax=ax, cmap="Blues", cbar=False)
    ax.set_title('Data matrix')
    ax.set_xticks([])
    ax.set_yticks([])
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 1.5, top - 0.5)    
    
    if noise:
        fig, ax = pyplot.subplots(figsize=(12,3))
        sns.heatmap(P.T, ax=ax, cmap="Blues", cbar=False)
        ax.set_title('Data matrix with noise added, SNR = {}'.format(SNR))
        ax.set_xticks([])
        ax.set_yticks([])
        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom + 1.5, top - 0.5)    

    fig = pyplot.figure(figsize=(14,3))
    ax_list = [fig.add_subplot(1,2,i+1) for i in range(2)]

    sns.heatmap(true_W.T, ax=ax_list[0], cmap="Blues", cbar=False)
    ax_list[0].set_title('True W, range:[{:.3f}, {:.3f}]'.format(np.min(true_W), np.max(true_W)))
    ax_list[0].set_xticks([])
    ax_list[0].set_yticks([])
    bottom, top = ax_list[0].get_ylim()
    ax_list[0].set_ylim(bottom + 1.5, top - 0.5)
    
    sns.heatmap(true_Q, ax=ax_list[1], cmap="Blues", cbar=False)
    ax_list[1].set_title('True Q, range:[{:.3f}, {:.3f}]'.format(np.min(true_Q), np.max(true_Q)))
    ax_list[1].set_xticks([])
    ax_list[1].set_yticks([])
    bottom, top = ax_list[1].get_ylim()
    ax_list[1].set_ylim(bottom + 1.5, top - 0.5)
 
    pyplot.tight_layout()
    pyplot.show()
    
        
    return true_W, true_Q, C, P_clean, P


def greedy_sort(W, true_W):
    bestmatch_ordering = np.array([])
    bestmatch_cor = np.array([])
    tempidx = np.arange(W.shape[1])
    temp_W = W
    match_W = np.zeros_like(W)
    
    for k in range(W.shape[1]):
        pc = np.array([])
        pcorrelation=np.array([])

        if k < true_W.shape[1]:

            for d in range(temp_W.shape[1]):
                pc = np.append(pc, 1 - cosine(true_W[:,k], temp_W[:,d]))
            sort_idx = np.argsort(-pc)
            match_W[:,k] = temp_W[:, sort_idx[0]]
            bestmatch_ordering = np.append(bestmatch_ordering, tempidx[sort_idx[0]])
            bestmatch_cor = np.append(bestmatch_cor, pc[sort_idx[0]])

            temp_W = np.delete(temp_W, sort_idx[0], axis=1)
            tempidx = np.delete(tempidx, sort_idx[0])

        else:
            if temp_W.shape[1] > 0:
                sort_idx = np.argsort(-np.max(temp_W,axis=0))
                temp_W = temp_W[:,sort_idx]
                match_W[:,k] = temp_W[:,0]
                temp_W = np.delete(temp_W, sort_idx[0], axis=1)
                
    return match_W, bestmatch_ordering, bestmatch_cor


def plot_decomposition(W, Q, true_W, true_Q):


    match_W, orders, corr = greedy_sort(W, true_W)

    fig = pyplot.figure(figsize=(14,4))
    ax_list = [fig.add_subplot(2,2,i+1) for i in range(4)]

    sns.heatmap(true_W.T, ax=ax_list[0], cmap="Blues", cbar=False)
    ax_list[0].set_title('True W, range:[{:.3f}, {:.3f}]'.format(np.min(true_W), np.max(true_W)))
    ax_list[0].set_xticks([])
    ax_list[0].set_yticks([])
    bottom, top = ax_list[0].get_ylim()
    ax_list[0].set_ylim(bottom + 1.5, top - 0.5)
    sns.heatmap(match_W.T, ax=ax_list[2], cmap="Blues", cbar=False)
    ax_list[2].set_title('W (up to permutation), range:[{:.3f}, {:.3f}]'.format(np.min(W), np.max(W)))
    ax_list[2].set_xticks([])
    ax_list[2].set_yticks([])
    bottom, top = ax_list[2].get_ylim()
    ax_list[2].set_ylim(bottom + 1.5, top - 0.5)
    sns.heatmap(true_Q, ax=ax_list[1], cmap="Blues", cbar=False)
    ax_list[1].set_title('True Q, range:[{:.3f}, {:.3f}]'.format(np.min(true_Q), np.max(true_Q)))
    ax_list[1].set_xticks([])
    ax_list[1].set_yticks([])
    bottom, top = ax_list[1].get_ylim()
    ax_list[1].set_ylim(bottom + 1.5, top - 0.5)
    sns.heatmap(Q[:, orders.astype('int')].T, ax=ax_list[3], cmap="Blues", cbar=False)
    ax_list[3].set_title('Q (up to permutation), range:[{:.3f}, {:.3f}]'.format(np.min(Q), np.max(Q)))
    ax_list[3].set_xticks([])
    ax_list[3].set_yticks([])
    bottom, top = ax_list[3].get_ylim()
    ax_list[3].set_ylim(bottom + 1.5, top - 0.5)
    pyplot.tight_layout()
    pyplot.show()





        