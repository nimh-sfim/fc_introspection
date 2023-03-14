import os
import pickle
import time
from datetime import datetime
from tqdm import tqdm, trange


# math imports
import numpy as np
from numpy.linalg import norm
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import scipy
from scipy.stats import rankdata
from scipy.spatial.distance import pdist, squareform, cosine, canberra, euclidean, jensenshannon, correlation

from math import sqrt

import sklearn
from sklearn.metrics import mean_squared_error
from sklearn.utils import extmath
from sklearn.model_selection import KFold

from cvxopt import matrix, spmatrix, solvers, sparse, spdiag
import warnings


def initialize_NMF(X, n_components, random):
    eps=1e-6
    if random:
        U, S, V = extmath.randomized_svd(X, n_components)
    else:
        U, S, V = extmath.randomized_svd(X, n_components, random_state=0)
    W, H = np.zeros(U.shape), np.zeros(V.shape)

    W[:, 0] = np.sqrt(S[0]) * np.abs(U[:, 0])
    H[0, :] = np.sqrt(S[0]) * np.abs(V[0, :])
    for j in range(1, n_components):
        x, y = U[:, j], V[j, :]
        # extract positive and negative parts of column vectors
        x_p, y_p = np.maximum(x, 0), np.maximum(y, 0)
        x_n, y_n = np.abs(np.minimum(x, 0)), np.abs(np.minimum(y, 0))
        # and their norms
        x_p_nrm, y_p_nrm = norm(x_p), norm(y_p)
        x_n_nrm, y_n_nrm = norm(x_n), norm(y_n)
        m_p, m_n = x_p_nrm * y_p_nrm, x_n_nrm * y_n_nrm
        # choose update
        if m_p > m_n:
            u = x_p / x_p_nrm
            v = y_p / y_p_nrm
            sigma = m_p
        else:
            u = x_n / x_n_nrm
            v = y_n / y_n_nrm
            sigma = m_n
        lbd = np.sqrt(S[j] * sigma)
        W[:, j] = lbd * u
        H[j, :] = lbd * v

    W[W < eps] = 0
    H[H < eps] = 0
    return W, H

def mu_algorithm(P, k, beta, mask=None, max_iter=200, tol=1e-4):

    EPSILON = np.finfo(np.float32).eps
    W, H = initialize_NMF(P, k, random=True)
    betaW = beta*P.shape[1]
    betaH = beta*P.shape[0]
    
    initial_err = np.linalg.norm(mask*(P - W@H), ord='fro')**2 + betaW*np.sum(W) + betaH*np.sum(H)
    prev_err = initial_err
    iteration = 0
    while iteration < max_iter and prev_err > tol:
        Wnom = (mask*P)@H.T
        Wdenom = (mask*(W@H))@H.T + betaW
        Wdenom[Wdenom==0] = EPSILON
        Hnom = (W.T)@(mask*P)
        Hdenom = (W.T)@(mask*(W@H)) + betaH
        Hdenom[Hdenom==0] = EPSILON
        W *= Wnom/Wdenom
        H *= Hnom/Hdenom
        if iteration % 10 == 0:
            err = np.linalg.norm(mask*(P - W@H), ord='fro')**2 + betaW*np.sum(W) + betaH*np.sum(H)
            err_tol = (prev_err - err)/initial_err
            prev_err = err
        iteration += 1
            
    return W, H.T