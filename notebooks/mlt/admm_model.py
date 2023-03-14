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

def initialize(data, C, features, Wconstraint, Qconstraint):
    # init w, QT
    
    if C is not None:
        r = C.shape[1]
    else:
        r = 0
        
    W, QT = initialize_NMF(data, features+r, random=True)
    Q = QT.T
    
    W = W[:, :features]
    
    W[W < 0] = 0
    if Wconstraint[0] == True:
        W[W > Wconstraint[1]] = Wconstraint[1]

    Q[Q < 0] = 0
    if Qconstraint[0] == True:
        Q[Q > Qconstraint[1]] = Qconstraint[1]
        
    if C is not None:
        Z = W@(Q[:,:features].T) + C@(Q[:,features:].T)
    else:
        Z = W@(Q.T)
    Z[Z < 0] = 0
    Z[Z > np.max(data)] = np.max(data)

    aZ = np.zeros_like(data)

    return W, Q, Z, aZ

def obj_func(M, W, Q, C, beta, mask=None, regularizer=1, mode='model'):
    
    k = W.shape[1]
    if C is not None:
        r = C.shape[1]
    else:
        r = 0
    
    if mask is None:
        mask == np.ones_like(M)
        
#     betaW = beta
#     betaQ = beta
    betaW = beta*M.shape[1]
    betaQ = beta*M.shape[0]
        
    if mode == 'model':
        Wnorm = np.sum(np.linalg.norm(W, ord=regularizer, axis=1))
        Qnorm = np.sum(np.linalg.norm(Q, ord=regularizer, axis=1))
        if C is not None:
            return 0.5 * np.sum(mask*(M - W@(Q[:,:k].T) - C@(Q[:,k:].T) ) ** 2) + betaW*Wnorm + betaQ*Qnorm
        else:
            return 0.5 * np.sum(mask*(M - W@(Q.T)) ** 2) + betaW*Wnorm + betaQ*Qnorm
    if mode == 'valid':
        if C is not None:
            return 0.5 * np.sum(mask*(M - W@(Q[:,:k].T) - C@(Q[:,k:].T)) ** 2)
        else:
            return 0.5 * np.sum(mask*(M - W@(Q.T)) ** 2)
    
    


def fista_(A, B, tau, beta, rho, Y, mu, regularizer=1, maxit=200, X0=None, tol=1e-4):
    
    def fista_obj(X, A, B, Y, mu, beta, tau, rho, regularizer=1):
        gx = beta*np.linalg.norm(X, ord=regularizer, axis=0)
        fx = 0.5*(rho*np.linalg.norm(B - A@X, ord=2, axis=0)**2 + tau*np.linalg.norm(X - Y + mu/tau, ord=2, axis=0)**2)
        return gx + fx
    
    if X0 is None:
        X = np.zeros(A.shape[1], A.shape[0])
    else:
        X = X0.copy()
        
    prev_obj = fista_obj(X, A, B, Y, mu, beta, tau, rho, regularizer=regularizer)
    t = 2.0
    Op = 2*(rho*(A.T)@A + tau*np.eye(A.shape[1]))
    cross_term = 2*(rho*(A.T)@B + tau*(Y - mu/tau))
    L = np.linalg.norm(Op, ord=2)
#     L = np.real(spla(Op, k=1, which='LM', return_eigenvectors=False)[0])
    idx = np.arange(X.shape[1])
    for i in range(maxit):
        Xold = X[:,idx].copy()
        Gf = Op@X[:,idx] - cross_term[:,idx]
        X[:,idx] = soft_shrinkage(X[:,idx] - Gf / L, beta/L)
        t0 = t
        t = (1 + np.sqrt(1+ 4*t**2))/2
        X[:,idx] = X[:,idx] + ((t0 - 1) / t)*(X[:,idx] - Xold)
        obj = fista_obj(X[:,idx], A, B[:,idx], Y[:,idx], mu[:,idx], beta, tau, rho, regularizer=regularizer)
#         print('Fista_iter {}: energy = {}'.format(i, new_obj))
        idx = np.where(np.abs(prev_obj[idx] - obj)/np.abs(obj) > tol)[0]
    
        if idx.shape[0] == 0:
            return X
        else:
            prev_obj = obj
    return X

def soft_shrinkage(X, l):
    return np.sign(X) * np.maximum(np.abs(X) - l, 0.)



def admm_classo(X, A, B, beta, tau, rho, upper_constraint=(True, 1), regularizer=1,
                min_iter=1, max_iter=20, tol=1e-4):
    
    def Classo_obj(A, B, X, beta, regularizer=1):
        return 0.5*rho*np.sum((A@X - B)**2, axis=0) + beta*np.linalg.norm(X, ord=regularizer, axis=0)
    
    
    mu = np.zeros_like(X)
    Y = np.maximum(X + mu/tau, 0)
    if upper_constraint[0]:
        Y = np.minimum(Y, upper_constraint[1])
    
    prev_obj = Classo_obj(A, B, X, beta, regularizer=regularizer)
    idx = np.arange(X.shape[1])
    
    for i in range(max_iter):
        Xold = X[:,idx].copy()
        X[:,idx] = fista_(A, B[:,idx], tau, beta, rho, Y[:,idx], mu[:,idx], maxit=200, X0=Xold) # x0
        Y[:,idx] = np.maximum(X[:,idx] + mu[:,idx]/tau, 0)
        if upper_constraint[0]:
            Y[:,idx] = np.minimum(Y[:,idx], upper_constraint[1])
        mu[:,idx] += tau*(X[:,idx] - Y[:,idx])
        obj = Classo_obj(A, B[:,idx], X[:,idx], beta, regularizer=regularizer)
        
        idx = np.where(np.abs(prev_obj[idx] - obj)/np.abs(obj) > tol)[0]
        
        if i > min_iter:
            converged = True
            if idx.shape[0] > 0:
                converged = False
                
            if converged:
                return X.T, Y.T
    return X.T, Y.T



def admm_Tik(X, M, B, C, tau, rho, upper_constraint, min_iter=1, max_iter=20, tol=1e-4):
    
    def CTik_obj(A, B, C, X, rho):
        return 0.5*rho*np.sum((A*X - B)**2, axis=0) + 0.5*rho*np.sum((X - C)**2, axis=0)
    
    mu = np.zeros_like(X)
    Y = np.maximum(X + mu/tau, 0)
    if upper_constraint[0]:
        Y = np.minimum(Y, upper_constraint[1])
    
    prev_obj = CTik_obj(M, B, C, X, rho)
    idx = np.arange(X.shape[1])
        
    for i in range(max_iter):
        Xold = X[:,idx].copy()
        X[:,idx] = 1/(M[:,idx]**2 + tau + rho) * (M[:,idx]*B[:,idx] + tau*Y[:,idx] - mu[:,idx] + rho*C[:,idx])
        Y[:,idx] = np.maximum(X[:,idx] + mu[:,idx]/tau, 0)
        if upper_constraint[0]:
            Y[:,idx] = np.minimum(Y[:,idx], upper_constraint[1])
        mu[:,idx] += tau*(X[:,idx]-Y[:,idx])
        obj = CTik_obj(M[:,idx], B[:,idx], C[:,idx], X[:,idx], rho)

        idx = np.where(np.abs(prev_obj[idx] - obj)/np.abs(obj) > tol)[0]
        
        if i > min_iter:
            converged = True
            if idx.shape[0] > 0:
                converged = False
                
            if converged:
                return X, Y
            
    return X, Y



def admm(M, k, beta, C=None, rho=2.0, tau=2.0, mask=None,
         Wconstraint=(False, 1), Qconstraint=(False, 1),
         min_iter=10, max_iter=200, tol=1e-4,
         save_dir='./results/', save_every=(False, 20), outputfile='result'):

    if mask is None:
        mask = np.ones_like(M)
        
    solvers.options['show_progress'] = False
    
    # confounder dimensions
    if C is not None:
        r = C.shape[1]
    else:
        r = 0

    # initialization
    W, Q, Z, aZ = initialize(M, C, k, Wconstraint, Qconstraint)

    # initial distance value
    obj_history = [np.inf]
    
    # tqdm setting
    tqdm_iterator = trange(max_iter, desc='Loss', leave=True)
    
    # rescale regularizer coefficient based on size of W and Q
#     betaW = beta
#     betaQ = beta
    betaW = beta*M.shape[1]
    betaQ = beta*M.shape[0]

    # Main iteration
    for i in tqdm_iterator:
        
        # subproblem 1
        if C is not None:
            B = Z + aZ/rho - C@(Q[:,k:].T)
        else:
            B = Z + aZ/rho
        _, W = admm_classo(W.T, Q[:,:k], B.T, betaW, tau, rho, upper_constraint=Wconstraint)

        # subproblem 2
        if C is not None:
            A = np.column_stack((W, C))
        else:
            A = W
        _, Q = admm_classo(Q.T, A, Z + aZ/rho, betaQ, tau, rho, upper_constraint=Qconstraint)

        # subproblem 3
        if C is not None:
            R = np.column_stack((W, C))@(Q.T)
        else:
            R = W@(Q.T)
        _, Z = admm_Tik(Z, mask, mask*M, R - aZ/rho, tau, rho, (True, np.max(M)))

        # auxiliary varibles update        
        aZ += rho*(Z - R)
        
        # Iteration info
        obj_history.append(obj_func(M, W, Q, C, beta, mask, regularizer=1))
        tqdm_iterator.set_description("Loss: {:.4}".format(obj_history[-1]))
        tqdm_iterator.refresh()

        # Check convergence
        if i > min_iter:
            converged = True
            if np.abs(obj_history[-1] - obj_history[-2])/np.abs(obj_history[-1]) < tol:
                print('Algorithm converged with relative error < {}.'.format(tol))
            else:
                converged = False
            
            if converged:
                if save_every[0]:
                    output = {'W':W, 'Q':Q, 'C':C, 'obj_history':obj_history[1:]}
                    with open(os.path.join(save_dir, outputfile+'.pickle'), 'wb') as handle:
                        pickle.dump(output, handle, protocol=4)
                
                return W, Q, obj_history[1:]
            
        if save_every[0]:
            if i % save_every[1] == 0 and i > 0:
                output = {'W':W, 'Q':Q, 'C':C, 'obj_history':obj_history[1:]}
                with open(os.path.join(save_dir, outputfile+'_temp-{}'.format(i)+'.pickle'), 'wb') as handle:
                    pickle.dump(output, handle, protocol=4)

    print('Max iteration reached')
    if save_every[0]:
        output = {'W':W, 'Q':Q, 'C':C, 'obj_history':obj_history[1:]}
        with open(os.path.join(save_dir, outputfile+'.pickle'), 'wb') as handle:
            pickle.dump(output, handle, protocol=4)
                
    return W, Q, obj_history[1:]

def kfold_select(P, Krow, Kcol):
    block_list = []
    
    if (Krow > 1) & (Kcol > 1):
        krowf = KFold(n_splits=Krow, shuffle=True)
        kcolf = KFold(n_splits=Kcol, shuffle=True)
        for _, HOrow_idx in krowf.split(np.arange(P.shape[0]).astype('int')):
            for _, HOcol_idx in kcolf.split(np.arange(P.shape[1]).astype('int')):
                uHO = np.zeros(P.shape[0])
                uHO[HOrow_idx] = 1
                vHO = np.zeros(P.shape[1])
                vHO[HOcol_idx] = 1
                block = [uHO, vHO]
                block_list.append(block)
    if (Krow > 1) & (Kcol == 1):
        krowf = KFold(n_splits=Krow, shuffle=True)
        for _, HOrow_idx in krowf.split(np.arange(P.shape[0]).astype('int')):
            uHO = np.zeros(P.shape[0])
            uHO[HOrow_idx] = 1
            vHO = np.ones(P.shape[1])
            
            block = [uHO, vHO]
            block_list.append(block)
    if (Krow == 1) & (Kcol > 1):
        kcolf = KFold(n_splits=Kcol, shuffle=True)
        for _, HOcol_idx in kcolf.split(np.arange(P.shape[1]).astype('int')):
            uHO = np.ones(P.shape[0])
            vHO = np.zeros(P.shape[1])
            vHO[HOcol_idx] = 1
            block = [uHO, vHO]
            block_list.append(block)

    return block_list


    
def block_CV_mask(P, Krow, Kcol, J=10):
    
    block_list = kfold_select(P, Krow, Kcol)
    
    idx = np.arange(len(block_list))
    n = idx.shape[0]
    nc = int(np.ceil(n/J))
    if n % J != 0:
        warnings.warn('nblocks is not divisible!')
    np.random.shuffle(idx)
    
    mask_train_list = []
    mask_valid_list = []
    for j in range(J):
        block_idx = idx[j*nc:(j+1)*nc]
        
        mask_valid = np.zeros_like(P)
        for k in block_idx:
            block_vectors = block_list[k]
            print(block_vectors[0])
            mask_valid += np.outer(block_vectors[0], block_vectors[1])
        
        mask_valid[mask_valid > 1] == 1
        mask_train = 1 - mask_valid
        mask_train_list.append(mask_train)
        mask_valid_list.append(mask_valid)
        
    return mask_train_list, mask_valid_list
    
def random_CV_mask(P, J=10):
    idx = np.arange(P.shape[0]*P.shape[1])
    n = idx.shape[0]
    nc = int(np.ceil(n/J))
    np.random.shuffle(idx)
    mask_train_list = []
    mask_valid_list = []
    for j in range(J):
        mask_train = np.ones_like(P).reshape(-1)
        mask_train[idx[j*nc:(j+1)*nc]] = 0
        mask_valid = 1 - mask_train
        mask_train_list.append(mask_train.reshape(P.shape[0], P.shape[1]))
        mask_valid_list.append(mask_valid.reshape(P.shape[0], P.shape[1]))
    return mask_train_list, mask_valid_list
    
def embed_holdout(M, d, beta, C, rho, mask_train, mask_valid, data_mask, regularizer=1, outputfile='results', verbose=True):
    start = time.time()
    W, Q, _ = admm(M, d, beta, C=C, rho=rho, mask=mask_train*data_mask, Wconstraint=(True, 1), outputfile=outputfile)
    
    train_error = obj_func(M, W, Q, C, beta, mask = mask_train*data_mask, regularizer=regularizer, mode='valid')
    valid_error = obj_func(M, W, Q, C, beta, mask = mask_valid*data_mask, regularizer=regularizer, mode='valid')
    
    embedding_stat = [d, train_error, valid_error]
    end = time.time()
    if verbose:
        print('k={},b={},rho={}, Train-err={:.3f},Recon-err={:.3f},Time={:.1f}s'
              .format(d, beta, rho, train_error, valid_error, end-start))
    return embedding_stat



def nnQPsolver(H, f, upperbound, diagonal=False):
    n = H.shape[0]
    if diagonal:
        P = spdiag(matrix(H))
        q = matrix(f)
        if upperbound[0] == False:
            G = spdiag( matrix( -np.ones(n) ) )
            h = matrix( np.zeros(n) )
        elif upperbound[0] == True:
            G = spmatrix( np.hstack(( -np.ones(n), np.ones(n) )),
                         np.arange(2*n),
                         np.hstack(( np.arange(n), np.arange(n) )) )
            h = matrix( np.hstack((np.zeros(n), upperbound[1]*np.ones(n) )) )
        sol = solvers.qp(P,q,G,h)
        return np.array(sol['x']).flatten()
        
    else:
        P = matrix(H)
        q = matrix(f)
        if upperbound[0] == False:
            G = matrix( -np.eye(n) )
            h = matrix( np.zeros(n) )
        elif upperbound[0] == True:
            G = matrix( np.vstack((-np.eye(n), np.eye(n))) )
            h = matrix( np.hstack((np.zeros(n), upperbound[1]*np.ones(n) )) )
        sol = solvers.qp(P,q,G,h)
        return np.array(sol['x']).flatten()