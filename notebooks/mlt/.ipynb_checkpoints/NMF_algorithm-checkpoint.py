import numpy as np
import pandas as pd

from sklearn.decomposition import NMF
from admm_model import admm
from mu_model import mu_algorithm

import seaborn as sns
import time
from tqdm import tqdm

from matplotlib import pyplot
from matplotlib.pyplot import *



class NMF_model:
    
    def __init__(self, data_matrix, data_mask, dimension,
                 ppmi_preprocess=False,
                 method='generic',
                 sparsity_parameter=1.0,
                 Wbound=(False, 1.0),
                 Qbound=(False, 1.0)):
        
        # Algorithm to decomposition data matrix M into M = W Q'
        
        # ===input===
        # data_matrix : the data matrix
        
        # data_mask : binary matrix with same shape as the data matrix, 0=data not avaialble, 1=data available
        #             if all data entries are available, simply set data_mask = np.ones(data_matrix)
        
        # dimension : dimension of the representation
        # ppmi_preprocess : whether to normalize the data matrix by positive pointwise mutual information
        # method : two algorithms are available, which are 'generic' and 'admm'. Only 'admm' works for the Wbound and Qbound
        # sparsity_parameter : hyperparameter to control sparsit, larger the sparser. Zero corresponds to no regularization
        # Wbound : tuple for setting an upper bound of the entries in W. E.g. (True, 3.0) represents constraining entries
        #          in W to be within [0.0, 3.0]
        # Qbound : similar to Wbound
        
        # ===usage===
        # the function 'decomposition' wraps the NMF algorithm.
        # In this version :
        # - the sparsity parameters are set to be equivalent for both W and Q.
        # - only sparsity regularization is included
        #
        # The 'BIC_dimension' is a simple function estimating the dimension of the representation using BIC.
        # For large data matrix, it is time consuming.
        # The input 'search_range' is an integer array including all possible dimensions for testing.
        # E.g. setting 'search_range = np.arange(3, 20)' will search for the optimal dimension between 3 and 20.
        # This is a simple alternative to cross-validation approach, which is more costly
        
        # ===demonstration===
        # See 'demo.py' for details
        
        self.data_matrix = data_matrix
        self.data_mask = data_mask
        
        self.ppmi_preprocess = ppmi_preprocess
        self.dimension = dimension
        self.method = method
        self.sparsity_parameter = sparsity_parameter
        self.Wbound = Wbound
        self.Qbound = Qbound
        
        if self.ppmi_preprocess:
            self.data_matrix = np.nan_to_num(self.data_matrix)
            self.ppmi()
        else:
            self.P = self.data_matrix.copy()
        
        
    def output_datafile(self):
        
        np.savez('./datafile.npz', P=self.data_matrix, data_mask=self.data_mask)
        
    def ppmi(self):
        
        marginal_row = self.data_matrix.sum(axis=1)
        marginal_col = self.data_matrix.sum(axis=0)
        total = marginal_col.sum()
        expected = np.outer(marginal_row, marginal_col) / total
        P = self.data_matrix / expected
        with np.errstate(divide='ignore'):
            np.log(P, out=P)
        P.clip(0.0, out=P)
        
        self.P = P
        
    def decomposition(self):
            
        if self.method == 'generic':
            
            if np.all(self.data_mask == 1):
            
                model = NMF(n_components=self.dimension, init='random', random_state=0, 
                            alpha_W=self.sparsity_parameter, 
                            alpha_H='same',
                            l1_ratio=1.0,
                            max_iter=500)

                W = model.fit_transform(self.P)
                Q = model.components_.T
                
            else:
                
                print('mu algorithm')
                W, Q = mu_algorithm(self.P,
                                    self.dimension,
                                    self.sparsity_parameter,
                                    self.data_mask, 
                                    max_iter=500)
            
            
        elif self.method == 'admm':
                
            W, Q, obj_trend = admm(self.P, self.dimension, self.sparsity_parameter,
                                             C=None, rho=3, tau=3,
                                             mask=self.data_mask,
                                             Wconstraint=self.Wbound,
                                             Qconstraint=self.Qbound,
                                             tol=1e-3, max_iter=200)
                
        else:
            
            print('Method not implemented')
            W = self.P.copy()
            Q = self.P.copy().T
        
        return W, Q
            
    
    def BIC_dimension(self, search_range=np.arange(3, 20), plot=True, update=True):
        
        BIC = []
        for k in search_range:
            self.dimension = k
            W, Q = self.decomposition()
            
            mismatch_loss = 2 * 0.5 * np.linalg.norm(self.data_mask*(self.P - W@(Q.T)), ord='fro') ** 2
            freedom_loss = np.log(self.P.shape[0])*(self.P.shape[1]*(k+1) - k*(k-1)/2)
           
            BIC.append( mismatch_loss + freedom_loss )

        if plot:
            fig, ax = pyplot.subplots(figsize=(16,3))
            ax.plot(search_range, BIC,  marker='o')
            ax.set_title('BIC, dimension detected: {}'.format(search_range[np.argmin(BIC)]))
            pyplot.show()
            
        if update:
            self.dimension= search_range[np.argmin(BIC)]
            
        return search_range[np.argmin(BIC)]
    
    
    def obj_func(self, mask, W, Q):
        return 0.5 * np.sum(mask*(self.P - np.matmul(W, Q.T)) ** 2)
    
        
    def embed_holdout(self, mask_train, mask_valid):
        start = time.time()

        mask_train *= self.data_mask
        mask_valid *= self.data_mask
        
        self.data_mask = mask_train

        W, Q = self.decomposition()

        train_error = self.obj_func(mask_train, W, Q)
        valid_error = self.obj_func(mask_valid, W, Q)

        embedding_stat = [self.dimension, self.sparsity_parameter, train_error, valid_error]
        end = time.time()
        return embedding_stat
    

                
                
                
        
            
            