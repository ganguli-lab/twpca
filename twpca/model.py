import numpy as np
from sklearn.decomposition import TruncatedSVD, NMF
from .warp import dba_align#, apply_warp

class TWPCA(object):
    def __init__(self, n_components, nonneg=False):

        # model options
        self.n_components = n_components
        self.nonneg = nonneg

        # attributes set by calling fit
        self.warping_funcs = None

        # TODO warps for each component

    def fit(self, data):
        """Fits warps and components.
        """

        # data dimensions
        #   K = number of time series
        #   T = length of each time series
        #   N = number of measurements per timestep
        K, T, N = data.shape

        # unfold data tensor into (T*K) x N matrix
        matrix = data.reshape(-1, N)

        # first do PCA 
        D = NMF if self.nonneg else TruncatedSVD
        decomp = D(n_components=self.n_components)
        
        # tensor of low-dimensional time series (K x T x n_components)
        self.V = decomp.fit_transform(matrix).reshape(K, T, n_components)
        
        # projection into low-dimensional space (N x n_components matrix)
        self.U = np.transpose(decomp.components_)

        # do time warping in low-dimensional space
        self.V_template, self.warping_funcs, self.optim_hist = dba_align(V)


    def transform(self, data):
        """Applies warps to data.
        """
        if self.warping_funcs is None:
            raise ValueError('Warps are not fit. Must call model.fit(...) before model.transform(...).')
        
        return warp(data, self.warping_funcs)



# derivative of tau(V) with respect to 