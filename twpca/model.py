import numpy as np
from sklearn.decomposition import TruncatedSVD, NMF
from sdtw import SoftDTW
from fastdtw import fastdtw
from sdtw.distance import SquaredEuclidean
from scipy.spatial.distance import sqeuclidean
from tqdm import tqdm
from .soft_dtw import soft_barycenter
import deepdish as dd

class TWPCA(object):
    def __init__(self, n_components=1, smoothness=1, nonneg=False):

        # model options
        self.n_components = n_components
        self.smoothness = smoothness
        self.nonneg = nonneg

        # attributes set by calling fit
        self._soft_warps = None
        self._hard_warps = None
        self._barycenter = None

    def fit(self, data, **kwargs):
        """Fits warps and components.
        """

        # data dimensions
        #   K = number of time series
        #   T = length of each time series
        #   N = number of measurements per timestep
        K, T, N = data.shape

        # unfold data tensor into (T*K) x N matrix
        matrix = data.reshape(-1, N)

        # Do PCA
        #   V : low-dimensional time series (K x T x components)
        #   U : loadings across (N x components)
        if self.nonneg:
            decomp = NMF(n_components=self.n_components)
        else:
            decomp = TruncatedSVD(n_components=self.n_components)
        self.V = decomp.fit_transform(matrix).reshape(K, T, self.n_components)
        self.U = np.transpose(decomp.components_)

        # Use soft dtw to align to a smooth template
        init = self.V[np.random.randint(K)] # initial template
        self._optim_result, self._barycenter = soft_barycenter(self.V, gamma=self.smoothness, **kwargs)

        # compute soft warps
        self._soft_warps = []
        for v in self.V:
            D = SquaredEuclidean(self._barycenter, v)
            sdtw = SoftDTW(D, gamma=self.smoothness)
            value = sdtw.compute()
            w = sdtw.grad()
            self._soft_warps.append(w / np.sum(w, axis=1, keepdims=True))

    def inverse_soft_transform(self, X):
        """Applies inverse warping functions (misaligns data)
        """
        return np.array([np.dot(w.T, trial) for w, trial in zip(warps, X)])

    def soft_transform(self, X):
        """Applies warping functions (aligns data)
        """
        warps = self.soft_warps
        return np.array([np.dot(w, trial) for w, trial in zip(warps, X)])

    def inverse_hard_transform(self, X):
        """Applies inverse warping functions (misaligns data)
        """
        warped_data = []
        for trial, warp in zip(X, self.hard_warps):
            values = [list() for t in range(T)]
            for i in warp:
                values[i[1]].append(trial[i[0]])   
            warped_data.append(np.array([np.mean(v, axis=0) for v in values]))
        return np.array(warped_data)

    def hard_transform(self, X):
        """Applies warping functions (aligns data)
        """
        warped_data = []
        for trial, warp in zip(tqdm(X), self.hard_warps):
            values = [list() for t in range(X.shape[1])]
            for i in warp:
                values[i[1]].append(trial[i[0]])
            warped_data.append(np.array([np.mean(v, axis=0) for v in values]))
        return np.array(warped_data)

    def save(self, fname):
        """Saves serialized version of model.
        """
        return dd.io.save(fname, self.__dict__)

    def load(self, fname):
        """Loads previously saved model. Overwrites data.
        """
        modeldict = dd.io.load(fname)
        for k, v in modeldict.items():
            setattr(self, k, v)
        return self

    @property
    def trial_average(self):
        return np.dot(self._barycenter, self.U.T)

    @property
    def barycenter(self):
        if self._barycenter is None:
            raise ValueError('Warps are not fit. Must call model.fit(...) before accessing warps.')
        else:
            return self._barycenter

    @property
    def soft_warps(self):
        if self._soft_warps is None:
            raise ValueError('Warps are not fit. Must call model.fit(...) before accessing warps.')
        else:
            return self._soft_warps
    
    @property
    def hard_warps(self):
        if self._barycenter is None:
            raise ValueError('Warps are not fit. Must call model.fit(...) before accessing warps.')
        elif self._hard_warps is None:
            self._hard_warps = []
            for x in tqdm(self.V, desc='Computing hard warps', leave=True):
                self._hard_warps.append(fastdtw(x, self._barycenter, dist=sqeuclidean)[1])
        return self._hard_warps
