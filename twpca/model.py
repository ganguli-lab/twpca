import numpy as np
from sklearn.decomposition import TruncatedSVD, NMF
from sdtw import SoftDTW
from .dtw import dtw
from sdtw.distance import SquaredEuclidean
from scipy.spatial.distance import sqeuclidean
from tqdm import tqdm
from .soft_dtw import soft_barycenter
import deepdish as dd

class TWPCA(object):
    def __init__(self, n_components=1, smoothness=1, nonneg=False,
                 alpha=0.0, l1_ratio=0.0, verbose=True, hard_warp_simplicity=0):

        # model options
        self.n_components = n_components
        self.smoothness = smoothness
        self.nonneg = nonneg
        self.verbose = verbose
        self.hard_warp_simplicity = hard_warp_simplicity
        self.alpha = alpha
        self.l1_ratio = l1_ratio

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
            decomp = NMF(n_components=self.n_components, alpha=self.alpha, l1_ratio=self.l1_ratio)
        else:
            decomp = TruncatedSVD(n_components=self.n_components)
        self.V = decomp.fit_transform(matrix).reshape(K, T, self.n_components)
        self.U = np.transpose(decomp.components_)

        # Use soft dtw to align to a smooth template
        init = self.V[np.random.randint(K)] # initial template
        self._barycenter, self._optim_result = soft_barycenter(self.V, self.smoothness, verbose=self.verbose, **kwargs)

        # compute soft warps
        self._soft_warps = []
        for v in self.V:
            D = SquaredEuclidean(self._barycenter, v)
            sdtw = SoftDTW(D, gamma=self.smoothness)
            value = sdtw.compute()
            w = sdtw.grad()
            self._soft_warps.append(w / np.sum(w, axis=1, keepdims=True))

        # reset hard warps
        self._hard_warps = None

    def soft_transform(self, X):
        """Applies warping functions (aligns data)
        """
        return np.array([np.dot(w, trial) for w, trial in zip(self.soft_warps, X)])

    def inverse_soft_transform(self, X):
        """Applies inverse warping functions (un-aligns data)
        """
        return np.array([np.dot(w.T, trial) for w, trial in zip(self.soft_warps, X)])

    def hard_transform(self, X):
        """Applies warping functions (aligns data)
        """
        warped_data = []
        for trial, warp in zip(X, self.hard_warps):
            values = [list() for t in range(X.shape[1])]
            for i in warp:
                values[i[1]].append(trial[i[0]])
            warped_data.append(np.array([np.mean(v, axis=0) for v in values]))
        return np.array(warped_data)

    def hard_transform_spikes(self, X):
        """Applies warping functions to binary data
        """
        X = np.atleast_3d(X)
        warped_data = []
        for trial, _warp in zip(X, self.hard_warps):
            warp = np.array(_warp)
            warped_trial = np.zeros(trial.shape)
            for n, neuron in enumerate(trial.T):
                spikes = np.argwhere(neuron).ravel()
                for s in spikes:
                    t = np.round(np.mean(warp[warp[:,0]==s, 1])).astype(int)
                    warped_trial[t, n] = 1
            warped_data.append(warped_trial)
        return np.squeeze(warped_data)

    def save(self, fname):
        """Saves serialized version of model.
        """
        return dd.io.save(fname, self.__dict__)

    def load_from_dict(self, modeldict):
        """Loads previously saved model
        """
        for k, v in modeldict.items():
            setattr(self, k, v)
        return self        

    def load(self, fname):
        """Loads previously saved model. Overwrites data.
        """
        return self.load_from_dict(dd.io.load(fname))

    @property
    def trial_average(self):
        return np.dot(self.barycenter, self.U.T)

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
            self._hard_warps = self.compute_hard_warps(self.hard_warp_simplicity)
        return self._hard_warps

    def compute_hard_warps(self, simplicity):
        # override default simplicity parameter
        warps = []
        itr = tqdm(self.V, desc='Computing hard warps') if self.verbose else self.V
        for x in itr:
            warps.append(dtw(x, self._barycenter, sqeuclidean, simplicity))
        return warps
