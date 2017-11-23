import numpy as np
from sklearn.decomposition import TruncatedSVD, NMF
from fastdtw import dtw
from sdtw import SoftDTW
from sdtw.barycenter import sdtw_barycenter
from sdtw.distance import SquaredEuclidean
from scipy.spatial.distance import sqeuclidean
from tqdm import tqdm
from .dba import dba_align

class TWPCA(object):
    def __init__(self, n_components, smoothness=1):

        # model options
        self.n_components = n_components
        self.smoothness = smoothness

        # attributes set by calling fit
        self.warping_funcs = None

    def fit(self, data, **kwargs):
        """Fits warps and components.
        """

        # data dimensions
        #   K = number of time series
        #   T = length of each time series
        #   N = number of measurements per timestep
        K, T, N = data.shape
        self.data = data

        # unfold data tensor into (T*K) x N matrix
        matrix = data.reshape(-1, N)

        # Do PCA
        #   V : low-dimensional time series (K x T x components)
        #   U : loadings across (N x components)
        self.decomp = TruncatedSVD(n_components=self.n_components)
        self.V = self.decomp.fit_transform(matrix).reshape(K, T, self.n_components)
        self.U = np.transpose(self.decomp.components_)

        # Use soft dtw to align to a smooth template
        init = self.V[np.random.randint(K)] # initial template
        self.template = sdtw_barycenter(self.V, init, gamma=self.smoothness, **kwargs)

        # # warp each trial to the template (in low-dimensional space)
        # self.warping_funcs = [np.array(dtw(v, self.template, dist=sqeuclidean)[1]) for v in tqdm(self.V)]
        # self.inverse_warps = [np.fliplr(w) for w in self.warping_funcs]

        self.warping_funcs = []
        for v in self.V:
            D = SquaredEuclidean(self.template, v)
            sdtw = SoftDTW(D, gamma=self.smoothness)
            value = sdtw.compute()
            self.warping_funcs.append(sdtw.grad())

    def reconstruct(self):
        """
        """
        if self.warping_funcs is None:
            raise ValueError('Warps are not fit. Must call model.fit(...) before model.transform(...).')
        UV = np.dot(self.template, self.U.T)
        return np.array([np.dot(w.T, UV) for w in self.warping_funcs])

    def transform(self):
        """Applies warps to data.
        """
        if self.warping_funcs is None:
            raise ValueError('Warps are not fit. Must call model.fit(...) before model.transform(...).') 
        return np.array([np.dot(w, trial) for w, trial in zip(self.warping_funcs, self.data)])

    def fit_transform(self, data):
        """Fits warps and applies them.
        """
        self.fit(data)
        return self.transform(data)

def _warp(data, warping_funcs):
    T = data.shape[1]
    warped_data = []
    for trial, warp in zip(data, warping_funcs):
        values = [list() for t in range(T)]
        for i in warp:
            values[i[1]].append(trial[i[0]])   
        warped_data.append(np.array([np.mean(v, axis=0) for v in values]))
    return np.array(warped_data)

# def _warp(data, warping_funcs):
#     """Applies warps
#     """
#     t = np.arange(data.shape[1])
#     aligned_data = []
#     for trial, warp in zip(data, warping_funcs):
#         # f = interp1d(t, trial, axis=0)
#         g = interp1d(np.arange(len(warp)), trial[warp], axis=0)
#         aligned_data.append(g(np.linspace(0, len(warp)-1, len(t))))

#     return np.array(aligned_data)

# def _dtw_median(data):
#     """Warps all trials to a template.
#     """

#     # choose template trial (closest to trial-average)
#     err_per_trial = np.sum((data - np.mean(data, axis=0))**2, axis=(1,2))
#     template = data[np.argmin(err_per_trial)]

#     # fit warps
#     iterator = tqdm
#     warps = np.array([fastdtw(trial, template, dist=_sqerr)[-1][0] for trial in iterator(data)])

#     # for iter in range(3):
#     #     template = np.mean(_warp(data, warps), axis=0)
#     #     warps = np.array([fastdtw(trial, template, dist=_sqerr)[-1][0] for trial in iterator(data)])

#     return warps



