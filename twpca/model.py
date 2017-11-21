import numpy as np
from sklearn.decomposition import TruncatedSVD, NMF
from dtw import fastdtw
from scipy.interpolate import interp1d
from tqdm import tqdm

class TWPCA(object):
    def __init__(self, n_components, share_warps=True, nonneg=False):

        # model options
        self.n_components = n_components
        self.share_warps = share_warps
        self.nonneg = nonneg

        # attributes set by calling fit
        self.warping_funcs = None

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
        self.V = decomp.fit_transform(matrix).reshape(K, T, self.n_components)
        
        # projection into low-dimensional space (N x n_components matrix)
        self.U = np.transpose(decomp.components_)

        # do time warping in low-dimensional space
        if self.share_warps:
            self.warping_funcs = _dtw_median(self.V) # warp all components together
        else:
            self.warping_funcs = np.array([_dtw_median(v) for v in self.V.T]) # warp each component separately


    def transform(self, data):
        """Applies warps to data.
        """
        if self.warping_funcs is None:
            raise ValueError('Warps are not fit. Must call model.fit(...) before model.transform(...).')
        
        return _warp(data, self.warping_funcs)

    def fit_transform(self, data):
        """Fits warps and applies them.
        """
        self.fit(data)
        return self.transform(data)

def _sqerr(x, y):
    """Squared error cost function.
    """
    return np.sum((x - y)**2)

def _warp(data, warping_funcs):
    """Applies warps
    """
    t = np.arange(data.shape[1])
    aligned_data = []
    for trial, warp in zip(data, warping_funcs):
        # f = interp1d(t, trial, axis=0)
        g = interp1d(np.arange(len(warp)), trial[warp], axis=0)
        aligned_data.append(g(np.linspace(0, len(warp)-1, len(t))))

    return np.array(aligned_data)

def _dtw_median(data):
    """Warps all trials to a template.
    """

    # choose template trial (closest to trial-average)
    err_per_trial = np.sum((data - np.mean(data, axis=0))**2, axis=(1,2))
    template = data[np.argmin(err_per_trial)]

    # fit warps
    iterator = tqdm
    warps = np.array([fastdtw(trial, template, dist=_sqerr)[-1][0] for trial in iterator(data)])

    # for iter in range(3):
    #     template = np.mean(_warp(data, warps), axis=0)
    #     warps = np.array([fastdtw(trial, template, dist=_sqerr)[-1][0] for trial in iterator(data)])

    return warps



