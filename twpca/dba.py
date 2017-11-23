import numpy as np
from fastdtw import fastdtw, dtw
from tqdm import trange
from scipy.spatial.distance import sqeuclidean

def dba_align(data, iterations, template=None):
    """Use DBA to align a collection of 1D time-series

    Args
    ----
    data : array-like
        single neuron raster across trials, shape = [n_trials, n_time]
    iterations : int
        number of iterations to run DBA
    template : array-like (optional)
        the initial template / dba average (if None, init with trial-average)
    center: bool (optional, default True)
        whether to center the warps when computing the template

    Returns
    -------
    template : np.ndarray
        Canonical timeseries identified by DBA
    warps : list of ndarray
        each element is a [n_index_matchings x 2] matrix holding the indices into
        the trial and into the template. The indices into the trial time series
        for trial k is `warps[k][:, 0]` and the indices into the template are
        `warps[k][:, 1]`. Note - the number of index matchings can be variable
        from trial to trial.
    cost_history : np.ndarray
        vector holding the cost function over
    """

    # objective history
    if template is None:
        template = np.mean(data, axis=0)

    cost_history = []
    K, T, N = data.shape
    templates = [template]

    # main optimization loop
    for itr in trange(iterations):
        cost_history.append(0.0)
        warps = []
        
        # compute statistics for next template
        for trial in data:
            dist, path = dtw(trial, template, dist=sqeuclidean)
            cost_history[-1] += dist
            warps.append(np.array(path))
        
        # Apply warps to data
        values = [list() for t in range(T)]
        # avg_warp = [list() for t in range(T)]
        for trial, idx in zip(data, warps):
            for i in idx:
                # avg_warp[i[1]].append(i[0])
                values[i[1]].append(trial[i[0]])

        template = np.array([np.mean(v, axis=0) for v in values])
        templates.append(template)

        # avg_warp = np.array([np.mean(i, axis=0) for i in avg_warp])
        # avg_warp = np.clip(np.round(avg_warp).astype(int), 0, T-1)

        # for i in warp:
        #     values[i[1]].append(trial[i[0]])   
        # warped_data.append(np.array([np.mean(v, axis=0) for v in values]))
        
    return templates, warps, cost_history

