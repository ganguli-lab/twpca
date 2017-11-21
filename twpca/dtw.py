import numpy as np
from dtw import fastdtw
from tqdm import trange

def dba_align(data, iterations=2,
              template=None, center=False):
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
        template = data.mean(axis=0)

    cost_history = []
    K, T, N = data.shape

    # main optimization loop
    for itr in trange(iterations):
        cost_history.append(0.0)
        counts = np.zeros((K, T), dtype=np.int32)
        values = np.zeros((K, T, N))
        warps = []
        # compute statistics for next template
        for tidx, trial in enumerate(data):
            dist, cost, acc, path = fastdtw(trial, template, dist = lambda x, y: np.sum((x-y)**2))
            cost_history[-1] += dist
            warps.append(np.array(path))
        # When computing the template, use a centered version of the warps
        # that forces the mean warp to be the identity function
        if center:
            template_warps = center_warps(warps, n_time)
        else:
            template_warps = warps
        # Apply warps to data
        for tidx, idx in enumerate(template_warps):
            for i in idx.T:
                counts[tidx, i[1]] += 1
                values[tidx, i[1]] += data[tidx, i[0]]
        # Add fudge factor in denominator to avoid divide by 0
        template = values.sum(0) / (counts.sum(0)[:,None] + 1e-9)

    return template, warps, cost_history
