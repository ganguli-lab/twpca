import numpy as np
from tqdm import tqdm

def heldout_transform(model, data, features=None):
    """Leave-one-out crossvalidation for warping

    Parameters:
    -----------
        model (TWPCA object) : the model to fit
        data (ndarray) : time series data (trials x timepoints x features)
        features (iterable) : optional, if specified only compute transform for these features

    Returns:
    --------
        soft (ndarray) : soft transformed time series
        hard (ndarray) : hard transformed time series
    """

    # trials, timepoints, features
    K, T, N = data.shape

    # by default, compute transform for every feature
    if features is None:
        features = np.arange(N)

    # suppress printing
    was_verbose = model.verbose
    model.verbose = False

    # hold out each feature, and compute its transforms
    soft, hard = [], []
    for n in tqdm(features):
        # fit on training data
        trainset = list(set(range(N)) - {n})
        model.fit(data[:,:,trainset])
        # transform the held out feature
        soft.append(model.soft_transform(data[:,:,n]))
        hard.append(model.hard_transform(data[:,:,n]))

    # turn model printing back on
    if was_verbose:
        model.verbose = True

    soft = np.array(soft).transpose(1,2,0)
    hard = np.array(hard).transpose(1,2,0)
    return soft, hard
