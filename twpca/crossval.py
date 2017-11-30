import numpy as np
from tqdm import tqdm, trange
from copy import deepcopy

def heldout_fits(model, data):
    """Leave-one-out crossvalidation for warping

    Parameters:
    -----------
        model (TWPCA object) : the model to fit
        data (ndarray) : time series data (trials x timepoints x features)

    Returns:
    --------
        soft (ndarray) : soft transformed time series
        hard (ndarray) : hard transformed time series
    """

    # trials, timepoints, features
    K, T, N = data.shape

    # suppress printing
    was_verbose = model.verbose
    model.verbose = False

    # hold out each feature, and compute its transforms
    modeldicts = []
    for n in trange(N):
        # fit on training data
        trainset = list(set(range(N)) - {n})
        model.fit(data[:,:,trainset])
        # save results
        modeldicts.append(deepcopy(model.__dict__))

    return modeldicts


def heldout_transform(model, modeldicts, transform, data):

    # trials, timepoints, features
    data = np.atleast_3d(data)
    K, T, N = data.shape
    
    newdata = []
    for n, d in enumerate(tqdm(modeldicts)):
        model.load_from_dict(d)
        newdata.append(transform(data[:,:,n]))

    # transpose data to form neurons x time x trials
    return np.array(newdata).transpose(1,2,0)
