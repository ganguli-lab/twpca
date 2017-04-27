"""
TWPCA cross-validation and hyperparameter search routines
"""
import numpy as np
import tensorflow as tf
import itertools
from tqdm import tqdm
from .model import TWPCA
from .regularizers import curvature
from .utils import stable_rank

__all__ = ['cross_validate', 'hyperparam_search', 'err_per_component']

def k_fold(N, K):
    """K-fold cross validation iterator

    Args:
        N: int, number of datapoitns
        K: int, number of folds
    """
    idx = np.arange(N)
    for k in range(K):
        train_idx = idx[idx % K != k]
        test_idx = idx[idx % K == k]
        yield train_idx, test_idx

def leave_n_out(N, K):
    """Leave K out cross validation iterator

    Args:
        N: int, number of datapoitns
        K: int, number to leave out in training set
    """
    idx = np.arange(N)
    i = 0
    while i < N-1:
        r = range(i, i + n)
        i += n
        train_idx = np.setdiff1d(idx, r)
        test_idx = idx[r]
        yield train_idx, test_idx    

def cross_validate(model, data, method, K, max_crossval=np.inf, **fit_kw):
    """Runs specified cross-validation method.

    Args:
        model: TWPCA model instance
        data: array-like (n_trials x n_timepoints x n_neurons)
        method: string specifying method {'kfold', 'leaveout'}
        K: int, crossvalidation parameter

    Returns:
        results: dict, model parameters and metrics calculated for all sessions
    """

    if method == 'kfold':
        partitions = k_fold(data.shape[2], K)
    elif method == 'leaveout':
        partitions = leave_k_out(data.shape[2], K)
    else:
        raise ValueError('Cross-validation method not recognized.')

    nfits = 0
    results = []
    for train, test in partitions:
        # partition dataset
        traindata = np.atleast_3d(data[:, :, train])
        testdata = np.atleast_3d(data[:, :, test])

        # fit model to training set
        tf.reset_default_graph()
        sess = tf.Session()
        model.fit(traindata, sess=sess, **fit_kw)
        # TODO - warmstart fits?

        # assess dimensionality of warped vs unwarped testdata
        warped_testdata = model.transform(testdata)
        warped_rank, unwarped_rank = [], []
        for n in range(testdata.shape[2]):
            warped_rank.append(stable_rank(warped_testdata[:, :, n]))
            unwarped_rank.append(stable_rank(testdata[:, :, n]))

        # evaluate test error
        time_factors = sess.run(model._warped_time_factors).reshape(-1, model.n_components)
        _, resid, _, _ = np.linalg.lstsq(time_factors, testdata.reshape(-1, len(test)))

        # compile results
        results.append(
            {'params': model.params,
             'warped_rank': np.array(warped_rank),
             'unwarped_rank': np.array(unwarped_rank),
             'test_idx': test,
             'train_idx': train,
             'train_error': model.obj_history[-1],
             'test_error': np.sum(resid),
             'obj_history': model.obj_history
            }
        )

        # clean-up tensorflow
        sess.close()

        # terminate early if user is impatient
        nfits += 1
        if nfits >= max_crossval:
            break

    return results

def hyperparam_search(data, n_components, warp_scales, time_scales,
                      warp_reg=None, time_reg=None,
                      crossval_method='kfold', K=5, max_crossval=np.inf,
                      fit_kw=dict(lr=(1e-1, 1e-2), niter=(250, 500), progressbar=False),
                      **model_kw):
    """Performs cross-validation over number of components, warp regularization scale, and
    temporal regularization scale.
    """

    # defaults for warp and time regularization
    if warp_reg is None:
        warp_reg = lambda s: curvature(scale=s, power=1)
    if time_reg is None:
        time_reg = lambda s: curvature(scale=s, power=2, axis=0)

    # initialize results dict
    results = {
        'n_components': [],
        'warp_scale': [],
        'time_scale': [],
        'crossval_data': [],
        'mean_test': [],
        'mean_train': [],
        'mean_dim_change': []
    }

    # run cross-validation for all specified hyperparameters
    for nc, ws, ts in zip(tqdm(n_components), warp_scales, time_scales):

        model = TWPCA(nc, warp_regularizer=warp_reg(ws), time_regularizer=time_reg(ts), **model_kw)
        _result = cross_validate(model, data, crossval_method, K, max_crossval=max_crossval, **fit_kw)

        results['n_components'].append(nc)
        results['warp_scale'].append(ws)
        results['time_scale'].append(ts)
        results['crossval_data'].append(_result)
        results['mean_test'].append(np.mean([r['test_error'] for r in _result]))
        results['mean_train'].append(np.mean([r['train_error'] for r in _result]))
        results['mean_dim_change'].append(np.mean([r['warped_rank']-r['unwarped_rank'] for r in _result]))

    return results

def err_per_component(data, component_range,
                      fit_kw=dict(lr=(1e-1, 1e-2), niter=(250, 500), progressbar=False),
                      **model_args):
    """Compares vanilla PCA to twPCA searching over number of components
    """

    # basic attributes
    n_trials, n_timepoints, n_neurons = data.shape
    data_norm = np.linalg.norm(data.ravel())

    # compute error for trial-average PCA
    pca_rel_err = []
    u, s, v = np.linalg.svd(data.mean(axis=0), full_matrices=False)
    
    for rank in component_range:
        pred = np.dot(u[:, :rank] * s[:rank], v[:rank])[None, :, :]
        resid = data - np.tile(pred, (n_trials, 1, 1))
        pca_rel_err.append(np.linalg.norm(resid.ravel()) / data_norm)

    # compute error for twPCA
    twpca_rel_err = []
    for n_components in component_range:
        model = TWPCA(n_components, **model_args)

        tf.reset_default_graph()
        sess = tf.Session()
        model.fit(data, sess=sess, **fit_kw)

        resid = data - model._sess.run(model.X_pred)
        twpca_rel_err.append(np.linalg.norm(resid.ravel()) / data_norm)

    return pca_rel_err, twpca_rel_err
