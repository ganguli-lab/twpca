"""
TWPCA cross-validation and hyperparameter search routines
"""
import numpy as np
import tensorflow as tf
import itertools
from tqdm import tqdm, trange
from .model import TWPCA
from .regularizers import curvature
from .utils import stable_rank

__all__ = ['cross_validate', 'hyperparam_search']

def k_fold_iter(N, K):
    """K-fold cross validation iterator

    Args:
        N: int, number of datapoitns
        K: int, number of folds
    """
    idx = np.random.permutation(N)
    for k in range(K):
        train_idx = idx[idx % K != k]
        test_idx = idx[idx % K == k]
        yield train_idx, test_idx

def leave_k_out_iter(N, K):
    """Leave K out cross validation iterator

    Args:
        N: int, number of datapoitns
        K: int, number to leave out in training set
    """
    idx = np.random.permutation(N)
    i = 0
    while i < N-1:
        r = range(i, i + K)
        i += K
        train_idx = np.setdiff1d(idx, r)
        test_idx = idx[r]
        yield train_idx, test_idx    

def cross_validate(model, data, method, K, max_fits=np.inf, seed=1234, **fit_kw):
    """Runs specified cross-validation method.

    Args:
        model: TWPCA model instance
        data: array-like (n_trials x n_timepoints x n_neurons)
        method: string specifying method {'kfold', 'leaveout'}
        K: int, crossvalidation parameter

    Keyword Args:
        max_fits: int, maximum number of partitions to train on
        seed: int, passed to numpy.random.seed()
        **fit_kw: additional keywords passed to model.fit

    Returns:
        results: dict, model parameters and metrics calculated for all sessions
    """

    # set random state
    np.random.seed(seed)

    # set up cross validation partitions
    if method == 'kfold':
        partitions = k_fold_iter(data.shape[2], K)
    elif method == 'leaveout':
        partitions = leave_k_out_iter(data.shape[2], K)
    else:
        raise ValueError('Cross-validation method not recognized.')

    nfits = 0
    results = []
    for train, test in partitions:
        # partition dataset
        traindata = data[:, :, train]
        testdata = data[:, :, test]

        # fit model to training set
        tf.reset_default_graph() # TODO (ben): better session management
        sess = tf.Session()
        model.fit(traindata, sess=sess, **fit_kw)

        # assess dimensionality of warped vs unwarped testdata
        warped_testdata = model.transform(testdata)
        warped_rank, unwarped_rank = [], []
        for n in range(testdata.shape[2]):
            warped_rank.append(stable_rank(warped_testdata[:, :, n]))
            unwarped_rank.append(stable_rank(testdata[:, :, n]))

        # evaluate test error
        X_pred = model.predict(testdata)
        test_error = np.mean((testdata - X_pred)**2)

        # compile results
        results.append(
            {'params': model.params,
             'warped_rank': np.array(warped_rank),
             'unwarped_rank': np.array(unwarped_rank),
             'test_idx': test,
             'train_idx': train,
             'train_error': model._sess.run(model.recon_cost),
             'test_error': test_error,
             'obj_history': model.obj_history
            }
        )

        # clean-up tensorflow
        sess.close()

        # terminate early if user is impatient
        nfits += 1
        if nfits >= max_fits:
            break

    return results

def hyperparam_search(data, n_components, warp_scales, time_scales,
                      warp_reg=None, time_reg=None,
                      crossval_method='kfold', K=5, max_fits=np.inf,
                      fit_kw=dict(lr=(1e-1, 1e-2), niter=(250, 500), progressbar=False),
                      **model_kw):
    """Performs cross-validation over number of components, warp regularization scale, and
    temporal regularization scale.

    Args:
        data: array-like, trials x time x neurons dataset
        n_components: sequence of ints, number of components in each model
        warp_scales: sequence of floats, scale of warp regularization in each model
        time_scales: sequence of floats, scale of time factor regularization in each model

    Usage:
        Hyperparameters are provided as sequences, which are combined sequentially to form
        each model. For example, the following code fits three models. The first model has 1
        component, a warp regularization scale of 1e-3, and a time regularization scale
        of 1e-2. The second model has 2 components, a stronger penalty scale on the warps
        (1e-2), and a stronger penalty scale on the temporal factors (1e-1). And so forth.
        ```
        n_components = [1, 2, 3]
        warp_scales = [1e-3, 1e-2, 1e-2]
        time_scales = [1e-2, 1e-1, 1e-1]
        hyperparam_search(data, n_components, warp_scales, time_scales, ...)
        ```

    Keywork Args:
        warp_reg: function, takes scalar and outputs a regularization term (in tensorflow) for
                    the warps. By default, `warp_reg = twpca.regularizers.curvature(s, power=1)`.
        time_reg: function, takes scalar and outputs a regularization term (in tensorflow) for
                    the time factors. By default,
                    `time_reg = twpca.regularizers.curvature(s, power=2, axis=0)`.
        crossval_method: str, specifies cross validation. One of {'kfold' (default), 'leavout'}.
        K: int, cross validation parameter. For example, K = 5 specifies 5-fold cross validation
                    when crossval_method = 'kfold', and K = 1 specifies leave-1-out validation
                    when crossval_method = 'leaveout'
        max_fits: int, number of fits per model. For example max_fits = 1 means that only a single
                  training and test set are evaluated (default: np.inf).
        fit_kw: dict, keyword arguments passed to model.fit
        **model_kw: additional keywords are passed to twpca.TWPCA(...)

    Returns:
        results: dict, contains parameters and statistics of fitted models
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
        _result = cross_validate(model, data, crossval_method, K, max_fits=max_fits, **fit_kw)

        results['n_components'].append(nc)
        results['warp_scale'].append(ws)
        results['time_scale'].append(ts)
        results['crossval_data'].append(_result)
        results['mean_test'].append(np.mean([r['test_error'] for r in _result]))
        results['mean_train'].append(np.mean([r['train_error'] for r in _result]))
        results['mean_dim_change'].append(np.mean([r['warped_rank']-r['unwarped_rank'] for r in _result]))

    for k in results.keys():
        results[k] = np.array(results[k])

    return results

def error_per_component(data, component_range,
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
