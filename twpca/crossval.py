"""
TWPCA cross-validation and hyperparameter search routines
"""
import numpy as np
import tensorflow as tf
import itertools
from .model import TWPCA
from .regularizers import curvature
from .utils import stable_rank

__all__ = ['cross_validate', 'hyperparam_gridsearch']

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

def cross_validate(model, data, method, K, max_fits=np.inf, **fit_kw):
    """???"""

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
        if nfits >= max_fits:
            break

    return results

def hyperparam_gridsearch(data, warp_penalties=(0.1,), time_penalties=(0.1,),
                          crossval_method='kfold', K=5, max_crossval_fits=np.inf,
                          fit_kw=dict(lr=(1e-1, 1e-2), niter=(250, 500)), **model_args):

    # TODO - make this more flexible to try other types of regularization
    warp_reg = lambda s: curvature(scale=s, power=1)
    time_reg = lambda s: curvature(scale=s, power=2)

    I, J = len(warp_penalties), len(time_penalties)
    results = np.empty((I, J), dtype=object)

    warp_penalties = np.tile(np.atleast_2d(warp_penalties), (J, 1)).T
    time_penalties = np.tile(np.atleast_2d(time_penalties), (I, 1))

    for i, j in itertools.product(range(I), range(J)):
        warp_penalty = warp_penalties[i, j]
        time_penalty = time_penalties[i, j]

        model = TWPCA(**model_args, warp_regularizer=warp_reg(warp_penalty), time_regularizer=time_reg(time_penalty))
        results[i, j] = cross_validate(model, data, crossval_method, K, max_fits=max_crossval_fits, **fit_kw)

    return results, warp_penalties, time_penalties
