"""
TWPCA utilities
"""
import numpy as np
import tensorflow as tf

__all__ = ['get_uninitialized_vars', 'initialize_new_vars', 'unfold', 'stable_rank']


def get_uninitialized_vars(sess, var_list=None):
    """Gets a uninitialized variables in the current session

    Args:
        sess: a tensorflow session
        var_list (optional): list of variables to check. If None (default),
            then all global variables are checked

    Returns:
        uninitialized_vars: list of variables from var_list (or from the list
            of global variables if var_list was None) that have not yet
            been initialized
    """
    if var_list is None:
        var_list = tf.global_variables()
    is_init = sess.run(list(map(tf.is_variable_initialized, var_list)))
    return [v for v, init in zip(var_list, is_init) if not init]


def initialize_new_vars(sess, var_list=None):
    """Initializes any new, uninitialized variables in the session

    Args:
        sess: a tensorflow session
        var_list: list of variables to check (see `var_list` in
            `get_uninitialized_vars` for more information)
    """
    sess.run(tf.variables_initializer(get_uninitialized_vars(sess, var_list)))


def unfold(data, axis):
    """Unfold tensor along specified mode

    Args:
        data: numpy nd-array
        axis: int

    Returns:
        data_matrix: numpy 2d-array
    """
    return np.rollaxis(data, axis).reshape(data.shape[axis], -1)


def stable_rank(matrix):
    """Computes the stable rank of a matrix

    Args:
        matrix 2-D numpy array

    Returns:
        r: int

    Notes
    -----
    The stable rank is defined as the Frobenius norm divided by the square of the operator norm.
    That is, it is the sum of the squares of the singular values divided by the maximum singular
    value squared. The stable rank is upper bounded by the standard rank (number of strictly
    positive singular values).
    """
    if matrix.ndim != 2:
        raise ValueError('number of dimensions expected to be 2.')
    svals_squared = np.linalg.svd(matrix, full_matrices=False, compute_uv=False) ** 2
    return svals_squared.sum() / svals_squared.max()


def compute_lowrank_factors(data, n_components, fit_trial_factors, last_idx, scale=1.0):
    """Gets initial values for factor matrices by SVD on tensor unfoldings

    Args:
        data: array-like
        n_components: int
        fit_trial_factors: bool, whether to compute trial factors
        last_idx: nd-array, list of ints holding last index before trial end
        scale: scale neuron and time factors by this amount, default 1.0
    """
    # do svd on trial-averaged data matrix
    # TODO: use randomized/truncated SVD to speed this up
    u, s, v = np.linalg.svd(np.nanmean(data, axis=0), full_matrices=False)
    sqs = np.sqrt(s) * scale

    # set neuron and time factors to top singular vectors
    time_fctr = (u[:, :n_components] * sqs[:n_components]).astype(np.float32)
    neuron_fctr = (v[:n_components].T * sqs[:n_components]).astype(np.float32)

    if not fit_trial_factors:
        return None, time_fctr, neuron_fctr

    # If trial factors also need to be initialized, do a single step of alternating
    # least squares (see Kolda & Bader, 2009) for CP tensor decomposition.
    else:
        Bpinv = np.linalg.pinv(neuron_fctr)
        trial_fctr = np.empty((data.shape[0], n_components), dtype=np.float32)
        for k, trial in enumerate(data):
            t = last_idx[k] # last index before NaN
            trial_fctr[k] = np.diag(np.linalg.pinv(time_fctr[:t]).dot(trial[:t]).dot(Bpinv.T))
        return trial_fctr, time_fctr, neuron_fctr
