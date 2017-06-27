"""
TWPCA utilities
"""
import numpy as np
import tensorflow as tf

def printvars():
    """Prints all tensorflow variables
    """
    for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
        print(v.name, v.get_shape().as_list())

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

def inverse_softplus(y):
    """Inverse of the softplus function."""
    return np.log(np.exp(y) - 1 + 1e-8)

def softplus(x):
    """Softplus rectifier function."""
    np.log(np.exp(x) + 1)

def correlate_nanmean(x, y, **kwargs):
    """Wrapper around np.correlate that handles NaNs."""
    x_ = np.nan_to_num(x)
    y_ = np.nan_to_num(y)
    xcorr = np.correlate(x_, y_, **kwargs)
    return xcorr
