"""
Functions for time warping.
"""

import numpy as np
from scipy.interpolate import interp1d
import tensorflow as tf


def warp(data, tau):
    """Applies a nonlinear temporal warping given by the indices (tau)

    Args:
        data: Tensor, shape [n_trials, canonical_length, neurons]

    Returns:
        warped_data: Tensor, shape [n_trials, n_timesteps, neurons]
    """
    # TODO add some asserts to make sure shapes are right, this will bite us otherwise
    data_shape = data.get_shape().as_list()
    n_neuron = data_shape[-1]

    # Make sure indices are in the right range.
    max_val = data_shape[1] - 1
    prev_idx = tf.clip_by_value(tf.cast(tf.floor(tau), tf.int32), 0, max_val)
    next_idx = tf.clip_by_value(prev_idx + 1, 0, max_val)

    def _warp_neuron(dat):
        prev_val = _get_values_at_coordinates(dat, prev_idx)
        next_val = _get_values_at_coordinates(dat, next_idx)
        val = prev_val + (tau - tf.cast(prev_idx, tf.float32)) * (next_val - prev_val)
        return val

    return tf.stack([_warp_neuron(data[:, :, i]) for i in range(n_neuron)], axis=2)


def _get_values_at_coordinates(array, coordinates):
    """Extract values at given coordinates.
    Args:
        array: 2D array
        coordinates: (x, y) ?
    """
    coordinates_flat = tf.reshape(coordinates, [-1])
    indices = tf.tile(tf.reshape(tf.range(tf.shape(coordinates)[0]), [-1, 1]),
                      [1, tf.shape(coordinates)[1]])
    indices_flat = tf.reshape(indices, [-1])
    vals = tf.gather_nd(array, tf.stack((indices_flat, coordinates_flat), axis=1))

    return tf.reshape(vals, tf.shape(coordinates))


def _invert_warp_indices(indices, n_timesteps, shared_length):
    """Compute the inverse of the warping functions."""
    xs = np.arange(n_timesteps)
    xs_eval = np.arange(shared_length)
    new_warps = []
    for t, trial_warp in enumerate(indices):
        f = interp1d(trial_warp, xs, bounds_error=False, fill_value=np.nan)
        new_warps.append(f(xs_eval))
    return np.array(new_warps).astype(np.float32)
