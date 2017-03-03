"""
Functions for time warping.
"""

import numpy as np
from scipy.interpolate import interp1d
import tensorflow as tf


def generate_warps(n_trials, n_timesteps, shared_length, warptype, init, origin_idx=None, data=None):
    """Generate parameters and warping function.

    Args:
        n_trials: number of trials
        n_timesteps: number of timesteps for each trial in clock space
        shared_length: number of timesteps in shared space
        warptype: type of warping, one of ('nonlinear', 'affine', 'shift', 'scale')
        init: initialization type ('zero', 'randn', or 'shfit')
        origin_idx: fix warps to be identity at this location
        data : optional, ndarray, shape = [n_trials x trial_length x n_neurons]
            only needed when using 'shift' initialization scheme
    Returns:
        warps: warping functions
        inv_warps: inverse of the warping functions
        params: list of parameters underlying the warping function
    """

    # initialize the delta_taus
    if init == 'randn':
        dtau_init = 0.5 * np.random.randn(n_trials, n_timesteps)
        init_scales = np.ones((n_trials,))
        init_shifts = np.zeros((n_trials,))
    elif init == 'zero':
        dtau_init = np.zeros((n_trials, n_timesteps))
        init_scales = np.ones((n_trials,))
        init_shifts = np.zeros((n_trials,))
    elif init == 'shift':
        if data is None:
            raise ValueError("To use shift initialization, you must pass data to generate_warps.")
        dtau_init = np.zeros((n_trials, n_timesteps))
        psth = np.mean(data, axis=0)
        init_shifts = []
        num_neurons = data.shape[-1]
        for trial in data:
            xcorr = np.zeros(n_timesteps)
            for n in range(num_neurons):
                xcorr += np.correlate(psth[:, n], trial[:, n], mode='same')
            init_shifts.append(np.argmax(xcorr) - (n_timesteps / 2))
        init_shifts = np.array(init_shifts)
        init_scales = np.ones((n_trials,))

    else:
        raise ValueError("Initialization method not recongnized: %s" % init)

    tau_params = tf.Variable(dtau_init.astype(np.float32), name="tau_params")

    # d_tau[k, t] is the positive change for trial k warping function at time t
    d_tau = tf.nn.softplus(tau_params) / tf.log(2.0)

    # tau(t) is the warping function, parameterized by tau_params
    tau_shift = tf.Variable(init_shifts.astype(np.float32), name="tau_shift")
    tau_scale = tf.Variable(init_scales.astype(np.float32), name="tau_scale")

    tau = tau_scale[:, None] * tf.cumsum(d_tau, 1) + tau_shift[:, None]

    # renormalize the warps so that they average to the identity line across trials.
    #   i.e. mean(tau[:,t]) = t
    linear_ramp = tf.range(0, shared_length, dtype=tf.float32)
    tau = tau - tf.reduce_mean(tau, axis=0, keep_dims=True) + linear_ramp[None, :]

    # Force warps to be identical at origin idx
    if origin_idx is not None:
        pin = tau - tau[:, origin_idx][:, None] + origin_idx
        tau = tf.clip_by_value(pin, 0, n_timesteps - 1)

    # compute the parameters for the inverse function
    tau_inv = tf.py_func(_invert_warp_indices, [tau, n_timesteps, shared_length], tf.float32)

    # declare which parameters are trainable
    # Always include shift/scale with nonlinear transformation
    if warptype == 'nonlinear':
        params = (tau_params, tau_shift, tau_scale)
    elif warptype == 'affine':
        params = (tau_shift, tau_scale)
    elif warptype == 'shift':
        params = (tau_shift,)
    elif warptype == 'scale':
        params = (tau_scale,)
    else:
        valid_warptypes = ('nonlinear', 'affine', 'shift', 'scale')
        "Invalid warptype={}. Must be one of {}".format(warptype, valid_warptypes)

    return tau, tau_inv, params


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

    # hack to ensure nd_gather has gradients for pre-1.0.0 tensorflow
    vals = tf.gather_nd(array, tf.stack((indices_flat, coordinates_flat), axis=1))

    return tf.reshape(vals, tf.shape(coordinates))


def _invert_warp_indices(indices, n_timesteps, shared_length):
    """Compute the inverse of the warping functions."""
    xs = np.arange(n_timesteps)
    xs_eval = np.arange(shared_length)
    new_warps = []
    for t, trial_warp in enumerate(indices):
        f = interp1d(trial_warp, xs, bounds_error=False, fill_value="extrapolate")
        new_warps.append(f(xs_eval))
    return np.array(new_warps).astype(np.float32)
