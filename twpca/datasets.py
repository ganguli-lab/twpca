import os
import numpy as np
import scipy.io as spio
from scipy.ndimage import gaussian_filter1d

def jittered_neuron(t=None, feature=None, n_trial=61, jitter=1.0, gain=0.0, noise=0.05, seed=1234):
    """Generates a synthetic dataset of a single neuron with a jittered firing pattern.

    Parameters
    ----------
    t : array_like
        vector of within-trial timepoints
    feature : function
        produces a jittered instance of the feature (takes time shift as an input)
    n_trial : int
        number of trials
    jitter : float
        standard deviation of trial-to-trial shifts
    gain : float
        standard deviation of trial-to-trial changes in amplitude
    noise : float
        scale of additive gaussian noise
    seed : int
        seed for the random number generator

    Returns
    -------
    canonical_feature : array_like
        vector of firing rates on a trial with zero jitter
    aligned_data : array_like
        n_trial x n_time x 1 array of de-jittered noisy data
    jittered_data : array_like
        n_trial x n_time x 1 array of firing rates with jitter and noise
    """

    # default time base
    if t is None:
        t = np.linspace(-5, 5, 150)

    # default feature
    if feature is None:
        feature = lambda tau: np.exp(-(t-tau)**2)

    # noise matrix
    np.random.seed(seed)
    noise = noise*np.random.randn(n_trial, len(t))

    # generate jittered data
    gains = 1.0 + gain*np.random.randn(n_trial)
    shifts = jitter*np.random.randn(n_trial)
    jittered_data = np.array([g*feature(s) for g, s in zip(gains, shifts)]) + noise

    # generate aligned data
    aligned_data = np.array([g*feature(0) for g in gains]) + noise

    return feature(0), np.atleast_3d(aligned_data), np.atleast_3d(jittered_data)
