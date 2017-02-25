"""Time-warped principal component analysis model"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
from tqdm import trange
import tensorflow as tf
from . import warp, utils
from .regularizers import l2, curvature

class TWPCA(BaseEstimator, TransformerMixin):
    def __init__(self, n_components, shared_length,
                 trial_regularizer=l2(1e-6),
                 time_regularizer=l2(1e-6),
                 neuron_regularizer=l2(1e-6),
                 fit_trial_factors=False,
                 warptype='nonlinear',
                 warpinit='zero',
                 warp_regularizer=curvature(),
                 origin_idx=None):
        """Time-warped Principal Components Analysis.

        Args:
            n_components: number of components to use
            shared_length: length of each trial in the warped/shared space
            trial_regularizer (optional): regularization on the trial factors (default: l2(1e-6))
            time_regularizer (optional): regularization on the time factors (default: l2(1e-6))
            neuron_regularizer (optional): regularization on the neuron factors (default: l2(1e-6))
            fit_trial_factors (optional): whether or not to fit weights on each trial, in addition
                to the neuron and time factors (default: False)
            warptype: type of warps to allow ('nonlinear', 'affine', 'shift', or 'scale'). The
                default is 'nonlinear' which allows for full nonlinear warping
            warpinit: either 'zero' or 'randn'
            warp_regularizer (optional): regularization on the warp function (default: curvature())
            origin_idx (optional): if not None, all warping functions are pinned (aligned) at this
                index. (default: None)
        """
        self.n_components = n_components
        self.shared_length = shared_length
        self.fit_trial_factors = fit_trial_factors
        self.warptype = warptype
        self.warpinit = warpinit
        self.origin_idx = origin_idx

        # store regularization terms (these will match the variables in _params)
        self._regularizers = {
            'trial': trial_regularizer,
            'time': time_regularizer,
            'neuron': neuron_regularizer,
            'warp': warp_regularizer,
        }

        # store tensorflow variables (parameters) and session
        self._params = {}
        self._sess = None

    def fit(self, X, optimizer=tf.train.AdamOptimizer, niter=1000, lr=1e-3, sess=None):
        """Fit the twPCA model

        Args:
            X: 3D numpy array with dimensions [n_trials, n_timepoints, n_neurons]
            optimizer (optional): a tf.train.Optimizer class (default: AdamOptimizer)
            niter (optional): number of iterations to run the optimizer for (default: 1000)
            sess (optional): tensorflow session to use for running the computation. If None,
                then a new session is created. (default: None)
        """

        # convert niter and lr to iterables if given as scalars
        if (not np.iterable(niter)) and (not np.iterable(lr)):
            niter = (niter,)
            lr = (lr,)

        # niter and lr must have the same number of elements
        elif np.iterable(niter) and np.iterable(lr):
            niter = list(niter)
            lr = list(lr)
            if len(niter) != len(lr):
                raise ValueError("niter and lr must have the same length.")
        else:
            raise ValueError("niter and lr must either be numbers or iterables of the same length.")

        # conversion to float32, asserts the array is finite and at least 3D
        np_X = np.atleast_3d(check_array(X, allow_nd=True, dtype=np.float32))
        self.X = tf.constant(np_X)

        # pull out dimensions
        n_trials, n_timesteps, n_neurons = np_X.shape

        # build the parameterized warping functions
        self._params['warp'], self._inv_warp, warp_vars = warp.generate_warps(n_trials,
            n_timesteps, self.shared_length, self.warptype, self.warpinit, self.origin_idx, data=np_X)

        # Initialize factor matrices
        if self.fit_trial_factors:
            trial_init, time_init, neuron_init = utils._compute_lowrank_factors(np_X, self.n_components)
            self._params['trial'] = tf.Variable(trial_init, name="trial_factors")
        else:
            time_init, neuron_init = utils._compute_lowrank_factors(np_X, self.n_components, modes=(1, 2))

        # create tensorflow variables for factor matrices
        self._params['time'] = tf.Variable(time_init, name="time_factors")
        self._params['neuron'] = tf.Variable(neuron_init, name="neuron_factors")

        # warped time factors warped for each trial
        warped_time_factors = warp.warp(tf.tile(tf.expand_dims(self._params['time'], [0]), [n_trials, 1, 1]), self._params['warp'])

        if self.fit_trial_factors:
            # trial i, time j, factor k, neuron n
            self.X_pred = tf.einsum('ik,ijk,nk->ijn', self._params['trial'], warped_time_factors, self._params['neuron'])
        else:
            # trial i, time j, factor k, neuron n
            self.X_pred = tf.einsum('ijk,nk->ijn', warped_time_factors, self._params['neuron'])

        # total objective
        self.recon_cost = tf.reduce_mean((self.X_pred - self.X)**2)
        self.objective = self.recon_cost + self.regularization
        self.obj_history = []

        # create a tensorflow session if necessary
        if sess is None:
            sess = tf.Session()
        self._sess = sess

        # create train_op
        self._lr = tf.placeholder(tf.float32, name="learning_rate")
        self._opt = optimizer(self._lr)
        var_list = [v for k, v in self._params.items() if k != 'warp'] + list(warp_vars)
        self._train_op = self._opt.minimize(self.objective, var_list=var_list)

        # initialize variables
        utils.initialize_new_vars(self._sess)

        # run the optimizer
        for train_args in zip(niter, lr):
            self.train(*train_args)

        return self

    def train(self, niter, lr, progressbar=True):
        """Partially fits the model."""

        if progressbar:
            iterator = trange
        else:
            iterator = range

        ops = [self.objective, self._train_op]
        self.obj_history += [self._sess.run(ops, feed_dict={self._lr: lr})[0]
                             for tt in iterator(niter)]

    @property
    def params(self):
        """Returns a dictionary of factors and warps"""
        values = self._sess.run(list(self._params.values()))
        return dict(zip(self._params.keys(), values))

    def transform(self, X=None):
        """Transform the dataset from trial space into the shared space (de-jitters the raw data).

        Note: this uses the data that was used to initialize and fit the time parameters.

        Returns:
            [n_trial, shared_length, n_neuron] Tensor of data warped into shared space
        """
        if X is None:
            X_tf = self.X
        elif isinstance(X, np.ndarray):
            X_tf = tf.constant(X, dtype=tf.float32)
        elif isinstance(X, tf.Tensor):
            X_tf = X
        else:
            raise ValueError("X must be a numpy array or tensorflow tensor")

        return self._sess.run(warp.warp(X_tf, self._inv_warp))

    @property
    def regularization(self):
        """Computes the total regularization cost"""
        return sum(self._regularizers[key](param) for key, param in self._params.items())
