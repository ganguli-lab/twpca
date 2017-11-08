import warnings

import numpy as np
import tensorflow as tf
from sklearn.decomposition import NMF, TruncatedSVD
from tqdm import trange

from . import utils, warp
from .regularizers import curvature, l2


class TWPCA(object):
    def __init__(self, data, n_components,
                 sess=None,
                 shared_length=None,
                 trial_regularizer=l2(1e-6),
                 time_regularizer=l2(1e-6),
                 neuron_regularizer=l2(1e-6),
                 nonneg=False,
                 fit_trial_factors=False,
                 center_taus=False,
                 warptype='nonlinear',
                 warpinit='linear',
                 warp_regularizer=curvature(),
                 origin_idx=None,
                 warps=None,
                 optimizer=tf.train.AdamOptimizer):
        """Time-warped Principal Components Analysis

        Args:
            data: ndarray containing (trials x timepoints x neurons) data
            n_components: number of components to use

        Keyword Args (optional):
            sess : tensorflow Session (a new session is created if not provided)
            shared_length : int, length of each trial in the warped/shared space. If not
                            specified, defaults to n_timepoints in `data`.
            trial_regularizer : regularization on the trial factors (default: l2(1e-6))
            time_regularizer : regularization on the time factors (default: l2(1e-6))
            neuron_regularizer : regularization on the neuron factors (default: l2(1e-6))
            nonneg : bool, if True the factors are constrained to be nonnegative (default: False)
            fit_trial_factors : If True, fit weights on each trial, in addition to the neuron
                                and time factors (default: False)
            warptype : type of warps to allow ('fixed', 'nonlinear', 'affine', 'shift', or 'scale'). The
                default is 'nonlinear' which allows for full nonlinear warping.
            warpinit : type of warp initialization ('identity', 'linear', 'shift')
            warp_regularizer : regularization on the warp function (default: curvature())
            origin_idx : int, if specified all warping functions are pinned to start at this index.
            warps : numpy array (shared_length x n_trials), if provided, the warps are initialized to
                    these functions.
            optimizer : tf.train.Optimizer class, that creates the initial training operation
                        (default: AdamOptimizer)

        Note:
            All attributes that are tensorflow variables are preceded by an underscore, while python-
            accessible versions of these attributes lack the underscore. For example, `self.data`
            contains the numpy array passed in by the user, while `self._data` refers to the
            (constant) Tensor holding this data in the tensorflow graph.
        """

        # model options
        self.n_components = n_components
        self.fit_trial_factors = fit_trial_factors
        self.warptype = warptype
        self.warpinit = warpinit
        self.origin_idx = origin_idx
        self.nonneg = nonneg
        self.center_taus = center_taus

        # create tensorflow session
        self._sess = tf.Session() if sess is None else sess

        # store the data as a numpy array
        self.data = np.atleast_3d(data.astype(np.float32))

        # convert to a tensorflow tensor and set NaNs to zero
        self._data = tf.constant(np.nan_to_num(self.data))

        # data dimensions
        self.n_trials, self.n_timepoints, self.n_neurons = self.data.shape
        if self.n_components > self.n_neurons:
            raise ValueError('TWPCA does not support models with more components than neurons')

        # mask out missing data
        mask = np.isfinite(self.data).astype(np.float32)
        num_datapoints = np.sum(mask)
        _mask = tf.constant(mask)

        # Compute last non-nan index for each trial.
        # Note we find the first occurence of a non-nan in the reversed mask,
        # and then map that index back to the original mask
        rev_last_idx = np.argmax(np.all(mask, axis=-1)[:, ::-1], axis=1)
        self.last_idx = self.n_timepoints - rev_last_idx

        # dimension of latent temporal factors
        if shared_length is None:
            self.shared_length = self.n_timepoints
        else:
            self.shared_length = shared_length

        # get regularization terms (these will match the variables in _params)
        regularizers = {
            'trial': trial_regularizer,
            'time': time_regularizer,
            'neuron': neuron_regularizer,
            'warp': warp_regularizer,
        }

        # Create dictionaries to hold tensorflow variables, and model parameters. Various transformations (e.g. softplus
        # to enforce nonnegativity) are applied to the variables. The resulting parameters are typically all the user
        # needs to care about / interpret.
        self._vars, self._params = {}, {}

        # initialize the warping functions
        self.assign_warps(warps)

        # sets up tensorflow variables for warps
        _pos_tau = tf.nn.softplus(self._vars['tau']) / tf.log(2.0)
        _warp = tf.nn.softplus(self._vars['tau_scale'][:, None]) * tf.cumsum(_pos_tau, 1) + self._vars['tau_shift'][:, None]

        # Force mean intercept to be zero and min slope to be one
        if center_taus:
            mean_intercept = tf.reduce_mean(_warp[:, 0])
            min_slope = tf.reduce_min(_warp[:, -1] - _warp[:, 0]) / (self.n_timepoints - 1)
            _warp = (_warp - mean_intercept) / min_slope

        # Force warps to be identical at origin idx
        if origin_idx is not None:
            pin = _warp - _warp[:, origin_idx][:, None] + origin_idx
            _warp = tf.clip_by_value(pin, 0, self.n_timepoints - 1)

        # store the warping and inverse warping function
        self._params['warp'] = _warp
        _args = [_warp, self.n_timepoints, self.shared_length]
        self._inv_warp = tf.py_func(warp._invert_warp_indices, _args, tf.float32)

        # initialize the factor matrices
        self.assign_factors()

        # if nonnegative model, transform factor matrices by softplus rectifier
        f = tf.nn.softplus if self.nonneg else tf.identity
        self._params['time'] = f(self._vars['time'])
        self._params['neuron'] = f(self._vars['neuron'])
        if self.fit_trial_factors:
            self._params['trial'] = f(self._vars['trial'])

        # compute warped time factors for each trial
        _tiled_fctr = tf.tile(tf.expand_dims(self._params['time'], [0]), [self.n_trials, 1, 1])
        self._warped_time_factors = warp.warp(_tiled_fctr, self._params['warp'])

        # reconstruct full tensor
        if self.fit_trial_factors:
            # trial i, time j, factor k, neuron n
            self._pred = tf.einsum('ik,ijk,nk->ijn', self._params['trial'], self._warped_time_factors, self._params['neuron'])

        else:
            # trial i, time j, factor k, neuron n
            self._pred = tf.einsum('ijk,nk->ijn', self._warped_time_factors, self._params['neuron'])

        # objective function (note that nan values are zeroed out by _mask)
        self._regularization = tf.reduce_sum([regularizers[k](self._params[k])
                                              for k in self._params.keys()])
        self._recon_cost = tf.reduce_sum(_mask * (self._pred - self._data)**2) / num_datapoints
        self._objective = self._recon_cost + self._regularization

        # initialize optimizer
        self._learning_rate = tf.placeholder(tf.float32, shape=[])
        self.create_train_op(optimizer)
        utils.initialize_new_vars(self._sess)

    def create_train_op(self, optimizer):
        """Assign the training operation.

        Args:
            optimizer: tf.train.Optimizer instance
        """
        # declare which variables are trainable
        factor_names = set(self._vars.keys()) - set(['tau', 'tau_scale', 'tau_shift'])
        warp_names = set(['tau', 'tau_scale', 'tau_shift'])
        if self.warptype == 'nonlinear':
            pass  # all warp variables are trainable
        elif self.warptype == 'affine':
            warp_names.remove('tau')
        elif self.warptype == 'shift':
            warp_names -= set(['tau', 'tau_scale'])
        elif self.warptype == 'scale':
            warp_names -= set(['tau', 'tau_shift'])
        elif self.warptype == 'fixed':
            warp_names -= set(['tau', 'tau_scale', 'tau_shift'])
        else:
            valid_warptypes = ('nonlinear', 'affine', 'shift', 'scale', 'fixed')
            raise ValueError("Invalid warptype={}. Must be one of {}".format(self.warptype, valid_warptypes))
        warp_vars = [self._vars[k] for k in warp_names]
        factor_vars = [self._vars[k] for k in factor_names]
        self._opt = optimizer(self._learning_rate)
        self._train_factor_op = self._opt.minimize(self._objective, var_list=factor_vars)
        self._train_warp_op = self._opt.minimize(self._objective, var_list=warp_vars)
        self._train_op = self._opt.minimize(self._objective,
                var_list=factor_vars + warp_vars)
        self.obj_history = []
        utils.initialize_new_vars(self._sess)

    def assign_factors(self):
        """Assign the factor matrices by matrix/tensor decomposition on warped data.
        """

        # apply inverse warps to data to get a better estimate of initial factors
        data = self.transform(self.data)
        data = np.nan_to_num(data)

        # do matrix decomposition for initial factor matrices
        if self.n_neurons == 1:
            time_fctr = np.nanmean(data, axis=0)
            neuron_fctr = np.atleast_2d([1.0])

        else:
            # do matrix decomposition on trial-averaged data matrix
            DecompModel = NMF if self.nonneg else TruncatedSVD
            decomp_model = DecompModel(n_components=self.n_components)
            time_fctr = decomp_model.fit_transform(np.nanmean(data, axis=0))
            neuron_fctr = np.transpose(decomp_model.components_)

        # rescale factors to same length
        s_time = np.linalg.norm(time_fctr, axis=0)
        s_neuron = np.linalg.norm(neuron_fctr, axis=0)
        s = np.sqrt(s_time * s_neuron)
        time_fctr = (time_fctr * s / s_time)
        neuron_fctr = (neuron_fctr * s / s_neuron)

        # if necessary, apply inverse softplus
        if self.nonneg:
            time_fctr = utils.inverse_softplus(time_fctr)
            neuron_fctr = utils.inverse_softplus(neuron_fctr)

        # initialize trial_factors by pseudoinverse of neuron factor
        if self.fit_trial_factors:
            # TODO - fix this when data is missing at random
            Bpinv = np.linalg.pinv(neuron_fctr)
            trial_fctr = np.empty((data.shape[0], self.n_components), dtype=np.float32)
            for k, trial in enumerate(data):
                t = self.last_idx[k]    # last index before NaN
                trial_fctr[k] = np.diag(np.linalg.pinv(time_fctr[:t]).dot(trial[:t]).dot(Bpinv.T))

        # check if variables have been created yet. If so, overwrite them, otherwise create
        # them and store them in self._vars
        if 'time' in self._vars.keys():
            assignment_ops = [tf.assign(self._vars['time'], time_fctr),
                              tf.assign(self._vars['neuron'], neuron_fctr)]
            if self.fit_trial_factors:
                assignment_ops += [tf.assign(self._vars['neuron'], trial_fctr)]
            self._sess.run(assignment_ops)
        else:
            self._vars['time'] = tf.Variable(time_fctr, name='time_factors', dtype=tf.float32)
            self._vars['neuron'] = tf.Variable(neuron_fctr, name='neuron_factors', dtype=tf.float32)
            if self.fit_trial_factors:
                self._vars['trial'] = tf.Variable(trial_fctr, name='trial_factors', dtype=tf.float32)

    def assign_warps(self, warps, normalize_warps=False):
        """Assign values to the warping functions.

        Args:
            warps (optional): numpy array (trials x shared_length) holding warping functions
            normalize_warps (optional): whether to normalize user specified warps (default: False)

        If warps is not specified, the warps are initialized by the self.warpinit method
        """

        # different warp intializations
        if warps is not None:
            # warps proved by user
            shift = warps[:, 0] - 1
            scale = np.ones(self.n_trials)
            dt = np.maximum(0, np.diff(warps, axis=1))  # enforce monotonic increasing warps
            tau = np.hstack((np.ones((self.n_trials, 1)), dt))

        elif self.warpinit == 'identity':
            scale = np.ones(self.n_trials) * (self.shared_length / self.n_timepoints)
            shift = -np.ones(self.n_trials)  # zero index
            tau = np.ones((self.n_trials, self.n_timepoints))

        elif self.warpinit == 'linear':
            # warps linearly stretched to common trial length
            scale = np.max(self.last_idx) / np.array(self.last_idx)
            shift = -np.ones(self.n_trials)  # zero index
            tau = np.ones((self.n_trials, self.n_timepoints))

        elif self.warpinit == 'shift':
            # use cross-correlation to initialize the shifts
            psth = np.nanmean(self.data, axis=0)
            shift = []
            for tidx, trial in enumerate(self.data):
                xcorr = np.zeros(self.n_timepoints)
                for n in range(self.n_neurons):
                    xcorr += utils.correlate_nanmean(psth[:, n], trial[:self.last_idx[tidx], n], mode='same')
                shift.append(np.argmax(xcorr) - (self.last_idx[tidx] / 2))
            shift = np.array(shift)
            scale = np.ones(self.n_trials)
            tau = np.ones((self.n_trials, self.n_timepoints))

        else:
            _args = (self.warpinit, ('identity', 'linear', 'shift'))
            raise ValueError('Invalid warpinit={}. Must be one of {}'.format(*_args))

        # invert the softplus transform that is applied to scale
        scale = utils.inverse_softplus(scale)
        tau = utils.inverse_softplus(tau * np.log(2.0))

        # check if warps were already initialized. If so overwrite them, otherwise create them
        if 'warp' in self._params.keys():
            ops = []
            tau_vars = (self._vars['tau_shift'], self._vars['tau_scale'], self._vars['tau'])
            for _v, v in zip(tau_vars, (shift, scale, tau)):
                ops += [tf.assign(_v, tf.constant(v, dtype=tf.float32))]
            self._sess.run(ops)
        else:
            self._vars['tau'] = tf.Variable(tau, name='tau', dtype=tf.float32)
            self._vars['tau_shift'] = tf.Variable(shift, name='tau_shift', dtype=tf.float32)
            self._vars['tau_scale'] = tf.Variable(scale, name='tau_scale', dtype=tf.float32)
            utils.initialize_new_vars(self._sess)

    def fit(self, niter=1000, lr=1e-3, vars="both", progressbar=True):
        """Fit the twPCA model

        Args:
            niter (optional): number of iterations to run the optimizer for (default: 1000)
            lr (optional): float, learning rate for the optimizer (default: 1e-3)
            vars (optional): which variables to train, one of:
                "both": train both factors and time warps
                "warps": train time warps, but not factors
                "factors": train factors, but not time warps
            progressbar (optional): whether to print a progressbar (default: True)
        """
        # convert niter and lr to iterables if given as scalars
        if (not np.iterable(niter)) and (not np.iterable(lr)):
            niter, lr = (niter,), (lr,)
        elif np.iterable(niter) and np.iterable(lr):
            if len(niter) != len(lr):
                raise ValueError("niter and lr must have the same length.")
        else:
            raise ValueError("niter and lr must either be numbers or iterables of the same length.")
        # select which training op to run
        if vars == "both":
            train_op = self._train_op
        elif vars == "warps":
            train_op = self._train_warp_op
        elif vars == "factors":
            train_op = self._train_factor_op
        else:
            raise ValueError("vars must be one of 'both', 'warps', or 'factors' but got %s"%vars)
        ops = [self._objective, train_op]
        # run the optimizer
        iterator = trange if progressbar else range
        for l, n in zip(lr, niter):
            feed = {self._learning_rate: l}
            self.obj_history += [self._sess.run(ops, feed_dict=feed)[0] for _ in iterator(n)]
        return self

    def transform(self, data=None):
        """Transform the dataset from trial space into the shared space (de-jitters the raw data).

        Note: this uses the data that was used to initialize and fit the time parameters.

        Returns:
            [n_trials, shared_length, n_neurons] numpy array of data warped into shared space
        """
        if data is None:
            data = self._data
        elif isinstance(data, np.ndarray):
            data = tf.constant(np.atleast_3d(data), dtype=tf.float32)
        elif not isinstance(data, tf.Tensor):
            raise ValueError("X must be a numpy array or tensorflow tensor")

        return self._sess.run(warp.warp(data, self._inv_warp))

    def predict(self, X=None):
        """Return model prediction of activity on each trial.

        Args:
            X (optional) : 3D numpy array with shape [n_trials, n_timepoints, n_neurons]

        Note: If `X` is not provided, the prediction of the model on training data
              (i.e. provided to `model.fit` function) is returned. If a new `X` is
              provided then it is assumed these are held-out neurons; in this case,
              `X` should have the same n_trials and n_timepoints as the training data
              provided to `model.fit`. The temporal factors and warps are re-used, and
              the neuron factors are newly fit in a least-squares sense.

        Returns:
            X_pred : 3D numpy array with shape [n_trials, n_timepoints, n_neurons] holding
                     low-dimensional model prediction.
        """
        if self._sess is None:
            raise ValueError('No model has been fit - must call fit before predict.')

        if X is None:
            return self._sess.run(self._pred)

        elif isinstance(X, np.ndarray):
            # input is a (trial x time x neuron) dataset of unwarped data
            n_trials, n_timepoints, n_neurons = X.shape

            # grab the warped temporal factors
            if self.fit_trial_factors:
                warped_factors = self._sess.run(self._warped_time_factors)
                trial_factors = self._sess.run(self._params['trial'])
                warped_factors *= trial_factors[:, None, :]     # broadcast multiply across trials
            else:
                warped_factors = self._sess.run(self._warped_time_factors)

            # check input size
            if warped_factors.shape[0] != n_trials:
                raise ValueError('Data does not have the expected number of trials.')
            if warped_factors.shape[1] != n_timepoints:
                raise ValueError('Data does not have the expected number of timepoints.')

            # reshape the factors and data into matrices
            # time factors is (trial-time x components); X_unf is (trial-time x neurons)
            time_factors = warped_factors.reshape(-1, self.n_components)
            X_unf = X.reshape(-1, n_neurons)

            # mask nan values (only evaluate when all neurons are recorded)
            mask = np.all(np.isfinite(X_unf), axis=-1)

            # do a least-squares solve to fit the neuron factors
            neuron_factors = np.linalg.lstsq(time_factors[mask, :], X_unf[mask, :])[0]

            # reconstruct and reshape the predicted activity
            pred = np.dot(neuron_factors.T, time_factors.T)     # (neurons x trials-time)
            pred = pred.T.reshape(*X.shape)     # (trials x time x neuron)

            return pred

    def dump(self):
        """Serializes model variables"""
        return self._sess.run(self._vars)

    def load(self, new_vars):
        """Assigns model variables from numpy arrays"""
        self._sess.run([tf.assign(self._vars[k], v) for k, v in new_vars.items()])

    @property
    def shifts_and_scales(self):
        if self.warptype == 'nonlinear':
            warnings.warn("TWPCA was fit with warptype == 'nonlinear', so" +
                          "shifts will only be approximate. Consider fitting" +
                          "with warptype == 'affine', 'shift', or 'scale'.")
        _v = [self._vars['tau_shift'], self._vars['tau_scale']]
        shifts, scales = self._sess.run(_v)
        return shifts, utils.softplus(scales)

    @property
    def params(self):
        """Returns a dictionary of factors and warps"""
        return self._sess.run(self._params)

    @property
    def objective(self):
        """Computes the full objective function that's optimized."""
        return self._sess.run(self._objective)

    @property
    def recon_cost(self):
        """Computes the mean squared error of the model."""
        return self._sess.run(self._recon_cost)

    @property
    def regularization(self):
        """Computes the regularization penalty on the model."""
        return self._sess.run(self._regularization)

    @property
    def warped_time_factors(self):
        """Computes the time factors warped into clock space for each trial."""
        return self._sess.run(self._warped_time_factors)
