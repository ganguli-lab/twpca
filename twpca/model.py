import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin # TODO: is this still valid?
from sklearn.decomposition import NMF, TruncatedSVD
from tqdm import trange

import tensorflow as tf
from . import warp, utils
from .regularizers import l2, curvature


class TWPCA(BaseEstimator, TransformerMixin):

    def __init__(self, data, n_components,
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
                 origin_idx=None):
        """Time-warped Principal Components Analysis

        Args:
            data: ndarray containing (trials x timepoints x neurons) data
            n_components: number of components to use
            shared_length (optional): length of each trial in the warped/shared space. If None,
                the length of the dataset is used (default: None)
            trial_regularizer (optional): regularization on the trial factors (default: l2(1e-6))
            time_regularizer (optional): regularization on the time factors (default: l2(1e-6))
            neuron_regularizer (optional): regularization on the neuron factors (default: l2(1e-6))
            nonneg (optional): whether latent factors are constrained to be nonnegative (default: False)
            fit_trial_factors (optional): whether or not to fit weights on each trial, in addition
                to the neuron and time factors (default: False)
            warptype: type of warps to allow ('nonlinear', 'affine', 'shift', or 'scale'). The
                default is 'nonlinear' which allows for full nonlinear warping
            warpinit: either 'identity', 'linear', 'shift', 'randn'
            warp_regularizer (optional): regularization on the warp function (default: curvature())
            origin_idx (optional): if not None, all warping functions are pinned (aligned) at this
                index. (default: None)
        """

        # store the data in numpy and tensorflow
        self.data = np.atleast_3d(data.astype(np.float32))
        # set NaNs to zero so tensorflow doesn't choke
        self._data = tf.constant(np.nan_to_num(self.data))

        # data dimensions
        self.n_trials, self.n_timepoints, self.n_neurons = self.data.shape
        self.n_components = n_components
        
        # mask out missing data
        self.mask = np.isfinite(self.data).astype(np.float32)
        self.num_datapoints = np.sum(self.mask)
        self._mask = tf.constant(self.mask)

        # Compute last non-nan index for each trial.
        # Note we find the first occurence of a non-nan in the reversed mask,
        # and then map that index back to the original mask
        rev_last_idx = np.argmax(np.all(self.mask, axis=-1)[:, ::-1], axis=1)
        self.last_idx = self.n_timepoints - rev_last_idx
        
        # dimension of latent temporal factors
        if shared_length is None:
            self.shared_length = self.n_timepoints
        else:
            self.shared_length = shared_length

        # model options
        self.fit_trial_factors = fit_trial_factors
        self.warptype = warptype
        self.warpinit = warpinit
        self.origin_idx = origin_idx
        self.nonneg = nonneg
        self.center_taus = center_taus

        # store regularization terms (these will match the variables in _params)
        self._regularizers = {
            'trial': trial_regularizer,
            'time': time_regularizer,
            'neuron': neuron_regularizer,
            'warp': warp_regularizer,
        }

        # store tensorflow variables, parameters and session
        self._train_vars = {} # tensorflow variables that are optimized
        self._params = {}     # model parameters (transformations applied to train_vars)

        # sets up tensorflow variables for warps
        self._initialize_warps()

        # create tensorflow variables for factor matrices
        self._train_vars['time'] = tf.get_variable('time_factors', shape=(self.n_timepoints, self.n_components))
        self._train_vars['neuron'] = tf.get_variable('neuron_factors', shape=(self.n_neurons, self.n_components))
        if self.fit_trial_factors:
            self._train_vars['trial'] = tf.get_variable('trial_factors', shape=(self.n_neurons, self.n_components))

        # if nonnegative model, transform factor matrices by softplus rectifier
        f = tf.nn.softplus if self.nonneg else tf.identity
        self._params['time'] = f(self._train_vars['time'])
        self._params['neuron'] = f(self._train_vars['neuron'])
        if self.fit_trial_factors:
            self._params['trial'] = f(self._train_vars['trial'])

        # compute warped time factors for each trial
        _tiled_fctr = tf.tile(tf.expand_dims(self._params['time'], [0]), [self.n_trials, 1, 1])
        self._warped_time_factors = warp.warp(_tiled_fctr, self._params['warp'])

        # reconstruct full tensor 
        if self.fit_trial_factors:
            # trial i, time j, factor k, neuron n
            self.pred = tf.einsum('ik,ijk,nk->ijn', self._params['trial'], self._warped_time_factors, self._params['neuron'])
        else:
            # trial i, time j, factor k, neuron n
            self.pred = tf.einsum('ijk,nk->ijn', self._warped_time_factors, self._params['neuron'])

        # objective function (note that nan values are zeroed out by self._mask)
        self._recon_cost = tf.reduce_sum(self._mask * (self.pred - self._data)**2) / self.num_datapoints
        self._objective = self._recon_cost + self._regularization

        # TODO: better session management
        self._sess = tf.Session()
        self.assign_warps()
        self.assign_factors()

    def assign_factors(self, transform_data=True):

        # apply inverse warps to data to get a better estimate of initial factors
        data = self.transform(self.data) if transform_data else self.data
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
            time_fctr = inverse_softplus(time_fctr)
            neuron_fctr = inverse_softplus(neuron_fctr)

        # initialize the factors
        assignment_ops = [tf.assign(self._train_vars['time'], time_fctr), 
                          tf.assign(self._train_vars['neuron'], neuron_fctr)]

        # intialize trial_factors by pseudoinverse of neuron factor
        if self.fit_trial_factors:
            # TODO - fix this when data is missing at random
            Bpinv = np.linalg.pinv(neuron_fctr)
            trial_fctr = np.empty((data.shape[0], n_components), dtype=np.float32)
            for k, trial in enumerate(data):
                t = last_idx[k] # last index before NaN
                trial_fctr[k] = np.diag(np.linalg.pinv(time_fctr[:t]).dot(trial[:t]).dot(Bpinv.T))
            assignment_ops += [tf.assign(self._train_vars['neuron'], trial_fctr)]

        # done initializing factors
        return self._sess.run(assignment_ops)

    def assign_warps(self, warps=None, normalize_warps=True):
        """Initialize the warping functions

        Args:
            warps (optional): numpy array (trials x shared_length) holding warping functions
            normalize_warps (optional): if True, normalize warps

        If warps is not specified, the warps are initialized by the self.warpinit method
        """

        # different warp intializations
        if warps is not None:
            # warps proved by user
            if normalize_warps:
                warps *= self.n_timepoints / np.max(warps)

            shift = warps[:, 0] - 1
            scale = np.ones(self.n_trials)
            dtau = np.hstack((np.ones((self.n_trials, 1)), np.diff(warps, axis=1)))
            tau = utils.inverse_softplus(np.maximum(0, dtau) * np.log(2.0))

        elif self.warpinit == 'linear':
            # warps linearly stretched to common trial length
            # (note: this initializes to identity if the trial lengths are the same)
            scale = np.max(self.last_idx) / np.array(self.last_idx)
            shift = np.zeros((self.n_trials,))
            tau = np.zeros((self.n_trials, self.n_timepoints))

        elif self.warpinit == 'shift':
            # use cross-correlation to initialize the shifts
            psth = np.nanmean(self.data, axis=0)
            shift = []
            for tidx, trial in enumerate(self.data):
                xcorr = np.zeros(self.n_timepoints)
                for n in range(self.n_neurons):
                    xcorr += utils.correlate_nanmean(psth[:, n], trial[:last_idx[tidx], n], mode='same')
                shift.append(np.argmax(xcorr) - (last_idx[tidx] / 2))
            shift = np.array(shift)
            scale = np.ones((self.n_trials,))
            tau = np.zeros((self.n_trials, self.n_timepoints))

        else:
            raise ValueError("Initialization method not recongnized: %s" % init)

        # assign the warps and return
        ops = []
        _var, var = (self.tau_shift, self.tau_scale, self.tau), (shift, scale, tau)
        for _v, v in zip(_var, var):
            ops += [tf.assign(_v, tf.constant(v, dtype=tf.float32))]
        return self._sess.run(ops)

    def fit(self, optimizer=tf.train.AdamOptimizer, niter=1000, lr=1e-3, sess=None, progressbar=True):
        """Fit the twPCA model

        Args:
            optimizer (optional): a tf.train.Optimizer class (default: AdamOptimizer)
            niter (optional): number of iterations to run the optimizer for (default: 1000)
            sess (optional): tensorflow session to use for running the computation. If None,
                then a new session is created. (default: None)
            progressbar (optional): whether to print a progressbar (default: True)
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

        # TODO: better session management
        if sess is not None:
            self._sess = sess

        # reset objective history
        self.obj_history = []

        # create train_op
        self._lr = tf.placeholder(tf.float32, name="learning_rate")
        self._opt = optimizer(self._lr)
        var_list = [v for k, v in self._train_vars.items() if k != 'warp'] + list(self._train_vars['warp'])
        self._train_op = self._opt.minimize(self._objective, var_list=var_list)

        # run the optimizer
        utils.initialize_new_vars(self._sess)
        iterator = trange if progressbar else range
        _ops = [self._objective, self._train_op]
        for i, l in zip(niter, lr):
            self.obj_history += [self._sess.run(_ops, feed_dict={self._lr: l})[0] for tt in iterator(i)]

        return self

    @property
    def params(self):
        """Returns a dictionary of factors and warps"""
        values = self._sess.run(list(self._params.values()))
        return dict(zip(self._params.keys(), values))

    def transform(self, data=None):
        """Transform the dataset from trial space into the shared space (de-jitters the raw data).

        Note: this uses the data that was used to initialize and fit the time parameters.

        Returns:
            [n_trials, shared_length, n_neurons] numpy array of data warped into shared space
        """
        if data is None:
            data = self._data
        elif isinstance(data, np.ndarray):
            data = tf.constant(data, dtype=tf.float32)
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
            raise ValueError('No model has been fit - must call TWPCA.fit() before TWPCA.predict().')

        if X is None:
            return self._sess.run(self.pred)

        elif isinstance(X, np.ndarray):
            # input is a (trial x time x neuron) dataset of unwarped data
            n_trials, n_timepoints, n_neurons = X.shape
            # grab the warped temporal factors
            if self.fit_trial_factors:
                warped_factors = self._sess.run(self._warped_time_factors)
                trial_factors = self._sess.run(self._params['trial'])
                warped_factors *= trial_factors[:, None, :] # broadcast multiply across trials
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
            pred = np.dot(neuron_factors.T, time_factors.T) # (neurons x trials-time)
            pred = pred.T.reshape(*X.shape) # (trials x time x neuron)

            return pred

    def _initialize_warps(self):

        self.tau = tf.get_variable('tau_params', shape=(self.n_trials, self.shared_length), dtype=tf.float32)
        self.tau_shift = tf.get_variable('tau_shift', shape=(self.n_trials,), dtype=tf.float32)
        self.tau_scale = tf.get_variable('tau_scale', shape=(self.n_trials,), dtype=tf.float32)

        _pos_tau = tf.nn.softplus(self.tau) / tf.log(2.0)
        _warp = self.tau_scale[:, None] * tf.cumsum(_pos_tau, 1) + self.tau_shift[:, None]

        # Force mean intercept to be zero and min slope to be one
        if self.center_taus:
            mean_intercept = tf.reduce_mean(_warp[:, 0])
            min_slope = tf.reduce_min(_warp[:, -1] - _warp[:, 0]) / (self.n_timepoints - 1)
            _warp = (_warp - mean_intercept) / min_slope

        # Force warps to be identical at origin idx
        if self.origin_idx is not None:
            pin = _warp - _warp[:, origin_idx][:, None] + origin_idx
            _warp = tf.clip_by_value(pin, 0, self.n_timepoints - 1)

        # store the warping function
        self._params['warp'] = _warp

        # store the inverse warping function
        _args = [_warp, self.n_timepoints, self.shared_length]
        self._inv_warp = tf.py_func(warp._invert_warp_indices, _args, tf.float32)

        # declare which parameters are trainable
        # Always include shift/scale with nonlinear transformation
        if self.warptype == 'nonlinear':
            self._train_vars['warp'] = [self.tau, self.tau_shift, self.tau_scale]
        elif self.warptype == 'affine':
            self._train_vars['warp'] = [self.tau_shift, self.tau_scale]
        elif self.warptype == 'shift':
            self._train_vars['warp'] = [self.tau_shift]
        elif self.warptype == 'scale':
            self._train_vars['warp'] = [self.tau_scale]
        else:
            valid_warptypes = ('nonlinear', 'affine', 'shift', 'scale')
            raise ValueError("Invalid warptype={}. Must be one of {}".format(warptype, valid_warptypes))

        return None

    @property
    def objective(self):
        return self._sess.run(self._objective)

    @property
    def recon_cost(self):
        return self._sess.run(self._recon_cost)

    @property
    def regularization(self):
        return self._sess.run(self._regularization)

    @property
    def _regularization(self):
        """Computes the total regularization cost"""
        return sum(self._regularizers[key](param) for key, param in self._params.items())
