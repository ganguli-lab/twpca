import numpy as np
from functools import wraps
from sklearn.decomposition import NMF, TruncatedSVD
from tqdm import trange

import tensorflow as tf
from . import warp, utils
from .regularizers import l2, curvature


def tf_graph_wrapper(func):
    """Wraps a class method with a tf.Graph context manager"""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        with self._graph.as_default():
            return func(self, *args, **kwargs)
    return wrapper


def tf_init(func):
    """Wraps an __init__ function with its own session and graph"""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        self._graph = tf.Graph()
        self._sess = tf.Session(graph=self._graph)
        return tf_graph_wrapper(func)(self, *args, **kwargs)
    return wrapper


class TFSandbox:
    """Sandboxes subclass to live in a separate graph/session"""
    def __init_subclass__(cls):
        for name, value in cls.__dict__.items():

            # patch __init__
            if name == '__init__':
                setattr(cls, name, tf_init(value))

            # all class methods get wrapped
            elif callable(value):
                setattr(cls, name, tf_graph_wrapper(value))

            # _sess and _graph are reserved keywords
            elif name in ('_sess', '_graph'):
                raise ValueError('subclass cannot use reserved keywords _sess and _graph.')

        # patch the getattribute method
        setattr(cls, '__getattr__',
                lambda self, x: self.run(x) if isinstance(x, tf.Variable) else x)

    @tf_graph_wrapper
    def init_vars(self):
        return self.run(tf.global_variables_initializer())

    @tf_graph_wrapper
    def run(self, ops):
        return self._sess.run(ops)


class TWPCA(TFSandbox):

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
                 origin_idx=None,
                 warps=None,
                 normalize_warps=True,
                 optimizer=tf.train.AdamOptimizer):
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
            warpinit: either 'identity', 'linear', 'shift'
            warp_regularizer (optional): regularization on the warp function (default: curvature())
            origin_idx (optional): if not None, all warping functions are pinned (aligned) at this
                index. (default: None)
        """

        # set up tensorflow framework
        self._graph = tf.Graph()
        self._sess = tf.Session(graph=self._graph)

        # store the data in numpy and tensorflow
        self.data = np.atleast_3d(data.astype(np.float32))
        # set NaNs to zero so tensorflow doesn't choke
        with self._graph.as_default():
            self._data = tf.constant(np.nan_to_num(self.data))

        # data dimensions
        self.n_trials, self.n_timepoints, self.n_neurons = self.data.shape
        self.n_components = n_components
        
        # mask out missing data
        self.mask = np.isfinite(self.data).astype(np.float32)
        self.num_datapoints = np.sum(self.mask)
        with self._graph.as_default():
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

        # create all tensorflow variables
        with self._graph.as_default():
            self._train_vars = {} # tensorflow variables that are optimized
            # factor matrix parameters
            self._train_vars['time'] = tf.get_variable('time_factors', shape=(self.n_timepoints, self.n_components))
            self._train_vars['neuron'] = tf.get_variable('neuron_factors', shape=(self.n_neurons, self.n_components))
            if self.fit_trial_factors:
                self._train_vars['trial'] = tf.get_variable('trial_factors', shape=(self.n_neurons, self.n_components))
            # warp parameters
            self.tau = tf.get_variable('tau', shape=(self.n_trials, self.shared_length), dtype=tf.float32)
            self.tau_shift = tf.get_variable('tau_shift', shape=(self.n_trials,), dtype=tf.float32)
            self.tau_scale = tf.get_variable('tau_scale', shape=(self.n_trials,), dtype=tf.float32)
            # learning rate for optimizers
            self._lr = tf.placeholder(tf.float32, shape=[])

            # sets up tensorflow variables for warps
            _pos_tau = tf.nn.softplus(self.tau) / tf.log(2.0)
            _warp = self.tau_scale[:, None] * tf.cumsum(_pos_tau, 1) + self.tau_shift[:, None]

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
            self._params = {'warp': _warp}
            _args = [_warp, self.n_timepoints, self.shared_length]
            self._inv_warp = tf.py_func(warp._invert_warp_indices, _args, tf.float32)

            # declare which parameters are trainable
            # Always include shift/scale with nonlinear transformation
            if warptype == 'nonlinear':
                self._train_vars['warp'] = [self.tau, self.tau_shift, self.tau_scale]
            elif warptype == 'affine':
                self._train_vars['warp'] = [self.tau_shift, self.tau_scale]
            elif warptype == 'shift':
                self._train_vars['warp'] = [self.tau_shift]
            elif warptype == 'scale':
                self._train_vars['warp'] = [self.tau_scale]
            else:
                valid_warptypes = ('nonlinear', 'affine', 'shift', 'scale')
                raise ValueError("Invalid warptype={}. Must be one of {}".format(warptype, valid_warptypes))

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

            # initialize values for tensorflow variables
            self.assign_train_op(optimizer)
            self.assign_warps(warps, normalize_warps)
            self.assign_factors()

    def assign_train_op(self, optimizer):
        """Assign the training operation
        """
        self._opt = optimizer(self._lr)
        var_list = [v for k, v in self._train_vars.items() if k != 'warp'] + list(self._train_vars['warp'])
        self._train_op = self._opt.minimize(self._objective, var_list=var_list)
        utils.initialize_new_vars(self._sess)

    def assign_factors(self):
        """Assign the factor matrices by matrix/tensor decomposition on warped data
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

    def assign_warps(self, warps, normalize_warps=True):
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

        elif self.warpinit == 'identity':
            scale = np.ones(self.n_trials)
            shift = np.zeros(self.n_trials)
            tau = np.zeros((self.n_trials, self.n_timepoints))

        elif self.warpinit == 'linear':
            # warps linearly stretched to common trial length
            scale = np.max(self.last_idx) / np.array(self.last_idx)
            shift = np.zeros(self.n_trials)
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
            scale = np.ones(self.n_trials)
            tau = np.zeros((self.n_trials, self.n_timepoints))

        else:
            _args = (self.warpinit, ('identity', 'linear', 'shift'))
            raise ValueError('Invalid warpinit={}. Must be one of {}'.format(*_args))

        # assign the warps and return
        ops = []
        for _v, v in zip((self.tau_shift, self.tau_scale, self.tau), (shift, scale, tau)):
            ops += [tf.assign(_v, tf.constant(v, dtype=tf.float32))]
        return self._sess.run(ops)

    def fit(self, optimizer=None, niter=1000, lr=1e-3, sess=None, progressbar=True):
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
            niter, lr = (niter,), (lr,)
        elif np.iterable(niter) and np.iterable(lr):
            if len(niter) != len(lr):
                raise ValueError("niter and lr must have the same length.")
        else:
            raise ValueError("niter and lr must either be numbers or iterables of the same length.")

        # reset optimizer if set by user
        if optimizer is not None:
            self.assign_train_op(optimizer)

        # reset objective history
        self.obj_history = []

        # run the optimizer
        iterator = trange if progressbar else range
        _ops = [self._objective, self._train_op]
        for i, l in zip(niter, lr):
            self.obj_history += [self._sess.run(_ops, feed_dict={self._lr: l})[0] for tt in iterator(i)]

        return self.obj_history

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
