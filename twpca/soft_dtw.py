# Author: Mathieu Blondel
# License: Simplified BSD

import numpy as np

from scipy.optimize import minimize

from sdtw import SoftDTW
from sdtw.distance import SquaredEuclidean


def soft_barycenter(X, gamma=1.0, weights=None, method="L-BFGS-B", tol=1e-1, max_iter=50, verbose=True):
    """
    Compute barycenter (time series averaging) using soft-DTW.

    Note: this is a lightly edited version of the `sdtw_barycenter` function
    from sdtw package.

    Parameters
    ----------
    X (ndarray) : time series (K x T x N)
    gamma (float) : smoothness parameter
    method (string): optimization method, passed to `scipy.optimize.minimize`.
    tol (float) : tolerance of optimization method
    max_iter (int) : maximum number of iterations.
    verbose (bool) : If True, display progress of optimization
    """

    # initial template (choose a trial at random)
    K = X.shape[0]
    init = np.mean(X, axis=0)

    def f(Z):
        # Compute objective value and grad at Z.
        f.count += 1
        Z = Z.reshape(*init.shape)
        m = Z.shape[0]
        G = np.zeros_like(Z)
        obj = 0
        for i in range(len(X)):
            D = SquaredEuclidean(Z, X[i])
            sdtw = SoftDTW(D, gamma=gamma)
            value = sdtw.compute()
            E = sdtw.grad()
            G += D.jacobian_product(E)
            obj += value
        obj = obj/len(X)
        if verbose:
            print('\riter: {},   objective: {}'.format(f.count, obj), end='')
        return obj, G.ravel()

    # use scipy.optimize
    f.count = 0
    res = minimize(f, init.ravel(), method=method, jac=True, tol=tol, options=dict(maxfun=max_iter, disp=False))

    # add new line to output
    if verbose:
        print('\rDone after {} iterations'.format(f.count))

    # optimization results to save
    results = {k: getattr(res, k) for k in ('fun', 'message', 'nfev', 'nit', 'success')}

    return results, res.x.reshape(*init.shape)
