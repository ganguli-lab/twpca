import numpy as np
from scipy.optimize import minimize
from sdtw import SoftDTW
from sdtw.distance import SquaredEuclidean

def soft_barycenter(X, gamma, method="L-BFGS-B", tol=1e-3,
                    max_iter=50, verbose=True):
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

    # initial barycenter
    init = X.mean(0)

    def f(Z):
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
        if verbose:
            print('\rFitting TWPCA. Objective after {} gradient calls: {}'.format(f.count, obj), end='', flush=True)
        return obj, G.ravel()

    # The function works with vectors so we need to vectorize barycenter_init.
    f.count = 0
    res = minimize(f, init.ravel(), method=method, jac=True,
                   tol=tol, options=dict(maxiter=max_iter, disp=False))

    # add new line to output
    if verbose:
        print('\rDone after {} iterations'.format(res.nit))

    results = {k: getattr(res, k) for k in ('fun', 'message', 'nfev', 'nit', 'success')}

    return res.x.reshape(*init.shape), results
