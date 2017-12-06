from scipy.spatial.distance import cdist
import numpy as np
from numba import jit

def dtw(x, y, dist, g):
    if np.ndim(x)==1:
        x = x.reshape(-1,1)
    if np.ndim(y)==1:
        y = y.reshape(-1,1)
    r, c = len(x), len(y)
    D0 = np.zeros((r + 1, c + 1))
    D0[0, 1:] = np.inf
    D0[1:, 0] = np.inf
    D1 = D0[1:, 1:]
    D0[1:,1:] = cdist(x,y,dist)
    for i in range(r):
        for j in range(c):
            D1[i, j] += min(D0[i, j], D0[i, j+1], D0[i+1, j])
    return _traceback(D0, g)

@jit
def _traceback(D, g):
    i, j = np.array(D.shape) - 2
    pq = [(i, j)]
    while ((i > 0) or (j > 0)):
        a, b, c = D[i, j], D[i, j+1], D[i+1, j]
        tb = np.argmin((a, (1+g)*b, (1+g)*c))
        if (tb == 0):
            i -= 1
            j -= 1
        elif (tb == 1):
            i -= 1
        else: # (tb == 2):
            j -= 1
        pq = [(i, j)] + pq
    return np.array(pq)
