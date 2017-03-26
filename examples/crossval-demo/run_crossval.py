"""
Runs hyperparameter search on some sythetic data
"""

# imports
import numpy as np
np.random.seed(1234)
from scipy.ndimage.filters import gaussian_filter1d
from twpca.datasets import jittered_population
from twpca.crossval import hyperparam_gridsearch

# get synthetic data
rates, spikes = jittered_population()
smooth_std = 1.0
smoothed_spikes = gaussian_filter1d(spikes, smooth_std, axis=1)

# save example figure of raw data
import matplotlib
matplotlib.use(‘Agg’)
import matplotlib.pyplot as plt
plt.figure(figsize=(10,3))
plt.subplot(121)
plt.imshow(rates[..., 0], aspect='auto', cmap=plt.cm.viridis); plt.colorbar()
plt.title('Firing rate for neuron 1')
plt.xlabel('Time')
plt.ylabel('Trial')
plt.subplot(122)
plt.imshow(spikes[..., 0], aspect='auto', cmap=plt.cm.viridis); plt.colorbar()
plt.title('Spikes for neuron 1')
plt.savefig('raw_data.png')
plt.close()

# Do hyperparameter serach by cross-validation
gridsearch_args = {
    'n_components': 1,
    'warp_penalties': np.logspace(-5, -1, 6),
    'time_penalties': np.logspace(-2, 2, 6),
    'fit_kw': {'niter': (250,), 'progressbar': False, 'lr': (0.1,)}
}
summary, results = hyperparam_gridsearch(smoothed_spikes, **gridsearch_args)

# save data and results
import pickle
pickle.dump({'rates': rates, 'spikes': spikes}, open("crossval_data.pickle", "wb"))
pickle.dump(summary, open("crossval_summary.pickle", "wb"))
pickle.dump(results, open("crossval_results.pickle", "wb"))
