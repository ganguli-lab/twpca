## :warning: Please use our newer code --- [Piecewise Linear Time Warping](https://github.com/ahwillia/affinewarp):

Our new work removes the assumption of low-dimensional dynamics, and uses a new optimization framework to avoid local minima in the warping function fitting routine. The [new code package](https://github.com/ahwillia/affinewarp) is also better optimized for speed, contains cross-validation routines, and has tools for working with spike data in continuous time.

## [DEPRECATED] Time warped principal components analysis (TWPCA)

[Ben Poole](https://cs.stanford.edu/~poole/) :beer:, [Alex H. Williams](http://alexhwilliams.info/) :studio_microphone:, [Niru Maheswaranathan](http://niru.org/) :soccer:

![image](https://img.shields.io/pypi/v/twpca.svg)

## Overview

### Installation

Again, **this package is deprecated**, so it should only be used as legacy software. But if you want to install it, you can do so manually:

```
git clone https://github.com/ganguli-lab/twpca
cd twpca
pip install -e .
```

### Description
Analysis of multi-trial neural data often relies on a strict alignment of neural activity to stimulus or behavioral events. However, activity on a single trial may be shifted and skewed in time due to differences in attentional state, biophysical kinetics, and other unobserved latent variables. This temporal variability can inflate the apparent dimensionality of data and obscure our ability to recover inherently simple, low-dimensional structure.

Here we present a novel method, time-warped PCA (twPCA), that simultaneously identifies temporal warps of individual trials and low-dimensional structure across neurons and time. Furthermore, we identify the temporal warping in a data-driven, unsupervised manner, removing the need for explicit knowledge of external variables responsible for temporal variability.

For more information, check out our [abstract](http://cs.stanford.edu/~poole/warptour.pdf) or [poster](http://cs.stanford.edu/~poole/twpca_poster.pdf).

*We also encourage you to look into our new package, [**affinewarp**](https://github.com/ahwillia/affinewarp), which was built with similar applications in mind.*

### Code
We provide code for twPCA in python (note: we use tensorflow as a backend for computation).

To apply twPCA to your own dataset, first install the code (`pip install twpca`) and load in your favorite dataset and shape it so that it is a 3D numpy array with dimensions (number of trials, number of timepoints per trial, number of neurons). For example, if you have a dataset with 100 trials each lasting 50 samples with 25 neurons, then your array should have shape (100, 50, 25).

Then, you can apply twPCA to your data by running `from twpca import TWPCA; model = TWPCA(data, n_components).fit()` where `n_components` is the number of low-rank factors you wish to fit and `data` is a 3D numpy as described above. A more thorough example is given below:

```python
from twpca import TWPCA
from twpca.datasets import jittered_neuron

# generates a dataset consisting of a single feature that is jittered on every trial.
# This helper function returns the raw feature, as well as the aligned (ground truth)
# data and the observed (jittered) data.
feature, aligned_data, raw_data = jittered_neuron()

# applies TWPCA to your dataset with the given number of components (this follows the
# scikit-learn fit/trasnform API)
n_components = 1
model = TWPCA(raw_data, n_components).fit()

# the model object now contains the low-rank factors
time_factors = model.params['time']         # compare this to the ground truth feature
neuron_factors = model.params['neuron']     # in this single-neuron example, this will be a scalar

# you can use the model object to align data (compare this to the aligned_data from above)
estimated_aligned_data = model.transform()
```

We have provided a more thorough [demo notebook](notebooks/demo.ipynb) demonstrating the application of tWPCA to a synthetic dataset.

## Further detail

### Motivation
Performing dimensionality reduction on misaligned time series produces illusory complexity. For example, the figure below shows that a dataset consisting of a single feature jittered across trials (red data) has illusory complexity (as the spectrum of singular values decays slowly).

![image](https://cloud.githubusercontent.com/assets/636625/23191412/430db774-f852-11e6-8176-8ea55d772d87.png)

### The twPCA model
To address this problem for a sequence of multi-dimensional time-series we simultaneously fit a latent factor model (e.g. a matrix decomposition), and time warping functions to align the latent factors to each measured time series. Each trial is modeled as a low-rank matrix where the neuron factors are fixed (gray box below) while the time factors vary from trial to trial by warping a canonical temporal factor differently on each trial.

![image](https://cloud.githubusercontent.com/assets/636625/23193786/866b0910-f85f-11e6-9987-948c8600c5ea.png)
