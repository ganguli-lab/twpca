"""
Time-warped Principal Components Analysis (twPCA)
=================================================

Simultaneous alignment and dimensionality reduction of multivariate data
"""

__version__ = '0.0.3'

from .model import TWPCA
from . import regularizers
from . import utils
from . import warp
from . import datasets
from . import crossval

__all__ = ['TWPCA', 'regularizers', 'utils', 'warp', 'datasets', 'crossval']
