"""
Time-warped Principal Components Analysis (twPCA)
=================================================

Simultaneous alignment and dimensionality reduction of multivariate data
"""

__version__ = '0.0.1'

from .model import TWPCA
from . import regularizers
from . import utils
from . import warp

__all__ = [
    TWPCA, regularizers, utils, warp
]
