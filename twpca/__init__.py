"""
Time-warped Principal Components Analysis (twPCA)
=================================================

Simultaneous alignment and dimensionality reduction of multivariate data
"""

__version__ = '0.0.3'

from .model import TWPCA
from . import datasets
from . import warp

__all__ = ['TWPCA']
