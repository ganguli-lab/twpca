"""Regularization penalties."""

import tensorflow as tf
from tensorflow.contrib.layers import l2_regularizer as l2

__all__ = ['curvature', 'l2']


def curvature(scale=1.0, power=1, axis=1):
    """Curvature penalty.

    Penalizes the average of the 2nd order finite differences of the array,
    raised to the given power.

    Args:
        scale: the strength of the penalty
        power: the power to apply to the difference vector (default: 1)
        axis: axis along which to apply the operator (default: 1)
    """

    def _regularizer_function(arr):
        ndim = len(arr.get_shape())

        # slices along the specified axis kwarg
        def axis_slice(start, stop):
            slices = [slice(None) for _ in range(ndim)]
            slices[axis] = slice(start, stop)
            return arr[slices]

        # computes second order differences along the given axis
        second_order_diff = 0.5 * (axis_slice(None, -2) + axis_slice(2, None)) - axis_slice(1, -1)

        # the curvature penalty is the second order differences raised to the given power
        curvature = (1. / power) * tf.abs(second_order_diff) ** power
        return scale * tf.reduce_mean(curvature)

    return _regularizer_function
