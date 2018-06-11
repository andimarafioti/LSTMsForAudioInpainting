import functools

import tensorflow as tf
from tensorflow.contrib.signal.python.ops import window_ops

from system.preAndPostProcessor import PreAndPostProcessor

__author__ = 'Andres'


class LogPreAndPostProcessor(PreAndPostProcessor):
    def _divideComplexIntoRealAndImag(self, complexTensor):
        magnitude = tf.abs(complexTensor)
        shiftedMagnitude = tf.nn.relu(magnitude - 1e-4) + 1e-4
        logMagnitude = tf.log(shiftedMagnitude) / tf.log(tf.constant(10, dtype=shiftedMagnitude.dtype))
        angle = tf.angle(complexTensor)
        real_part = logMagnitude * tf.cos(angle)
        imag_part = logMagnitude * tf.sin(angle)
        return tf.stack([real_part, imag_part], axis=-1, name='divideComplexIntoRealAndImag')
