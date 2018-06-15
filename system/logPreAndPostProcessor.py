import tensorflow as tf

from system.preAndPostProcessor import PreAndPostProcessor

__author__ = 'Andres'


class LogPreAndPostProcessor(PreAndPostProcessor):
    MAX_LOG = 2
    MIN_LOG = -2

    def _divideComplexIntoRealAndImag(self, complexTensor):
        magnitude = tf.abs(complexTensor)
        croppedMagnitude = tf.maximum(magnitude, 10**self.MIN_LOG)
        logMagnitude = tf.log(croppedMagnitude) / tf.log(tf.constant(10, dtype=croppedMagnitude.dtype))
        outputMagnitude = (logMagnitude - self.MIN_LOG) / (self.MAX_LOG - self.MIN_LOG)

        angle = tf.angle(complexTensor)
        real_part = outputMagnitude * tf.cos(angle)
        imag_part = outputMagnitude * tf.sin(angle)
        return tf.stack([real_part, imag_part], axis=-1, name='divideComplexIntoRealAndImag')

    def _inversePrediction(self, tensor):
        x = tensor[:, :, :, 0]
        y = tensor[:, :, :, 1]

        shifted_logmagnitude = tf.sqrt(tf.square(x) + tf.square(y))

        angle = tf.sign(tf.asin(y / (shifted_logmagnitude))) * tf.acos(x / (shifted_logmagnitude))

        recenteredMag = (shifted_logmagnitude * (self.MAX_LOG - self.MIN_LOG)) + self.MIN_LOG
        magnitude = (10 ** recenteredMag)

        real_part = magnitude * tf.cos(angle)
        imag_part = magnitude * tf.sin(angle)

        return tf.stack([real_part, imag_part], axis=-1, name='divideComplexIntoRealAndImag')
