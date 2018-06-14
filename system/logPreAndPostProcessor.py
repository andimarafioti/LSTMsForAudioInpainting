import tensorflow as tf

from system.preAndPostProcessor import PreAndPostProcessor

__author__ = 'Andres'


class LogPreAndPostProcessor(PreAndPostProcessor):
    def _divideComplexIntoRealAndImag(self, complexTensor):
        magnitude = tf.abs(complexTensor)
        croppedMagnitude = tf.maximum(magnitude, 1e-1)
        logMagnitude = tf.log(croppedMagnitude) / tf.log(tf.constant(10, dtype=croppedMagnitude.dtype))
        maxLogMag = 2
        minLogMag = -1
        outputMagnitude = (logMagnitude - minLogMag) / (maxLogMag - minLogMag)

        angle = tf.angle(complexTensor)
        real_part = outputMagnitude * tf.cos(angle)
        imag_part = outputMagnitude * tf.sin(angle)
        return tf.stack([real_part, imag_part], axis=-1, name='divideComplexIntoRealAndImag')

    def _inversePrediction(self, tensor):
        x = tensor[:, :, :, 0]
        y = tensor[:, :, :, 1]

        shifted_logmagnitude = tf.sqrt(tf.square(x) + tf.square(y))

        angle = tf.sign(tf.asin(y / (shifted_logmagnitude))) * tf.acos(x / (shifted_logmagnitude))

        maxLogMag = 2
        minLogMag = -1

        recenteredMag = (shifted_logmagnitude * (maxLogMag - minLogMag)) + minLogMag
        magnitude = (10 ** recenteredMag)

        real_part = magnitude * tf.cos(angle)
        imag_part = magnitude * tf.sin(angle)

        return tf.stack([real_part, imag_part], axis=-1, name='divideComplexIntoRealAndImag')
