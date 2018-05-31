import tensorflow as tf
from system.lstmSystem import LSTMSystem
from utils.colorize import colorize


class RealImagContextLSTMSystem(LSTMSystem):
    def __init__(self, architecture, aPreProcessor, lstmParameters, name):
        super().__init__(architecture, aPreProcessor, lstmParameters, name)
        self._SNR = tf.reduce_mean(
            self._pavlovs_SNR(self._architecture.output(), self._architecture.target(), onAxis=[1, 2, 3]))

    def _spectrogramImageSummary(self):
        output = tf.abs(tf.transpose(self._architecture.output()[0]))
        target = tf.abs(tf.transpose(self._architecture.target()[0]))
        total = tf.abs(tf.transpose(tf.concat([self._architecture.input()[0, :, :, 0:2], self._architecture.output()[0],
                          self._architecture.input()[0, :, :, 2:4]], axis=0)))

        return tf.summary.merge([tf.summary.image("Original", [colorize(target)]),
                                tf.summary.image("Generated", [colorize(output)]),
                                tf.summary.image("Complete", [colorize(total)])])
