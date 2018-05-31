import tensorflow as tf
from system.lstmSystem import LSTMSystem
from utils.colorize import colorize


class RealImagContextLSTMSystem(LSTMSystem):
    def __init__(self, architecture, aPreProcessor, lstmParameters, name):
        super().__init__(architecture, aPreProcessor, lstmParameters, name)
        self._SNR = tf.reduce_mean(
            self._pavlovs_SNR(self._architecture.output(), self._architecture.target(), onAxis=[1, 2, 3]))

    def _spectrogramImageSummary(self):
        complexOutput = self._architecture.output()[0]
        outputSpectrogram = tf.sqrt(tf.square(complexOutput[:, :, 0]) + tf.square(complexOutput[:, :, 1]))

        complexTarget = self._architecture.target()[0]
        targetSpectrogram = tf.sqrt(tf.square(complexTarget[:, :, 0]) + tf.square(complexTarget[:, :, 1]))

        complexLeft = self._architecture.input()[0, :, :, 0:2]
        leftSpectrogram = tf.sqrt(tf.square(complexLeft[:, :, 0]) + tf.square(complexLeft[:, :, 1]))

        complexRight = self._architecture.input()[0, :, :, 2:4]
        rightSpectrogram = tf.sqrt(tf.square(complexRight[:, :, 0]) + tf.square(complexRight[:, :, 1]))

        totalSpectrogram = tf.transpose(tf.concat([leftSpectrogram, outputSpectrogram,
                                                   rightSpectrogram], axis=0))

        return tf.summary.merge([tf.summary.image("Original", [colorize(tf.transpose(targetSpectrogram))]),
                                tf.summary.image("Generated", [colorize(tf.transpose(outputSpectrogram))]),
                                tf.summary.image("Complete", [colorize(totalSpectrogram)])])
