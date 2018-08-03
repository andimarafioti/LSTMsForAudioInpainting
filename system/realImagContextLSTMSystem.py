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
        outputSpectrogram = tf.sqrt(tf.reduce_sum(tf.square(complexOutput), axis=-1))

        complexTarget = self._architecture.target()[0]
        targetSpectrogram = tf.sqrt(tf.reduce_sum(tf.square(complexTarget), axis=-1))

        complexLeft = self._architecture.input()[0, :, :, 0:2]
        leftSpectrogram = tf.sqrt(tf.reduce_sum(tf.square(complexLeft), axis=-1))

        complexRight = self._architecture.input()[0, :, :, 2:4]
        rightSpectrogram = tf.sqrt(tf.reduce_sum(tf.square(complexRight), axis=-1))

        complexForward = self._architecture._forwardPrediction[0]
        forwardSpectrogram = tf.sqrt(tf.reduce_sum(tf.square(complexForward), axis=-1))

        complexBackward = self._architecture._backwardPrediction[0]
        backwardSpectrogram = tf.sqrt(tf.reduce_sum(tf.square(complexBackward), axis=-1))

        totalSpectrogram = tf.transpose(tf.concat([leftSpectrogram, outputSpectrogram,
                                                   rightSpectrogram], axis=0))

        frontPrediction = tf.transpose(tf.concat([leftSpectrogram, forwardSpectrogram,
                                                   rightSpectrogram], axis=0))

        backPrediction = tf.transpose(tf.concat([leftSpectrogram, backwardSpectrogram,
                                                   rightSpectrogram], axis=0))

        original = tf.transpose(tf.concat([leftSpectrogram, targetSpectrogram,
                                                   rightSpectrogram], axis=0))

        return tf.summary.merge([tf.summary.image("Original", [colorize(original)]),
                                 tf.summary.image("Forward", [colorize(frontPrediction)]),
                                tf.summary.image("Backward", [colorize(backPrediction)]),
                                tf.summary.image("Complete", [colorize(totalSpectrogram)])])
