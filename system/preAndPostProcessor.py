import functools

import tensorflow as tf
from tensorflow.contrib.signal.python.ops import window_ops

__author__ = 'Andres'


class PreAndPostProcessor(object):
    def __init__(self, params):
        super(PreAndPostProcessor, self).__init__()
        self._signalLength = params.signalLength()
        self._gapLength = params.gapLength()
        self._fftWindowLength = params.fftWindowLength()
        self._fftHopSize = params.fftHopSize()

    def signalLength(self):
        return self._signalLength

    def gapLength(self):
        return self._gapLength

    def fftWindowLenght(self):
        return self._fftWindowLength

    def fftHopSize(self):
        return self._fftHopSize

    def windowOverlap(self):
        return self._fftHopSize / self._fftWindowLength

    def padding(self):
        return self._fftWindowLength - self._fftHopSize

    def stftForGapOf(self, aBatchOfSignals):
        assert len(aBatchOfSignals.shape) == 2
        signalWithoutExtraSides = self._removeExtraSidesForSTFTOfGap(aBatchOfSignals)
        return self._realAndImagSTFT(signalWithoutExtraSides)

    def stftForTheContextOf(self, aBatchOfSignals):
        assert len(aBatchOfSignals.shape) == 2
        leftAndRightSideStacked = self._removeGap(aBatchOfSignals)
        leftAndRightSideStackedAndPadded = self._addPaddingForStftOfContext(leftAndRightSideStacked)

        realAndImagSTFTOfLeftSide = self._realAndImagSTFT(leftAndRightSideStackedAndPadded[:, 0])
        realAndImagSTFTOfRightSide = self._realAndImagSTFT(leftAndRightSideStackedAndPadded[:, 1])

        contextRealAndImagSTFT = tf.concat([realAndImagSTFTOfLeftSide, realAndImagSTFTOfRightSide], axis=-1)
        return self._removePaddedCoefs(contextRealAndImagSTFT)

    def _realAndImagSTFT(self, aBatchOfSignals):
        stft = tf.contrib.signal.stft(signals=aBatchOfSignals,
                                      frame_length=self._fftWindowLength, frame_step=self._fftHopSize)
        return self._divideComplexIntoRealAndImag(stft)

    def _removePaddedCoefs(self, aBatchOfStft):
        """batchOfSides should contain the left side on the last dimension, first two channels
        and the right side on the second two"""
        overlap = self.windowOverlap()
        if overlap < 1:
            paddedWindows = int(1 / overlap) - 1
            leftSideNoPadded = aBatchOfStft[:, :-paddedWindows, :, :2]
            rightSideNoPadded = aBatchOfStft[:, paddedWindows:, :, 2:]
            return tf.concat([leftSideNoPadded, rightSideNoPadded], axis=-1)
        else:
            return aBatchOfStft

    def inverseStftOfGap(self, batchOfStftOfGap):
        window_fn = functools.partial(window_ops.hann_window, periodic=True)
        inverse_window = tf.contrib.signal.inverse_stft_window_fn(self._fftWindowLength, forward_window_fn=window_fn)
        padded_gaps = tf.contrib.signal.inverse_stft(stfts=batchOfStftOfGap, frame_length=self._fftWindowLength,
                                                     frame_step=self._fftHopSize, window_fn=inverse_window)
        return padded_gaps[:, self.padding():-self.padding()]

    def inverseStftOfSignal(self, batchOfStftsOfSignal):
        window_fn = functools.partial(window_ops.hann_window, periodic=True)
        inverse_window = tf.contrib.signal.inverse_stft_window_fn(self._fftWindowLength, forward_window_fn=window_fn)
        return tf.contrib.signal.inverse_stft(stfts=batchOfStftsOfSignal, frame_length=self._fftWindowLength,
                                              frame_step=self._fftHopSize, window_fn=inverse_window)

    def _gapBeginning(self):
        return (self._signalLength - self._gapLength) // 2

    def _gapEnding(self):
        return self._gapBeginning() + self._gapLength

    def _removeExtraSidesForSTFTOfGap(self, batchOfSignals):
        return batchOfSignals[:, self._gapBeginning() - self.padding(): self._gapEnding() + self.padding()]

    def _removeGap(self, batchOfSignals):
        leftSide = batchOfSignals[:, :self._gapBeginning()]
        rightSide = batchOfSignals[:, self._gapEnding():]
        return tf.stack((leftSide, rightSide), axis=1)

    def _addPaddingForStftOfContext(self, batchOfSides):
        """batchOfSides should contain the left side on the first dimension and the right side on the second"""
        batchSize = batchOfSides.shape.as_list()[0]
        leftSidePadded = tf.concat((batchOfSides[:, 0], tf.zeros((batchSize, self.padding()))), axis=1)
        rightSidePadded = tf.concat((tf.zeros((batchSize, self.padding())), batchOfSides[:, 1]), axis=1)
        return tf.stack((leftSidePadded, rightSidePadded), axis=1)

    def _divideComplexIntoRealAndImag(self, complexTensor):
        real_part = tf.real(complexTensor)
        imag_part = tf.imag(complexTensor)
        return tf.stack([real_part, imag_part], axis=-1, name='divideComplexIntoRealAndImag')
