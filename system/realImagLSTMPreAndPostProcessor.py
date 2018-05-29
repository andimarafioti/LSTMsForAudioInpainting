import tensorflow as tf

from system.lstmPreAndPostProcessor import LSTMPreAndPostProcessor


class RealImagLSTMPreAndPostProcessor(LSTMPreAndPostProcessor):
	def _realAndImagStft(self, signal):
		stft = tf.contrib.signal.stft(signals=signal,
									  frame_length=self._lstmParameters.fftWindowLength(),
									  frame_step=self._lstmParameters.fftHopSize())
		real_part = tf.real(stft)
		imag_part = tf.imag(stft)
		return tf.stack([real_part, imag_part], axis=-1, name='divideComplexIntoRealAndImag')
