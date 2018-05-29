import tensorflow as tf


class LSTMLogPreAndPostProcessor(object):
	def __init__(self, lstmParameters):
		self._lstmParameters = lstmParameters

	def inputAndTarget(self, signal):
		realAndImagStft = self._realAndImagStft(signal)

		netInput = realAndImagStft[:, :-1, :]
		target = realAndImagStft[:, 1:, :]

		return netInput, target

	def _realAndImagStft(self, signal):
		stft = tf.contrib.signal.stft(signals=signal,
									  frame_length=self._lstmParameters.fftWindowLength(),
									  frame_step=self._lstmParameters.fftHopSize())
		return self._log10(tf.abs(stft)+1e-4)

	def _log10(self, tensor):
		numerator = tf.log(tensor)
		denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
		return numerator / denominator
