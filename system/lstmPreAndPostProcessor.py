import tensorflow as tf


class LSTMPreAndPostProcessor(object):
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
		return tf.abs(stft)
		# real_part = tf.real(stft)
		# imag_part = tf.imag(stft)
		# return tf.stack([real_part, imag_part], axis=-1, name='divideComplexIntoRealAndImag')
