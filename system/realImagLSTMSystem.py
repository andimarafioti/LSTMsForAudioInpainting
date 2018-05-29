import tensorflow as tf
from system.lstmSystem import LSTMSystem
from utils.colorize import colorize


class RealImagLSTMSystem(LSTMSystem):
	def _spectrogramImageSummary(self):
		frames = 12
		originalAndGeneratedSpectrogram = self._architecture.generateXOutputs(self._architecture.input(), frames)[0]
		originalAndGeneratedSpectrogram = tf.square(originalAndGeneratedSpectrogram[:, :, 0]) + \
										  tf.square(originalAndGeneratedSpectrogram[:, :, 1])
		originalAndGeneratedSpectrogram = tf.transpose(originalAndGeneratedSpectrogram)
		originalAndGeneratedSpectrogram = colorize(originalAndGeneratedSpectrogram)
		originalAndGeneratedSpectrogram = tf.expand_dims(originalAndGeneratedSpectrogram, 0)

		originalImage = originalAndGeneratedSpectrogram[:, :, int(self._lstmParameters.fftFrames()) -
															  frames - 1:int(self._lstmParameters.fftFrames()) - 1, :]
		generatedImage = originalAndGeneratedSpectrogram[:, :, int(self._lstmParameters.fftFrames()) - 1:, :]

		return tf.summary.merge([tf.summary.image("Original", originalImage),
								 tf.summary.image("Generated", generatedImage),
								 tf.summary.image("Complete", originalAndGeneratedSpectrogram[
															  :, :, int(self._lstmParameters.fftFrames()) - frames:,
															  :])])
