import tensorflow as tf
from architecture.simpleLSTMArchitecture import SimpleLSTMArchitecture


class RealImagLSTMArchitecture(SimpleLSTMArchitecture):
	def _lossGraph(self):
		with tf.variable_scope("Loss"):
			targetSquaredNorm = tf.reduce_sum(tf.square(self._target), axis=[1, 2, 3])

			error = self._target - self._output
			error_per_example = tf.reduce_sum(tf.square(error), axis=[1, 2, 3])

			reconstruction_loss = 0.5 * tf.reduce_sum(error_per_example * (1 + 5 / (targetSquaredNorm + 1e-4)))
			lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()]) * 1e-5
			total_loss = tf.add_n([reconstruction_loss, lossL2])

			total_loss_summary = tf.summary.scalar("total_loss", total_loss)
			l2_loss_summary = tf.summary.scalar("lossL2", lossL2)
			rec_loss_summary = tf.summary.scalar("reconstruction_loss", reconstruction_loss)
			self._lossSummaries = tf.summary.merge([rec_loss_summary, l2_loss_summary, total_loss_summary])

			return total_loss

	def _network(self, data, initial_state=None, reuse=False):
		real = super()._network(data[:, :, :, 0], initial_state, reuse)
		imag = super()._network(data[:, :, :, 1], initial_state, True)
		return tf.stack([real, imag], axis=-1)
