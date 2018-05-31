import tensorflow as tf
from architecture.contextEncoderLSTMArchitecture import ContextEncoderLSTMArchitecture


class RealImagContextEncoderLSTMArchitecture(ContextEncoderLSTMArchitecture):
    def _lossGraph(self):
        with tf.variable_scope("Loss"):

            reconstruction_loss = tf.reduce_sum(tf.square(self._target - self._output))
            lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()]) * 1e-5
            total_loss = tf.add_n([reconstruction_loss, lossL2])

            total_loss_summary = tf.summary.scalar("total_loss", total_loss)
            l2_loss_summary = tf.summary.scalar("lossL2", lossL2)
            rec_loss_summary = tf.summary.scalar("reconstruction_loss", reconstruction_loss)
            self._lossSummaries = tf.summary.merge([rec_loss_summary, l2_loss_summary, total_loss_summary])

            return total_loss

    def _network(self, data, reuse=False):
        real = super()._network(data[:, :, :, 0], reuse)
        imag = super()._network(data[:, :, :, 1], True)
        return tf.stack([real, imag], axis=-1)
