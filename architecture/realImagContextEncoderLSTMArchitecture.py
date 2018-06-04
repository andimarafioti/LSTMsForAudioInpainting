import tensorflow as tf
from architecture.contextEncoderLSTMArchitecture import ContextEncoderLSTMArchitecture


class RealImagContextEncoderLSTMArchitecture(ContextEncoderLSTMArchitecture):
    def _lossGraph(self):
        with tf.variable_scope("Loss"):
            freq_penalty = tf.range(start=0, limit=self._lstmParams.fftFreqBins(), delta=1, dtype=tf.float32) + 1

            reconstruction_loss = tf.reduce_sum(
                tf.reduce_sum(tf.square(self._target - self._output), axis=[0, 1, 3])*freq_penalty)

            lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()]) * 1e-5
            total_loss = tf.add_n([reconstruction_loss, lossL2])

            total_loss_summary = tf.summary.scalar("total_loss", total_loss)
            l2_loss_summary = tf.summary.scalar("lossL2", lossL2)
            rec_loss_summary = tf.summary.scalar("reconstruction_loss", reconstruction_loss)
            self._lossSummaries = tf.summary.merge([rec_loss_summary, l2_loss_summary, total_loss_summary])

            return total_loss

    def _network(self, context, reuse=False):
        real_context = tf.stack([context[:, :, :, 0], context[:, :, :, 2]], axis=-1)
        imag_context = tf.stack([context[:, :, :, 1], context[:, :, :, 3]], axis=-1)
        real = super()._network(real_context, reuse)
        imag = super()._network(imag_context, True)
        return tf.stack([real, imag], axis=-1)
