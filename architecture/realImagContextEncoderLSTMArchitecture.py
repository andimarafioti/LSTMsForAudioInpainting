import tensorflow as tf
from architecture.contextEncoderLSTMArchitecture import ContextEncoderLSTMArchitecture


class RealImagContextEncoderLSTMArchitecture(ContextEncoderLSTMArchitecture):
    def _lossGraph(self):
        with tf.variable_scope("Loss"):
            forward_reconstruction_loss = tf.reduce_sum(tf.square(self._target - self._forwardPrediction))
            backward_reconstruction_loss = tf.reduce_sum(tf.square(self._target - self._backwardPrediction))

            reconstruction_loss = tf.reduce_sum(tf.square(self._target - self._output))

            lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()]) * 1e-5
            total_loss = tf.add_n([reconstruction_loss, lossL2, forward_reconstruction_loss,
                                   backward_reconstruction_loss])

            total_loss_summary = tf.summary.scalar("total_loss", total_loss)
            l2_loss_summary = tf.summary.scalar("lossL2", lossL2)
            rec_loss_summary = tf.summary.scalar("reconstruction_loss", reconstruction_loss)
            frec_loss_summary = tf.summary.scalar("forw_reconstruction_loss", forward_reconstruction_loss)
            brec_loss_summary = tf.summary.scalar("back_reconstruction_loss", backward_reconstruction_loss)
            self._lossSummaries = tf.summary.merge([rec_loss_summary, l2_loss_summary, total_loss_summary,
                                                    frec_loss_summary, brec_loss_summary])

            return total_loss

    def _network(self, context, reuse=False):
        real_context = tf.stack([context[:, :, :, 0], context[:, :, :, 2]], axis=-1)
        imag_context = tf.stack([context[:, :, :, 1], context[:, :, :, 3]], axis=-1)
        real = super()._network(real_context, reuse)
        real_forward = self._forwardPrediction
        real_backward = self._backwardPrediction
        imag = super()._network(imag_context, True)
        imag_forward = self._forwardPrediction
        imag_backward = self._backwardPrediction
        self._forwardPrediction = tf.stack([real_forward, imag_forward], axis=-1)
        self._backwardPrediction = tf.stack([real_backward, imag_backward], axis=-1)
        print(self._backwardPrediction.shape)

        return tf.stack([real, imag], axis=-1)
