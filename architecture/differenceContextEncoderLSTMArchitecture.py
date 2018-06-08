import tensorflow as tf
import numpy as np
from architecture.contextEncoderLSTMArchitecture import ContextEncoderLSTMArchitecture


class DifferenceContextEncoderLSTMArchitecture(ContextEncoderLSTMArchitecture):
    def _lossGraph(self):
        with tf.variable_scope("Loss"):
            forward_reconstruction_loss = tf.reduce_sum(tf.square(self._target - self._forwardPrediction))
            backward_reconstruction_loss = tf.reduce_sum(tf.square(self._target - self._backwardPrediction))

            reconstruction_loss = tf.reduce_sum(tf.square(self._target - self._output))

            lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()]) * 1e-5
            total_loss = tf.add_n([reconstruction_loss, lossL2])

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
        real = self.__network(real_context, reuse)
        real_forward = self._forwardPrediction
        real_backward = self._backwardPrediction
        imag = self.__network(imag_context, True)
        imag_forward = self._forwardPrediction
        imag_backward = self._backwardPrediction
        self._forwardPrediction = tf.stack([real_forward, imag_forward], axis=-1)
        self._backwardPrediction = tf.stack([real_backward, imag_backward], axis=-1)

        return tf.stack([real, imag], axis=-1)

    def __network(self, context, reuse=False):
        with tf.variable_scope("Network", reuse=reuse):
            #prepare data
            forward_context = context[:, :, :, 0]
            forward_context = forward_context[:, 1:] - forward_context[:, :-1]
            backward_context = tf.reverse(context[:, :, :, 1], axis=[1])
            backward_context = -1 * (backward_context[:, 1:] - backward_context[:, :-1])

            #run through network
            forward_lstmed, forward_states = self._lstmNetwork(forward_context, None, reuse, 'forward_lstm')
            backward_lstmed, backward_states = self._lstmNetwork(backward_context, None, reuse, 'backward_lstm')

            forwards_gap = forward_lstmed[:, -1:, :]
            backwards_gap = backward_lstmed[:, -1:, :]

            for i in range(1, int(self._lstmParams.gapStftFrameCount())):
                next_frame, forward_states = self._lstmNetwork(forwards_gap[:, -1:, :], forward_states, True, 'forward_lstm')
                forwards_gap = tf.concat([forwards_gap, next_frame], axis=1)

                previous_frame, backward_states = self._lstmNetwork(backwards_gap[:, -1:, :], backward_states, True, 'backward_lstm')
                backwards_gap = tf.concat([backwards_gap, previous_frame], axis=1)

            #PostProcess LSTMed data
            backwards_gap = tf.reverse(backwards_gap, axis=[1])

            #mix the two predictions
            mixing_variables = self._weight_variable([7*2*self._lstmParams.fftFreqBins(),
                                                      self._lstmParams.fftFreqBins()])
            self._forwardPrediction = forwards_gap
            self._backwardPrediction = backwards_gap

            output = context[:, -1, :, 0]
            output = tf.expand_dims(output, axis=1)

            concat_gaps = tf.concat([forwards_gap, backwards_gap], axis=-1)
            left_side = forward_lstmed[:, -4:-1, :]
            left_doubled_side = tf.concat([left_side, left_side], axis=-1)

            right_side = tf.reverse(backward_lstmed[:, -4:-1, :], axis=[1])
            right_doubled_side = tf.concat([right_side, right_side], axis=-1)
            total_gaps = tf.concat([left_doubled_side, concat_gaps, right_doubled_side], axis=1)
            for i in range(int(self._lstmParams.gapStftFrameCount())):
                intermediate_output = tf.reshape(total_gaps[:, i:i+7, :], (self._lstmParams.batchSize(),
                                                                           7*2*self._lstmParams.fftFreqBins()))
                intermediate_output = tf.matmul(intermediate_output, mixing_variables)
                intermediate_output = output[:, -1] + intermediate_output
                intermediate_output = tf.expand_dims(intermediate_output, axis=1)
                output = tf.concat([output, intermediate_output], axis=1)
            return output[:, 1:]
