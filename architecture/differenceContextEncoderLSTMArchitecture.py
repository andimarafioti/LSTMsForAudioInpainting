import tensorflow as tf
from architecture.contextEncoderLSTMArchitecture import ContextEncoderLSTMArchitecture


class DifferenceContextEncoderLSTMArchitecture(ContextEncoderLSTMArchitecture):
    def _lstmNetwork(self, data, initial_state, reuse, name):
        with tf.variable_scope(name, reuse=reuse):
            dataset = tf.unstack(data, axis=-2)

            rnn_cell = tf.contrib.rnn.MultiRNNCell(
                [tf.contrib.rnn.LSTMCell(self._lstmParams.lstmSize()),
                 tf.contrib.rnn.LSTMCell(self._lstmParams.lstmSize()),
                 tf.contrib.rnn.LSTMCell(self._lstmParams.lstmSize())])
            outputs, states = tf.nn.static_rnn(rnn_cell, dataset, initial_state=initial_state, dtype=tf.float32)

            weights = self._weight_variable([self._lstmParams.lstmSize(), self._lstmParams.fftFreqBins()])
            biases = self._bias_variable([self._lstmParams.fftFreqBins()])

            mat_muled = tf.matmul(outputs[0], weights) + biases + dataset[-1]
            output = tf.expand_dims(mat_muled, axis=1)
            out_output = output

            for output in outputs[1:]:
                mat_muled = tf.matmul(output, weights) + biases + out_output[:, -1]
                output = tf.expand_dims(mat_muled, axis=1)
                out_output = tf.concat([out_output, output], axis=1)
            return out_output, states

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

        return tf.stack([real, imag], axis=-1)
