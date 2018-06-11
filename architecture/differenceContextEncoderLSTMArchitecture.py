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
