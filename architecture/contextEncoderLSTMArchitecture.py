import tensorflow as tf
import numpy as np
from architecture.architecture import Architecture


class ContextEncoderLSTMArchitecture(Architecture):
    def __init__(self, lstmParams):
        with tf.variable_scope("LSTMArchitecture"):
            self._lstmParams = lstmParams
            self._inputShape = (lstmParams.batchSize(), lstmParams.contextStftFrameCount(), lstmParams.fftFreqBins(), lstmParams.inputChannels())
            super().__init__()

    def inputShape(self):
        return self._inputShape

    def _lossGraph(self):
        with tf.variable_scope("Loss"):
            # normalize values !! divide by max input and multiply output

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

    def _lstmNetwork(self, data, initial_state, reuse, name):
        with tf.variable_scope(name, reuse=reuse):
            dataset = tf.unstack(data, axis=-2)

            with tf.variable_scope('linearInput', reuse=reuse):
                weights = self._weight_variable([self._lstmParams.fftFreqBins(), self._lstmParams.fftFreqBins()])
                biases = self._bias_variable([self._lstmParams.fftFreqBins()])

                datasetAfterLinear = np.empty([0, self._lstmParams.batchSize(), self._lstmParams.fftFreqBins()], dtype=np.float32)
                for data in dataset:
                    mat_muled = tf.matmul(data, weights) + biases
                    reshaped = tf.reshape(mat_muled, [-1, self._lstmParams.fftFreqBins()])
                    datasetAfterLinear = tf.concat([datasetAfterLinear, [reshaped]], axis=0)
            unstacked = tf.unstack(datasetAfterLinear, axis=0)

            rnn_cell = tf.contrib.rnn.MultiRNNCell(
                [tf.contrib.rnn.LSTMCell(self._lstmParams.lstmSize()),
                 tf.contrib.rnn.LSTMCell(self._lstmParams.lstmSize()),
                 tf.contrib.rnn.LSTMCell(self._lstmParams.lstmSize())])
            outputs, states = tf.nn.static_rnn(rnn_cell, unstacked, initial_state=initial_state, dtype=tf.float32)

            out_output = np.empty([data.shape[0], 0, self._lstmParams.fftFreqBins()])
            weights = self._weight_variable([self._lstmParams.lstmSize(), self._lstmParams.fftFreqBins()])
            biases = self._bias_variable([self._lstmParams.fftFreqBins()])

            for output in outputs:
                mat_muled = tf.matmul(output, weights) + biases
                output = tf.reshape(mat_muled, [-1, 1, self._lstmParams.fftFreqBins()])
                out_output = tf.concat([out_output, output], axis=1)
            return out_output, states

    # def normalize(self, data):
    #     maxim = tf.reduce_max(data)
    #     minim = tf.reduce_min(data)
    #     normed = (data-minim)/(maxim-minim+1e-8)
    #     return normed

    # def deNormalize(self, normalDataAndoriginalData):
    #     normalData, originalData = normalDataAndoriginalData[0], normalDataAndoriginalData[1]
    #     originalMax = tf.reduce_max(originalData)
    #     originalMin = tf.reduce_min(originalData)
    #     unNormed = normalData*(originalMax-originalMin+1e-8)+originalMin
    #     return unNormed

    def _network(self, context, reuse=False):
        with tf.variable_scope("Network", reuse=reuse):
            #prepare data
            forward_context = context[:, :, :, 0]
            backward_context = tf.reverse(context[:, :, :, 1], axis=[1])

            #normalize
            # normal_forward_context = tf.layers.batch_normalization(forward_context, training=self._isTraining)
            # normal_backward_context = tf.layers.batch_normalization(backward_context, training=self._isTraining)

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

            output = tf.zeros([self._lstmParams.batchSize(), 0, self._lstmParams.fftFreqBins()])

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
                intermediate_output = tf.expand_dims(intermediate_output, axis=1)
                output = tf.concat([output, intermediate_output], axis=1)
            return output

            # with tf.variable_scope('forward', reuse=reuse):
            #     mixing_forward_variables = self._weight_variable(
            #         [self._lstmParams.gapStftFrameCount(), self._lstmParams.fftFreqBins(), self._lstmParams.fftFreqBins()])
            # with tf.variable_scope('backward', reuse=reuse):
            #     mixing_backward_variables = self._weight_variable(
            #         [self._lstmParams.gapStftFrameCount(), self._lstmParams.fftFreqBins(), self._lstmParams.fftFreqBins()])
            #
            # self._forwardVars = mixing_forward_variables
            # self._backwardVars = mixing_backward_variables
            #
            # self._forwardPrediction = forwards_gap
            # self._backwardPrediction = backwards_gap
            #
            # output = tf.zeros([self._lstmParams.batchSize(), 0, self._lstmParams.fftFreqBins()])
            #
            # for i in range(int(self._lstmParams.gapStftFrameCount())):
            #     intermediate_output = tf.matmul(forwards_gap[:, i, :], mixing_forward_variables[i]) + tf.matmul(
            #         backwards_gap[:, i, :], mixing_backward_variables[i])
            #     intermediate_output = tf.expand_dims(intermediate_output, axis=1)
            #     output = tf.concat([output, intermediate_output], axis=1)
            #
            # return output

    def _weight_variable(self, shape):
        return tf.get_variable('W', shape, initializer=tf.contrib.layers.xavier_initializer())

    def _bias_variable(self, shape):
        return tf.get_variable('bias', shape, initializer=tf.contrib.layers.xavier_initializer())
