import tensorflow as tf
import numpy as np
import copy
from system.dnnSystem import DNNSystem
from utils.colorize import colorize
from utils.legacy.plotSummary import PlotSummary
from utils.strechableNumpyArray import StrechableNumpyArray
from utils.tfReader import TFReader


class LSTMSystem(DNNSystem):
    def __init__(self, architecture, aPreProcessor, lstmParameters, name):
        self._aPreProcessor = aPreProcessor
        self._lstmParameters = lstmParameters
        self._windowSize = lstmParameters.signalLength()
        self._batchSize = lstmParameters.batchSize()
        self._audio = tf.placeholder(tf.float32, shape=(self._batchSize, self._windowSize), name='audio_data')
        self._preProcessForContext = aPreProcessor.stftForTheContextOf(self._audio)
        self._preProcessForGap = aPreProcessor.stftForGapOf(self._audio)
        super().__init__(architecture, name)
        self._SNR = tf.reduce_mean(self._pavlovs_SNR(self._architecture.output(), self._architecture.target(), onAxis=[1, 2]))

    def reconstruct(self, aBatchOfSignals, model_num):
        with tf.Session() as sess:
            path = self.modelsPath(model_num)
            saver = tf.train.Saver()
            saver.restore(sess, path)
            print("Model restored.")
            sess.run([tf.local_variables_initializer()])
            return self._reconstruct(sess, aBatchOfSignals)

    def _reconstruct(self, sess, aBatchOfSignals):
        reconstructed = StrechableNumpyArray()
        out_gaps = StrechableNumpyArray()
        input_shape = self._architecture.input().shape.as_list()
        input_shape[0] = 0
        contexts = np.empty(input_shape)

        for batch_num in range(int(len(aBatchOfSignals)/self._batchSize)):
            feed_dict = self._feedDict(aBatchOfSignals[batch_num*self._batchSize:(batch_num+1)*self._batchSize], sess, False)
            reconstructed_input, original, context = sess.run([self._architecture.output(), self._architecture.target(),
                                                      self._architecture.input()],
                                                     feed_dict=feed_dict)
            out_gaps.append(np.reshape(original, (-1)))
            reconstructed.append(np.reshape(reconstructed_input, (-1)))
            contexts = np.concatenate([contexts, context], axis=0)

        output_shape = self._architecture.output().shape.as_list()
        output_shape[0] = (batch_num+1) * self._batchSize
        reconstructed = reconstructed.finalize()
        reconstructed = np.reshape(reconstructed, output_shape)
        out_gaps = out_gaps.finalize()
        out_gaps = np.reshape(out_gaps, output_shape)

        return reconstructed, out_gaps, contexts

    def optimizer(self, learningRate):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            return tf.train.AdamOptimizer(learning_rate=learningRate).minimize(self._architecture.loss())

    def _feedDict(self, data, sess, isTraining=True):
        net_input, net_target = sess.run([self._preProcessForContext, self._preProcessForGap], feed_dict={self._audio: data})
        return {self._architecture.input(): net_input, self._architecture.target(): net_target,
                self._architecture.isTraining(): isTraining}

    def _spectrogramImageSummary(self):
        output = tf.transpose(self._architecture.output()[0])
        target = tf.transpose(self._architecture.target()[0])
        total = tf.transpose(tf.concat([self._architecture.input()[0, :, :, 0], self._architecture.output()[0],
                          self._architecture.input()[0, :, :, 1]], axis = 0))

        return tf.summary.merge([tf.summary.image("Original", [colorize(target)]),
                                tf.summary.image("Generated", [colorize(output)]),
                                tf.summary.image("Complete", [colorize(total)])])

    def _evaluate(self, summariesDict, feed_dict, validReader, sess):
        trainSNRSummaryToWrite = sess.run(summariesDict['train_SNR_summary'], feed_dict=feed_dict)

        try:
            audio = validReader.dataOperation(session=sess)
        except StopIteration:
            print("valid End of queue!")
            return [trainSNRSummaryToWrite]
        feed_dict = self._feedDict(audio, sess, False)
        validSNRSummary = sess.run(summariesDict['valid_SNR_summary'], feed_dict)
        imageSummary = sess.run(summariesDict['image_summaries'], feed_dict)

        # forwardVars = sess.run(self._architecture._forwardVars)
        # summariesDict['forward'].plotVector(forwardVars)
        # forwardSumm = summariesDict['forward'].produceSummaryToWrite(sess)
        # backwardVars = sess.run(self._architecture._backwardVars)
        # summariesDict['backward'].plotVector(backwardVars)
        # backwardSumm = summariesDict['backward'].produceSummaryToWrite(sess)

        return [trainSNRSummaryToWrite, validSNRSummary, imageSummary] #, forwardSumm, backwardSumm]

    def _loadReader(self, dataPath, capacity=int(1e6)):
        return TFReader(dataPath, self._windowSize, batchSize=self._batchSize, capacity=capacity, num_epochs=400)

    def _evaluationSummaries(self):
        summaries_dict = {'train_SNR_summary': tf.summary.scalar("training_SNR", self._SNR),
                        'valid_SNR_summary': tf.summary.scalar("validation_SNR", self._SNR),
                        'image_summaries': self._spectrogramImageSummary(),
                        'forward': PlotSummary('forward'),
                        'backward': PlotSummary('backward')}
        return summaries_dict

    def _squaredEuclideanNorm(self, tensor, onAxis=[1, 2, 3]):
        squared = tf.square(tensor)
        summed = tf.reduce_sum(squared, axis=onAxis)
        return summed

    def _log10(self, tensor):
        numerator = tf.log(tensor)
        denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
        return numerator / denominator

    def _pavlovs_SNR(self, y_orig, y_inp, onAxis=[1, 2, 3]):
        norm_y_orig = self._squaredEuclideanNorm(y_orig, onAxis)
        norm_y_orig_minus_y_inp = self._squaredEuclideanNorm(y_orig - y_inp, onAxis)
        return 10 * self._log10(norm_y_orig / norm_y_orig_minus_y_inp)
