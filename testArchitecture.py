from architecture.parameters.lstmParameters import LstmParameters
from architecture.simpleLSTMArchitecture import SimpleLSTMArchitecture
from system.lstmPreAndPostProcessor import LSTMPreAndPostProcessor
from system.lstmSystem import LSTMSystem
import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

sessionsName = "test_mag_"

batch_size = 64
params = LstmParameters(lstmSize=512, batchSize=batch_size, signalLength=5120, fftWindowLength=128, fftHopSize=32,
						countOfFrames=4)

with tf.variable_scope("forward_LSTM"):
	aForwardLSTM = SimpleLSTMArchitecture(inputShape=(params.batchSize(),
																 params.fftFrames()-1,
																 params.fftFreqBins()), lstmParams=params)
with tf.variable_scope("backward_LSTM"):
	aBackwardLSTM = SimpleLSTMArchitecture(inputShape=(params.batchSize(),
																 params.fftFrames()-1,
																 params.fftFreqBins()), lstmParams=params)

aPreProcessor = PreAndPostProcessor(params)

aPreProcessor = LSTMPreAndPostProcessor(params)

aContextEncoderSystem = LSTMSystem(aContextEncoderArchitecture, batch_size, aPreProcessor, params, sessionsName)

aContextEncoderSystem.train("../variationalAutoEncoder/fake_w5120_g1024_h512.tfrecords", "../variationalAutoEncoder/fake_w5120_g1024_h512.tfrecords", 1e-3)
