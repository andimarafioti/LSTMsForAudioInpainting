from architecture.parameters.lstmParameters import LstmParameters
from architecture.realImagLSTMArchitecture import RealImagLSTMArchitecture

import os

from system.realImagLSTMPreAndPostProcessor import RealImagLSTMPreAndPostProcessor
from system.realImagLSTMSystem import RealImagLSTMSystem

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

sessionsName = "test_RealImag_"

batch_size = 64
params = LstmParameters(lstmSize=512, batchSize=batch_size, signalLength=5120, fftWindowLength=128, fftHopSize=32,
						countOfFrames=4)

aContextEncoderArchitecture = RealImagLSTMArchitecture(inputShape=(params.batchSize(),
																 params.fftFrames()-1,
																 params.fftFreqBins(), 2), lstmParams=params)

aPreProcessor = RealImagLSTMPreAndPostProcessor(params)

aContextEncoderSystem = RealImagLSTMSystem(aContextEncoderArchitecture, batch_size, aPreProcessor, params, sessionsName)

aContextEncoderSystem.train("../variationalAutoEncoder/fake_w5120_g1024_h512.tfrecords", "../variationalAutoEncoder/fake_w5120_g1024_h512.tfrecords", 1e-3)
