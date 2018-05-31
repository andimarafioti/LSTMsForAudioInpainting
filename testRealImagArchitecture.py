from architecture.parameters.lstmContextInpaintingParameters import LstmContextInpaintingParameters
from architecture.realImagContextEncoderLSTMArchitecture import RealImagContextEncoderLSTMArchitecture
import os

from system.preAndPostProcessor import PreAndPostProcessor
from system.realImagContextLSTMSystem import RealImagContextLSTMSystem

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

sessionsName = "test_realImagContextLSTM_"

params = LstmContextInpaintingParameters(lstmSize=512, batchSize=64, signalLength=5120, inputChannels=4,
										 gapLength=1024, fftWindowLength=512, fftHopSize=128)

contextArchitecture = RealImagContextEncoderLSTMArchitecture(params)

aPreProcessor = PreAndPostProcessor(params)

aContextEncoderSystem = RealImagContextLSTMSystem(contextArchitecture, aPreProcessor, params, sessionsName)

aContextEncoderSystem.train("../variationalAutoEncoder/fake_w5120_g1024_h512.tfrecords", "../variationalAutoEncoder/fake_w5120_g1024_h512.tfrecords", 1e-3)
