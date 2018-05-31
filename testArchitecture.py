from architecture.contextEncoderLSTMArchitecture import ContextEncoderLSTMArchitecture
from architecture.parameters.lstmContextInpaintingParameters import LstmContextInpaintingParameters
from system.lstmSystem import LSTMSystem
import os

from system.preAndPostProcessor import PreAndPostProcessor

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

sessionsName = "test_mixingContextLSTM_"

params = LstmContextInpaintingParameters(lstmSize=512, batchSize=64, signalLength=5120, gapLength=1024,
										 fftWindowLength=512, fftHopSize=128)

contextArchitecture = ContextEncoderLSTMArchitecture(params)

aPreProcessor = PreAndPostProcessor(params)

aContextEncoderSystem = LSTMSystem(contextArchitecture, aPreProcessor, params, sessionsName)

aContextEncoderSystem.train("../variationalAutoEncoder/fake_w5120_g1024_h512.tfrecords", "../variationalAutoEncoder/fake_w5120_g1024_h512.tfrecords", 1e-3)
