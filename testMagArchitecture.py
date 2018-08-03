from architecture.contextEncoderLSTMArchitecture import ContextEncoderLSTMArchitecture
from architecture.parameters.lstmContextInpaintingParameters import LstmContextInpaintingParameters
import os

from system.lstmSystem import LSTMSystem
from system.magPreAndPostProcessor import MagPreAndPostProcessor

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

sessionsName = "test_magnet_"

params = LstmContextInpaintingParameters(lstmSize=512, batchSize=256, signalLength=5120, inputChannels=4,
										 gapLength=1024, fftWindowLength=256, fftHopSize=64)

contextArchitecture = ContextEncoderLSTMArchitecture(params)

aPreProcessor = MagPreAndPostProcessor(params)

aContextEncoderSystem = LSTMSystem(contextArchitecture, aPreProcessor, params, sessionsName)

# aContextEncoderSystem.train("../variationalAutoEncoder/nsynth_train_w5120_g1024_h512.tfrecords",
# 							"../variationalAutoEncoder/nsynth_valid_w5120_g1024_h512.tfrecords", 1e-3)
#
aContextEncoderSystem.train("../../aDataset/nsynth_train_w5120_g1024_h512.tfrecords", "../../aDataset/nsynth_valid_w5120_g1024_h512.tfrecords", 1e-3)
