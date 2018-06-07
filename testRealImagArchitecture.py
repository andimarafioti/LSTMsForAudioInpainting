from architecture.parameters.lstmContextInpaintingParameters import LstmContextInpaintingParameters
from architecture.realImagContextEncoderLSTMArchitecture import RealImagContextEncoderLSTMArchitecture
import os

from system.preAndPostProcessor import PreAndPostProcessor
from system.realImagContextLSTMSystem import RealImagContextLSTMSystem

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

sessionsName = "test_realImagContext3LSTM512_7WeightsOnLstmed_fft256_"

params = LstmContextInpaintingParameters(lstmSize=512, batchSize=256, signalLength=5120, inputChannels=4,
										 gapLength=1024, fftWindowLength=256, fftHopSize=64)

contextArchitecture = RealImagContextEncoderLSTMArchitecture(params)

aPreProcessor = PreAndPostProcessor(params)

aContextEncoderSystem = RealImagContextLSTMSystem(contextArchitecture, aPreProcessor, params, sessionsName)

aContextEncoderSystem.train("../../aDataset/nsynth_train_w5120_g1024_h512.tfrecords", "../../aDataset/nsynth_valid_w5120_g1024_h512.tfrecords", 1e-3, restoreNum=322000)
