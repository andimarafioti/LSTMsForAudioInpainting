from architecture.parameters.lstmContextInpaintingParameters import LstmContextInpaintingParameters
from architecture.realImagContextEncoderLSTMArchitecture import RealImagContextEncoderLSTMArchitecture
import os

from system.preAndPostProcessor import PreAndPostProcessor
from system.realImagContextLSTMSystem import RealImagContextLSTMSystem

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

sessionsName = "test_realImagContext3LSTM512_fft256_nsynth2048_"

params = LstmContextInpaintingParameters(lstmSize=512, batchSize=64, signalLength=6144, inputChannels=4,
										 gapLength=2048, fftWindowLength=256, fftHopSize=64)

contextArchitecture = RealImagContextEncoderLSTMArchitecture(params)

aPreProcessor = PreAndPostProcessor(params)

aContextEncoderSystem = RealImagContextLSTMSystem(contextArchitecture, aPreProcessor, params, sessionsName)

aContextEncoderSystem.train("../../aDataset/nsynth_train_w6144_g2048_h512.tfrecords", "../../aDataset/nsynth_valid_w6144_g2048_h512.tfrecords", 1e-3)
