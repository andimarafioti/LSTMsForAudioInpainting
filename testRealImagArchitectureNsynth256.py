from architecture.parameters.lstmContextInpaintingParameters import LstmContextInpaintingParameters
from architecture.realImagContextEncoderLSTMArchitecture import RealImagContextEncoderLSTMArchitecture
import os

from system.preAndPostProcessor import PreAndPostProcessor
from system.realImagContextLSTMSystem import RealImagContextLSTMSystem

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

sessionsName = "test_nsynth4096_bs256_fft256_"

params = LstmContextInpaintingParameters(lstmSize=512, batchSize=256, signalLength=12288, inputChannels=4,
										 gapLength=4096, fftWindowLength=256, fftHopSize=64)

contextArchitecture = RealImagContextEncoderLSTMArchitecture(params)

aPreProcessor = PreAndPostProcessor(params)

aContextEncoderSystem = RealImagContextLSTMSystem(contextArchitecture, aPreProcessor, params, sessionsName)

aContextEncoderSystem.train("../../aDataset/nsynth_train_w12288_g4096_h512.tfrecords", "../../aDataset/nsynth_valid_w12288_g4096_h512.tfrecords", 1e-3)
