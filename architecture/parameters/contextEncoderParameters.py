class ContextEncoderParameters(object):
    INPUT_CHANNELS = 4  # 2 sides, one for real and one for imag

    def __init__(self, batchSize, signalLength, gapLength, fftWindowLength, fftHopSize,
                 encoderParameters, fullyConnectedLayerParameters, decoderParameters):
        self._batchSize = batchSize
        self._signalLength = signalLength
        self._gapLength = gapLength
        self._fftWindowLength = fftWindowLength
        self._fftHopSize = fftHopSize
        self._encoderParameters = encoderParameters
        self._fullyConnectedLayerParameters = fullyConnectedLayerParameters
        self._decoderParameters = decoderParameters

    def architectureParameters(self):
        return [self.inputShape(), self._encoderParameters, self._decoderParameters, self._fullyConnectedLayerParameters]

    def preProcessorParameters(self):
        return [self._signalLength, self._gapLength, self._fftWindowLength, self._fftHopSize]

    def batchSize(self):
        return self._batchSize

    def inputShape(self):
        return self._batchSize, self._contextFrames(), self._fftFreqBins(), self.INPUT_CHANNELS

    def _fftFreqBins(self):
        self._fftWindowLength//2+1

    def _contextFrames(self):
        return ((self._signalLength-self._gapLength)/2)/self._fftHopSize
