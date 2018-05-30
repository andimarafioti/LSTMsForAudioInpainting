class LstmContextInpaintingParameters(object):
    def __init__(self, lstmSize, batchSize, signalLength, gapLength, fftWindowLength, fftHopSize):
        self._signalLength = signalLength
        self._gapLength = gapLength
        self._batchSize = batchSize
        self._lstmSize = lstmSize
        self._fftWindowLength = fftWindowLength
        self._fftHopSize = fftHopSize

    def preAndPostProcessorParams(self):
        return self._signalLength, self._gapLength, self._fftWindowLength, self._fftHopSize

    def signalLength(self):
        return self._signalLength

    def gapLength(self):
        return self._gapLength

    def batchSize(self):
        return self._batchSize

    def lstmSize(self):
        return self._lstmSize

    def _frameCountForSignalLength(self, length):
        return ((length - self._fftWindowLength) / self._fftHopSize) + 1

    def padding(self):
        return self._fftWindowLength - self._fftHopSize

    def gapStftFrameCount(self):
        return self._frameCountForSignalLength(self._gapLength + (self.padding()) * 2)

    def contextStftFrameCount(self):
        return self._frameCountForSignalLength(((self._signalLength - self._gapLength) / 2) + self.padding())

    def fftFreqBins(self):
        return self._fftWindowLength // 2 + 1

    def fftFrames(self):
        return (self._signalLength - self._fftWindowLength) / self._fftHopSize + 1

    def fftWindowLength(self):
        return self._fftWindowLength

    def fftHopSize(self):
        return self._fftHopSize
