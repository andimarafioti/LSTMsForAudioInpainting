class LstmParameters(object):
	def __init__(self, lstmSize, batchSize, signalLength, fftWindowLength, fftHopSize, countOfFrames):
		self._signalLength = signalLength
		self._batchSize = batchSize
		self._lstmSize = lstmSize
		self._fftWindowLength = fftWindowLength
		self._fftHopSize = fftHopSize
		self._countOfFrames = countOfFrames

	def signalLength(self):
		return self._signalLength

	def batchSize(self):
		return self._batchSize

	def countOfFrames(self):
		return self._countOfFrames

	def lstmSize(self):
		return self._lstmSize

	def fftFreqBins(self):
		return self._fftWindowLength//2+1

	def fftFrames(self):
		return (self._signalLength-self._fftWindowLength)/self._fftHopSize+1

	def inputFrames(self):
		return (self.fftFrames()-self._countOfFrames-1)*self.countOfFrames()

	def fftWindowLength(self):
		return self._fftWindowLength

	def fftHopSize(self):
		return self._fftHopSize
