import numpy as np

def fft(a, inverse = False):
    '''
    Given a sequence a, returns the dft. If inverse parameter is true, return the inverse of the dft.
    '''
    conj = 1 if inverse else -1
    originalLength = len(a)
    conjRootsOfUnity = [np.exp(conj*2*np.pi*1j*l/originalLength) for l in range (originalLength)]
    def recursiveFFT(FourierVector, N, conjugateFactor):
        if N == 1:
            return FourierVector.copy()
        else:
            NHalf = int(N/2)
            aEven = recursiveFFT(FourierVector[::2], NHalf, 2*conjugateFactor)
            aOdd = recursiveFFT(FourierVector[1::2], NHalf, 2*conjugateFactor)
            for i in range(N):
                FourierVector[i] = aEven[i % NHalf] + conjRootsOfUnity[i*conjugateFactor]*aOdd[i % NHalf]
            return FourierVector.copy()
    return recursiveFFT(a, originalLength, 1)/np.sqrt(originalLength)