import numpy as np
import librosa
from matplotlib import pyplot as plt 
from scipy.io import wavfile

def setUpAudioData():
    audioData, sampleRate = librosa.load('Beethoven_Symphony_No._5_Movement_2,_La_Folia_Variation_(measures_166_-_183).wav', sr=None)
    ##nyquist rate assuming frequency is band limited by what the human ear can hear, which is 20kHz
    nyquistRate = 40000
    lengthOfAudio = len(audioData)/sampleRate
    originalNumberOfSamples = len(audioData)
    ##As per Nyquist-Shannon theorem, we need a sample rate of 40kHz
    samplesNeededFromShannon = int(np.ceil(nyquistRate*originalNumberOfSamples/sampleRate))
    # Calculate the indices of the evenly spaced samples, such that there are the sufficient samples to satisfy shanon's theorem, and we remove extra samples
    indices = np.round(np.linspace(0, originalNumberOfSamples- 1, int(samplesNeededFromShannon))).astype(int)
    # Select the samples using the calculated indices
    audioData = audioData[indices]
    #We need the number of samples to satisfy Nyquist-Shannon theorem's criteria but also be of the form 2^N for our FFT function
    zeroesNeededForPwtwo = int(2**np.ceil(np.log2(samplesNeededFromShannon)) - samplesNeededFromShannon)
    #Add trailing zeroes so signal is of the form 2^N, without affecting the result
    audioData = np.pad(audioData, (0, int(zeroesNeededForPwtwo)), mode='constant', constant_values=(0, 0)).tolist()
    return (audioData, lengthOfAudio, samplesNeededFromShannon)

def setUpNoisy():
    audioData, sampleRate = librosa.load('Beethoven_Symphony_No._5_Movement_2,_La_Folia_Variation_(measures_166_-_183).wav', sr=None)
    ##nyquist rate assuming frequency is band limited by what the human ear can hear, which is 20kHz
    nyquistRate = 40000
    originalNumberOfSamples = len(audioData)
    #add noise. Low amplitude as noise will be "lower" than original sampled sound, and it will follow normal distribution. In this case, the mean amplitude of added noise is of 0.02
    audioData = audioData + 0.02*np.random.randn(len(audioData))
    ##As per Nyquist-Shannon theorem, we need a sample rate of 40kHz
    samplesNeededFromShannon = int(np.ceil(nyquistRate*originalNumberOfSamples/sampleRate))
    # Calculate the indices of the evenly spaced samples, such that there are the sufficient samples to satisfy shanon's theorem, and we remove extra samples
    indices = np.round(np.linspace(0, originalNumberOfSamples- 1, int(samplesNeededFromShannon))).astype(int)
    # Select the samples using the calculated indices
    audioData = audioData[indices]
    #We need the number of samples to satisfy Nyquist-Shannon theorem's criteria but also be of the form 2^N for our FFT function
    zeroesNeededForPwtwo = int(2**np.ceil(np.log2(samplesNeededFromShannon)) - samplesNeededFromShannon)
    #Add trailing zeroes so signal is of the form 2^N, without affecting the result
    audioData = np.pad(audioData, (0, int(zeroesNeededForPwtwo)), mode='constant', constant_values=(0, 0)).tolist()
    return (audioData)

def FFT(a, inverse = False):
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

def denoiseSignal(frequencyData, lowPassFilter):
    frequencyData = frequencyData.copy()
    psd = [aHat*np.conj(aHat)/len(frequencyData) for aHat in frequencyData]
    for i in range(len(psd)):
        power = psd[i]
        if power < lowPassFilter:
            frequencyData[i] = 0
    return frequencyData

audioData, stopTime, samplesNeededFromShannon = setUpAudioData()
noisyAudioData = setUpNoisy()
t= np.linspace(0, stopTime, samplesNeededFromShannon)

noisyFreqs = FFT(noisyAudioData.copy())
originalFreqs = FFT(audioData.copy())
denoisedFreqs = denoiseSignal(noisyFreqs, 2e-09)
denoisedSignal = FFT(denoisedFreqs.copy(), True)

reconstructNoisy = np.array(noisyAudioData[0:samplesNeededFromShannon], dtype=np.float32)
wavfile.write('noisywav.wav', int(samplesNeededFromShannon/stopTime), reconstructNoisy)
reconstructDenoised = np.array(denoisedSignal[0:samplesNeededFromShannon], dtype=np.float32)
wavfile.write('denoised.wav', int(samplesNeededFromShannon/stopTime), reconstructDenoised)

figure, axis = plt.subplots(3, 1)   
axis[0].plot(t, audioData[0:samplesNeededFromShannon],  linewidth=0.3)
axis[0].set_ylabel('Amplitude (normalized)')
axis[0].set_title('Clean Audio')
axis[0].set_xlabel("Time (s)")

axis[1].plot(t, noisyAudioData[0:samplesNeededFromShannon], linewidth=0.3)
axis[1].set_ylabel('Amplitude (normalized)')
axis[1].set_title('Noisy Audio')
axis[1].set_xlabel("Time (s)")

axis[2].plot(t, denoisedSignal[0:samplesNeededFromShannon],  linewidth=0.3)
axis[2].set_ylabel('Amplitude (normalized)')
axis[2].set_title('Denoised Audio')
axis[2].set_xlabel("Time (s)")

plt.subplots_adjust(hspace=2)
plt.show()