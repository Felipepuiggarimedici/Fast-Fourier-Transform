import numpy as np
import librosa
from matplotlib import pyplot as plt 

def setUpAudioData():
    audioData, sampleRate = librosa.load('pinkpanther10.wav', sr=None)
    ##nyquist rate assuming frequency is band limited by what the human ear can hear, which is 20kHz
    nyquistRate = 40000
    originalNumberOfSamples = len(audioData)
    print(originalNumberOfSamples, sampleRate)
    ##As per Nyquist-Shannon theorem, we need a sample rate of 40kHz
    samplesNeededFromShannon = np.ceil(nyquistRate*originalNumberOfSamples/sampleRate)
    # Calculate the indices of the evenly spaced samples, such that there are the sufficient samples to satisfy shanon's theorem, and we remove extra samples
    indices = np.round(np.linspace(0, originalNumberOfSamples- 1, int(samplesNeededFromShannon))).astype(int)
    # Select the samples using the calculated indices
    audioData = audioData[indices]
    #We need the number of samples to satisfy Nyquist-Shannon theorem's criteria but also be of the form 2^N for our FFT function
    zeroesNeededForPwtwo = int(2**np.ceil(np.log2(samplesNeededFromShannon)) - samplesNeededFromShannon)
    #Add trailing zeroes so signal is of the form 2^N, without affecting the result
    audioData = np.pad(audioData, (0, int(zeroesNeededForPwtwo)), mode='constant', constant_values=(0, 0)).tolist()
    return (audioData, originalNumberOfSamples)

def setUpNoisy():
    audioData, sampleRate = librosa.load('pinkpanther10.wav', sr=None)
    ##nyquist rate assuming frequency is band limited by what the human ear can hear, which is 20kHz
    nyquistRate = 40000
    originalNumberOfSamples = len(audioData)
    #add noise
    audioData = audioData + 2*np.random.randn(len(audioData))
    ##As per Nyquist-Shannon theorem, we need a sample rate of 40kHz
    samplesNeededFromShannon = np.ceil(nyquistRate*originalNumberOfSamples/sampleRate)
    # Calculate the indices of the evenly spaced samples, such that there are the sufficient samples to satisfy shanon's theorem, and we remove extra samples
    indices = np.round(np.linspace(0, originalNumberOfSamples- 1, int(samplesNeededFromShannon))).astype(int)
    # Select the samples using the calculated indices
    audioData = audioData[indices]
    #We need the number of samples to satisfy Nyquist-Shannon theorem's criteria but also be of the form 2^N for our FFT function
    zeroesNeededForPwtwo = int(2**np.ceil(np.log2(samplesNeededFromShannon)) - samplesNeededFromShannon)
    #Add trailing zeroes so signal is of the form 2^N, without affecting the result
    audioData = np.pad(audioData, (0, int(zeroesNeededForPwtwo)), mode='constant', constant_values=(0, 0)).tolist()
    return (audioData, originalNumberOfSamples)

def fft(f):
    n = len(f) 
    k = np.arange(n) 
    if n == 1: 
        return f 
    else: 
        Dfe = fft(f[::2]) 
        Dfo = fft(f[1::2]) 
        Dfe = np.hstack((Dfe,Dfe)) 
        Dfo = np.hstack((Dfo,Dfo)) 
        return Dfe + np.exp(-2*np.pi*1j*k/n)*Dfo
def FFT(f):
    return fft(f)/np.sqrt(len(f))

def denoiseSignal(frequencyData, lowPassFilter, highPassFilter):
    PSD = [aHat*np.conj(aHat)/len(frequencyData) for aHat in frequencyData]
    for i in range(len(PSD)):
        power = PSD[i]
        if power > highPassFilter or power < lowPassFilter:
            frequencyData[i] = 0
    return frequencyData

audioData, originalNumberOfSamples = setUpAudioData()
noisyAudioData, noisySamples = setUpNoisy()
x= np.arange(0, len(audioData))
noisyFreqs = FFT(noisyAudioData)
originalFreqs = FFT(audioData)
denoisedFreq = denoiseSignal(noisyFreqs, 1, 3)

figure, axis = plt.subplots(4, 1) 
axis[0].plot(x, noisyAudioData)
axis[1].plot(x, noisyFreqs)
axis[2].plot(x, originalFreqs)
axis[3].plot(x, denoisedFreq)
plt.show()