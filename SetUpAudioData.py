import librosa
import numpy as np

def setUpAudioData(nameOfWavFile, noise = False):
    '''
    Receives the name of the wav file to be imported. If noise is to  be added, second parameter is true
    Returns the audio data, with padding, the length of the audio and the original samples without padding.
    '''
    audioData, sampleRate = librosa.load(nameOfWavFile, sr=None)
    ##nyquist rate assuming frequency is band limited by what the human ear can hear, which is 20kHz
    nyquistRate = 40000
    lengthOfAudio = len(audioData)/sampleRate
    originalNumberOfSamples = len(audioData)
    if noise:
        #add noise. Low amplitude as noise will be "lower" than original sampled sound, and it will follow normal distribution. In this case, the mean amplitude of added noise is of 0.02
        audioData = audioData + 0.01*np.random.randn(len(audioData))
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