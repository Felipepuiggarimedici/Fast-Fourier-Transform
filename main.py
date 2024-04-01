import numpy as np
from matplotlib import pyplot as plt 
from scipy.io import wavfile
from SetUpAudioData import setUpAudioData
from FFT import fft
from Denoiser import denoiseSignal

audioData, stopTime, samplesNeededFromShannon = setUpAudioData("Beethoven_Symphony_No._5_Movement_2,_La_Folia_Variation_(measures_166_-_183).wav")
noisyAudioData = setUpAudioData("Beethoven_Symphony_No._5_Movement_2,_La_Folia_Variation_(measures_166_-_183).wav", noise = True)[0]
t= np.linspace(0, stopTime, samplesNeededFromShannon)

noisyFreqs = fft(noisyAudioData.copy())
#the low pass filter was determined by examining the PSD of the noisy version and the original one, so as to make a sensible filter. In reality it could be determined via analysis of the signal and the sources of the noise (to understand the pressure they create)
denoisedFreqs = denoiseSignal(noisyFreqs, 1.5e-09)
denoisedSignal = fft(denoisedFreqs.copy(), True)

reconstructNoisy = np.array(noisyAudioData[0:samplesNeededFromShannon], dtype=np.float32)
wavfile.write('noisywav.wav', int(samplesNeededFromShannon/stopTime), reconstructNoisy)
reconstructDenoised = np.array(denoisedSignal[0:samplesNeededFromShannon], dtype=np.float32)
wavfile.write('denoised.wav', int(samplesNeededFromShannon/stopTime), reconstructDenoised)

plt.style.use('dark_background')
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