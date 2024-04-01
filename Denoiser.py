import numpy as np

def denoiseSignal(frequencyData, lowPassFilter):
    frequencyData = frequencyData.copy()
    psd = [aHat*np.conj(aHat)/len(frequencyData) for aHat in frequencyData]
    for i in range(len(psd)):
        power = psd[i]
        if power < lowPassFilter:
            frequencyData[i] = 0
    return frequencyData