import numpy as np
import matplotlib.pyplot as plt
import pywt
import librosa

# Load a local audio file
file_path = 'sound.mp3'
speech_signal, fs = librosa.load(file_path, sr=None)

# Define wavelet and decomposition level
wavelet = 'db4'  # Daubechies 4 wavelet
level = 5        # Multilevel decomposition level

# Perform multilevel wavelet decomposition
coeffs = pywt.wavedec(speech_signal, wavelet, level=level)

# Plot original and decomposed signals
plt.figure(figsize=(10, 8))

plt.subplot(level + 2, 1, 1)
plt.plot(np.arange(len(speech_signal)) / fs, speech_signal)
plt.title('Original Speech Signal')

for i in range(level + 1):
    plt.subplot(level + 2, 1, i + 2)
    plt.plot(np.arange(len(coeffs[i])) / fs, coeffs[i])
    plt.title(f'Level {i} Coefficients')

plt.tight_layout()
plt.show()
