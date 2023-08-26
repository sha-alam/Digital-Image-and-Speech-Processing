import librosa
import numpy as np
import matplotlib.pyplot as plt

# Read the speech signal (replace 'sound.mp3' with your audio file)
speech_signal, fs = librosa.load('sound.mp3', sr=None)

# Compute short-time energy and zero-crossing rate using consistent frame length
frame_length = int(0.02 * fs)  # 20 ms frame length
hop_length = frame_length // 2  # 50% overlap

# Calculate squared signal for energy calculation
squared_signal = speech_signal ** 2

# Calculate energy for each frame
energy_frames = np.convolve(squared_signal, np.ones(frame_length), mode='valid')

# Calculate zero-crossing rate for each frame
zc_rate_frames = librosa.feature.zero_crossing_rate(y=speech_signal, frame_length=frame_length, hop_length=hop_length)[0]

# Resize zc_rate_frames to match the length of energy_frames
zc_rate_frames_resized = np.resize(zc_rate_frames, len(energy_frames))

# Compute thresholds
energy_threshold = 0.1 * np.max(energy_frames)
zc_threshold = 0.05 * np.max(zc_rate_frames_resized)

# Classify regions
voiced_regions = np.logical_and(energy_frames > energy_threshold, zc_rate_frames_resized > zc_threshold)
unvoiced_regions = np.logical_and(energy_frames > energy_threshold, zc_rate_frames_resized <= zc_threshold)
silence_regions = energy_frames <= energy_threshold

# Visualization
time = np.arange(len(speech_signal)) / fs

plt.subplot(3, 1, 1)
plt.plot(time, speech_signal)
plt.title('Original Speech Signal')

plt.subplot(3, 1, 2)
plt.plot(time[:len(voiced_regions)], voiced_regions, 'r', time[:len(unvoiced_regions)], unvoiced_regions, 'g', time[:len(silence_regions)], silence_regions, 'b')
plt.title('Voiced (red), Unvoiced (green), Silence (blue)')

plt.subplot(3, 1, 3)
plt.specgram(speech_signal, Fs=fs, NFFT=frame_length, noverlap=frame_length // 2)
plt.title('Spectrogram')

plt.tight_layout()
plt.show()