import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('fruit.jpg')
# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Fourier Transform
fft_image = np.fft.fft2(gray_image)
fft_shifted = np.fft.fftshift(fft_image)

# Define filters for high-frequency emphasis, low-pass Gaussian, and high-pass
D0 = 50  # Cut-off frequency for Gaussian and high-pass filters
c = 2    # High-frequency emphasis parameter
# Corrected calculation for H_high_emphasis
frequency_values = np.linspace(1, 1.5 * D0, fft_shifted.shape[0])[:, np.newaxis]
H_high_emphasis = 1 + c * (1 - np.exp(-(0.5 * (D0 / frequency_values)) ** 2))
H_low_pass = np.exp(-0.5 * ((np.linspace(-1, 1, fft_shifted.shape[0])[:, np.newaxis] * D0) ** 2))
H_high_pass = 1 - H_low_pass
# Apply filters in the frequency domain
enhanced_freq = H_high_emphasis * fft_shifted
smoothed_freq = H_low_pass * fft_shifted

# Calculate sharpened frequency domain
sharpened_freq = fft_shifted - smoothed_freq

# Perform Inverse Fourier Transform to obtain images
enhanced_image = np.fft.ifft2(np.fft.ifftshift(enhanced_freq))
smoothed_image = np.fft.ifft2(np.fft.ifftshift(smoothed_freq))
sharpened_image = np.fft.ifft2(np.fft.ifftshift(sharpened_freq))

# Display the results
plt.subplot(2, 4, 1)
plt.imshow(cv2.cvtColor(gray_image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')

plt.subplot(2, 4, 2)
plt.imshow(np.real(enhanced_image), cmap='gray')
plt.title('Enhanced Image')

plt.subplot(2, 4, 3)
plt.imshow(np.real(smoothed_image), cmap='gray')
plt.title('Smoothed Image')

plt.subplot(2, 4, 4)
plt.imshow(np.real(sharpened_image), cmap='gray')
plt.title('Sharpened Image')

# Display the frequency domain filters
plt.subplot(2, 4, 5)
plt.plot(H_high_emphasis)
plt.title('High-Frequency Emphasis Filter')

plt.subplot(2, 4, 6)
plt.plot(H_low_pass)
plt.title('Low-Pass Gaussian Filter')

plt.subplot(2, 4, 7)
plt.plot(H_high_pass)
plt.title('High-Pass Filter')

plt.show()
