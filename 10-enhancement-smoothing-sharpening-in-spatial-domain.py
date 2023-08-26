import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('fruit.jpg')

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply gamma correction for enhancement
gamma = 1.5  # Adjust the gamma value as needed
enhanced_image = np.power(gray_image / 255.0, gamma)
enhanced_image = np.uint8(enhanced_image * 255)

# Apply Gaussian blur for smoothing
sigma = 2.0  # Adjust the standard deviation as needed
smoothed_image = cv2.GaussianBlur(enhanced_image, (0, 0), sigma)

# Apply Laplacian sharpening for sharpening
sharpened_image = enhanced_image + 0.5 * (enhanced_image - smoothed_image)

# Display the results
plt.subplot(1, 4, 1)
plt.imshow(cv2.cvtColor(gray_image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')

plt.subplot(1, 4, 2)
plt.imshow(enhanced_image, cmap='gray')
plt.title('Enhanced Image')

plt.subplot(1, 4, 3)
plt.imshow(smoothed_image, cmap='gray')
plt.title('Smoothed Image')

plt.subplot(1, 4, 4)
plt.imshow(sharpened_image, cmap='gray')
plt.title('Sharpened Image')

plt.show()
