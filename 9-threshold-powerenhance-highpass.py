import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('fruit.jpg')

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply thresholding
_, binary_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)

# Apply power-law transformation for contrast enhancement
gamma = 1.5  # Adjust the gamma value as needed
enhanced_image = np.power(gray_image / 255.0, gamma)
enhanced_image = np.uint8(enhanced_image * 255)

# Apply Gaussian blur to the image
blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)  # Adjust the kernel size as needed

# Calculate the high-pass image by subtracting the blurred image from the original
high_pass_image = gray_image - blurred_image

# Display the original and thresholded images
plt.subplot(1, 4, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')

plt.subplot(1, 4, 2)
plt.imshow(binary_image, cmap='gray')
plt.title('Thresholded Image')

plt.subplot(1, 4, 3)
plt.imshow(enhanced_image, cmap='gray')
plt.title('Enhanced Contrast Image')

plt.subplot(1, 4, 4)
plt.imshow(high_pass_image, cmap='gray')
plt.title('High Pass Image')

plt.show()
