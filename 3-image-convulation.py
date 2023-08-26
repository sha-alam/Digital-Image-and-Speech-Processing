import cv2
import numpy as np
import matplotlib.pyplot as plt
# Load the image
image = cv2.imread('fruit.jpg', cv2.IMREAD_GRAYSCALE)  # Replace with your image file

# Define the 3x3 mask
mask = np.array([[1, 1, 1],
                 [1, -8, 1],
                 [1, 1, 1]])
# Perform convolution
convolved_image = cv2.filter2D(image, cv2.CV_64F, mask)
# Display the original and convolved images
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')

plt.subplot(1, 2, 2)
plt.imshow(convolved_image, cmap='gray')
plt.title('Convolved Image')

plt.show()