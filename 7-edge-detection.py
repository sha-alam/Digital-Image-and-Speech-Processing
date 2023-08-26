import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('pic1.jpg')
# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# Apply different edge detection operators
sobel_edges = cv2.Sobel(gray_image, cv2.CV_64F, 1, 1, ksize=3)
scharr_edges = cv2.Scharr(gray_image, cv2.CV_64F, 1, 0)
canny_edges = cv2.Canny(gray_image, threshold1=100, threshold2=200)
# Display the results
plt.subplot(1, 4, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')

plt.subplot(1, 4, 2)
plt.imshow(np.abs(sobel_edges), cmap='gray')
plt.title('Sobel Edges')
plt.subplot(1, 4, 3)
plt.imshow(np.abs(scharr_edges), cmap='gray')
plt.title('Scharr Edges')
plt.subplot(1, 4, 4)
plt.imshow(canny_edges, cmap='gray')
plt.title('Canny Edges')
plt.show()
