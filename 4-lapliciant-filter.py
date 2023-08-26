import cv2
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('fruit.jpg', cv2.IMREAD_GRAYSCALE)  # Replace with your image file
# Apply Laplacian filter
laplacian_image = cv2.Laplacian(image, cv2.CV_64F)

# Display the original and Laplacian-filtered images
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')

plt.subplot(1, 2, 2)
plt.imshow(laplacian_image, cmap='gray')
plt.title('Laplacian Filtered Image')

plt.show()
