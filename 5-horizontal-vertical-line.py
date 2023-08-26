import cv2
import numpy as np
import matplotlib.pyplot as plt
# Load the image
image = cv2.imread('line2.jpg')
# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# Apply Canny edge detection
edges = cv2.Canny(gray_image, threshold1=30, threshold2=100)
# Find horizontal and vertical lines using Hough Transform
lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=100, maxLineGap=5)
# Create a copy of the original image to draw lines on
line_image = image.copy()
# Draw the detected lines on the image
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
# Display the images
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(line_image, cv2.COLOR_BGR2RGB))
plt.title('Image with Detected Lines')
plt.show()
