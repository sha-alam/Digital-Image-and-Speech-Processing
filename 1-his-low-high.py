import cv2
import matplotlib.pyplot as plt

input_image = cv2.imread('fruit.jpg', cv2.IMREAD_GRAYSCALE)
histogram = cv2.calcHist([input_image], [0], None, [256], [0, 256])
filtered_image = cv2.GaussianBlur(input_image, (5, 5), 3)  # Adjust kernel size and sigma as needed
low_pass_image = cv2.GaussianBlur(input_image, (5, 5), 3)  # Low pass filtered image
high_pass_image = cv2.subtract(input_image, low_pass_image)

plt.subplot(1,3,1)
plt.plot(histogram)
plt.title('Histogram')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')

plt.subplot(1,3,2)
plt.imshow(cv2.cvtColor(filtered_image, cv2.COLOR_BGR2RGB))
plt.title('Low Pass Filtered Image')
plt.axis('off')

plt.subplot(1,3,3)
plt.imshow(cv2.cvtColor(high_pass_image, cv2.COLOR_BGR2RGB))
plt.title('High Pass Image')
plt.axis('off')
plt.show()