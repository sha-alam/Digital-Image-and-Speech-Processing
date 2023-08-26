import cv2
import numpy as np
from skimage import measure

# Load the image
image = cv2.imread('rice1.jpg', cv2.IMREAD_GRAYSCALE)
# Convert to binary image using thresholding
_, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# Perform connected component analysis
labels = measure.label(binary_image)
props = measure.regionprops(labels)

num_rice_grains = len(props)
area_range = (min(prop.area for prop in props), max(prop.area for prop in props))
avg_major_axis = np.mean([prop.major_axis_length for prop in props])
avg_perimeter = np.mean([prop.perimeter for prop in props])

print(f'Number of rice grains: {num_rice_grains}')
print(f'Rice grain area range: {area_range[0]} - {area_range[1]} pixels')
print(f'Average major axis length: {avg_major_axis:.2f} pixels')
print(f'Average perimeter: {avg_perimeter:.2f} pixels')

# Display the binary image
cv2.imshow('Binary Image with Rice Grains', binary_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
