import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the image
image = cv2.imread('Documentation/Images/image3.png')

# Check if image has loaded correctly
if image is None:
    print("Error: Image could not be read.")
    exit()

# Convert the image to the HSV color space
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Extract the saturation channel
saturation_channel = hsv_image[:, :, 1]
value_channel = hsv_image[:, :, 2]

# Apply thresholding on the saturation channel
# Here, we use a simple binary threshold with a threshold value of 128 (you can adjust this value)
_, thresholded_image = cv2.threshold(saturation_channel, 25, 255, cv2.THRESH_BINARY)
threshold_value = cv2.adaptiveThreshold(value_channel, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
combined_img = cv2.bitwise_and(thresholded_image, threshold_value)
cv2.imshow('thresholded_image', thresholded_image)
cv2.imshow('threshold_value', threshold_value)
cv2.imshow('combined', combined_img)
cv2.waitKey(0)

# Display and save the original image
plt.figure(figsize=(10, 5))

plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')

# Display and save the saturation channel
plt.subplot(1, 3, 2)
plt.title("Saturation Channel")
plt.imshow(saturation_channel, cmap='gray')
plt.axis('off')

# Display and save the thresholded image
plt.subplot(1, 3, 3)
plt.title("Thresholded Image")
plt.imshow(thresholded_image, cmap='gray')
plt.axis('off')

# Save the images
plt.savefig('thresholding_saturation.png', bbox_inches='tight')
plt.show()
