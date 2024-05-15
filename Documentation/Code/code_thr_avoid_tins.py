import cv2
import numpy as np

# Load the image
image = cv2.imread('Documentation/Images/manual_landing_raw.png')

# Convert to HSV color space
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define range for black color in HSV
lower_black = np.array([0, 0, 0])
upper_black = np.array([10, 10, 10])

# Create mask to extract black regions
mask = cv2.inRange(hsv, lower_black, upper_black)

# Apply the mask to the image
filtered_image = cv2.bitwise_and(image, image, mask=mask)

# Convert the filtered image to grayscale
gray = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2GRAY)

# Apply thresholding to get a binary image
_, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

# Apply Canny edge detection
edges = cv2.Canny(binary, 100, 200)

# Display the results
cv2.imshow('Original Image', image)
cv2.imshow('Filtered Image', filtered_image)
cv2.imshow('Binary Image', binary)
cv2.imshow('Edges', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
