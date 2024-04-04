import cv2 as cv

# Load the image
image = cv.imread('images/out.png')
print(image.shape)
image = cv.resize(image, (4032//10, 3040//10))

# Convert the image to HSV color space
hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)

# Split the HSV image into individual channels
h, s, v = cv.split(hsv_image)

# Apply thresholding on the saturation (S) channel
_, thresholded = cv.threshold(s, 100, 255, cv.THRESH_BINARY_INV)

# Display the thresholded image
cv.imshow('Thresholded Image', thresholded)
cv.imshow("Saturation Channel", s)
cv.waitKey(0)
cv.destroyAllWindows()