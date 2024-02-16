import cv2
from find_hough_circles import main

# Path to the input image
image_path = "pad.jpeg"

# Set optional parameters (if needed)
r_min = 10
r_max = 150
delta_r = 2
num_thetas = 150
bin_threshold = 0.5
min_edge_threshold = 50
max_edge_threshold = 150

# Call the main function of the Hough circle detection script
main(image_path)
