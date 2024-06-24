import cv2
import numpy as np
from matplotlib import pyplot as plt

def draw_edges_on_white_background(edges1, edges2):
    # Create a blank image with a white background
    white_background = np.full_like(edges1, 255)

    # Draw edges from the first image in light gray (127)
    white_background[edges1 > 100] = 100

    # Draw edges from the second image in dark gray (0)
    white_background[edges2 > 100] = 0

    return white_background

def extract_center_section(image, diameter, offset_x, offset_y):
    h, w = image.shape
    center_x, center_y = w // 2, h // 2
    radius = diameter // 2

    x_start = center_x - radius + offset_x
    y_start = center_y - radius + offset_y
    x_end = center_x + radius + offset_x
    y_end = center_y + radius + offset_y

    return image[y_start:y_end, x_start:x_end]

# Load the images in grayscale mode
image_path1 = 'Documentation/Images/finished/watertesting/no_inner_circle_low_canny.png'
image_path2 = 'Documentation/Images/finished/watertesting/no_inner_circle_low_canny_next.png'
edges1 = cv2.imread(image_path1, cv2.IMREAD_GRAYSCALE)
edges2 = cv2.imread(image_path2, cv2.IMREAD_GRAYSCALE)

# Check if images have loaded correctly
if edges1 is None or edges2 is None:
    print("Error: One or both images could not be read.")
    exit()

# Overlay the edges on a white background
combined_image = draw_edges_on_white_background(edges1, edges2)

# Extract the center section with a diameter of 50px and offset it
center_section = extract_center_section(combined_image, 150, 35, -70)

# Save the combined image
plt.figure(figsize=(10, 10))
plt.imshow(center_section, cmap='gray')
plt.axis('off')  # Hides the axis
plt.show()

# plt.savefig('Documentation/Images/finished/watertesting/no_inner_circle_low_canny_combined_center.png', bbox_inches='tight')
