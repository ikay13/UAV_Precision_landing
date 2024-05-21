import cv2
import numpy as np
from matplotlib import pyplot as plt

# Create a synthetic image with a circle
height, width = 200, 200
image = np.zeros((height, width), dtype=np.uint8)
center = (width // 2, height // 2)
radius = 50
cv2.circle(image, center, radius, 255, 2)

# Apply Hough Circle Transform to get the accumulator
def hough_circle_accumulator(image, min_radius, max_radius):
    rows, cols = image.shape
    accumulator = np.zeros((rows, cols, max_radius - min_radius), dtype=np.uint32)
    
    edges = cv2.Canny(image, 100, 200)
    
    for y in range(rows):
        for x in range(cols):
            if edges[y, x] > 0:
                for r in range(min_radius, max_radius):
                    for t in range(0, 360):
                        b = y - int(r * np.sin(np.deg2rad(t)))
                        a = x - int(r * np.cos(np.deg2rad(t)))
                        if 0 <= a < cols and 0 <= b < rows:
                            accumulator[b, a, r - min_radius] += 1
    
    return accumulator, edges

# Parameters for the Hough Circle Transform
min_radius = 40
max_radius = 60

# Get the accumulator and edge image
accumulator, edges = hough_circle_accumulator(image, min_radius, max_radius)

# Find the maximum value in the accumulator
max_accumulator = np.max(accumulator)
accumulator_image = np.max(accumulator, axis=2)

# Normalize the accumulator image for better visualization
accumulator_image = (accumulator_image / max_accumulator * 255).astype(np.uint8)

# Display the original image, edges, and accumulator image
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.title("Original Image with Circle")
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Edges (Canny)")
plt.imshow(edges, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Accumulator Space")
plt.imshow(accumulator_image, cmap='hot')
plt.axis('off')

plt.tight_layout()
plt.savefig('Documentation/Images/finished/accumulator_space.png', bbox_inches='tight')
plt.show()
