import cv2
import numpy as np
from matplotlib import pyplot as plt

def process_image(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Check if image has loaded correctly
    if image is None:
        print(f"Error: Image {image_path} could not be read.")
        return None, None, None

    # Convert the image to the HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Extract the saturation channel
    saturation_channel = hsv_image[:, :, 1]

    # Apply a threshold to the saturation channel
    _, thresholded_image = cv2.threshold(saturation_channel, 127, 255, cv2.THRESH_BINARY)

    return image, saturation_channel, thresholded_image

# Input images
image_paths = ['Documentation/Images/closed.png', 'Documentation/Images/semiclsoed.png', 'Documentation/Images/open.png']

# Process images
images = [process_image(image_path) for image_path in image_paths]

# Check if all images loaded correctly
if any(image_set is None for image_set in images):
    print("Error: One or more images could not be processed.")
    exit()

# Create the 3x4 grid for better comparison (3 rows for images, 4 columns including filenames)
fig, axes = plt.subplots(3, 4, figsize=(20, 15))

for i, (image_path, (original, saturation, thresholded)) in enumerate(zip(image_paths, images)):
    # Display filename
    axes[i, 0].text(0.5, 0.5, image_path, rotation=90, verticalalignment='center', horizontalalignment='center')
    axes[i, 0].axis('off')
    
    # Display original image
    axes[i, 1].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    axes[i, 1].set_title(f"Original Image {i+1}")
    axes[i, 1].axis('off')

    # Display saturation channel
    axes[i, 2].imshow(saturation, cmap='gray')
    axes[i, 2].set_title(f"Saturation Channel {i+1}")
    axes[i, 2].axis('off')

    # Display thresholded image
    axes[i, 3].imshow(thresholded, cmap='gray')
    axes[i, 3].set_title(f"Thresholded Image {i+1}")
    axes[i, 3].axis('off')

# Save the images
plt.tight_layout()
plt.savefig('comparison_grid.png', bbox_inches='tight')
plt.show()
