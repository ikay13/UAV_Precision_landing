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

# Input images (excluding 'closed.png')
image_paths = ['Documentation/Images/semiclsoed.png', 'Documentation/Images/open.png', '/home/mathis_ros/Documents/Images/manual_landing_raw.png']

# Process images
images = [process_image(image_path) for image_path in image_paths]

# Check if all images loaded correctly
if any(image_set is None for image_set in images):
    print("Error: One or more images could not be processed.")
    exit()

# Save images separately
for i, (image_path, (original, saturation, thresholded)) in enumerate(zip(image_paths, images)):
    # Convert original image to RGB for saving with matplotlib
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

    # Save original image
    plt.figure(figsize=(6, 6))
    plt.imshow(original_rgb)
    #plt.title(f"Original Image {i+1}")
    plt.axis('off')
    plt.savefig(f'Documentation/Images/finished/Saturation/original_image_{i+1}.png', bbox_inches='tight')
    plt.close()

    # Save saturation channel image
    plt.figure(figsize=(6, 6))
    plt.imshow(saturation, cmap='gray')
    #plt.title(f"Saturation Channel {i+1}")
    plt.axis('off')
    plt.savefig(f'Documentation/Images/finished/Saturation/saturation_channel_{i+1}.png', bbox_inches='tight')
    plt.close()


print("Images saved successfully.")
