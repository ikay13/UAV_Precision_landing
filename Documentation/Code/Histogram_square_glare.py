import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter

def process_image(image_path):
    # Load the image in grayscale mode
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Check if image has loaded correctly
    if image is None:
        print(f"Error: Image at {image_path} could not be read.")
        return None

    # Calculate histogram
    hist = cv2.calcHist([image], [0], None, [256], [0,256])

    # Apply Otsu's thresholding
    img_cols = image.shape[1]  # count number of pixel columns

    # Set size of blur filter relative to image size
    size_blur = int(img_cols / 100)
    if size_blur % 2 == 0:  # make sure size is odd
        size_blur += 1

    blur = cv2.GaussianBlur(image, (size_blur, size_blur), 0)
    ret, _ = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return hist, ret

def plot_histogram(hist, threshold, save_path, with_threshold=False):
    # Configure plot to mimic MATLAB's appearance
    plt.figure(figsize=(15, 7))
    plt.grid(True, linestyle='-', linewidth=1.5)
    plt.xlabel("Intensity Value", fontsize=26)
    plt.ylabel("Number of Occurrences", fontsize=26)
    plt.tick_params(axis='both', which='major', labelsize=24)

    # Setting up custom ScalarFormatter for scientific notation
    formatter = ScalarFormatter(useMathText=True)  # Use math text for consistency in font style
    formatter.set_powerlimits((-1, 1))  # Limits for switching to scientific notation
    formatter.set_scientific(True)  # Enable scientific notation
    plt.gca().yaxis.set_major_formatter(formatter)  # Apply formatter to the y-axis

    plt.plot(hist, color='black', linewidth=2.5)
    if with_threshold:
        plt.axvline(x=threshold, color='black', linestyle='--', label='Otsu Threshold: {:.2f}'.format(threshold), linewidth=2.5)
        plt.legend(loc='upper right', fontsize=22)
    # Apply the updated font size for scientific notation separately
    plt.gcf().axes[0].yaxis.get_offset_text().set_fontsize(24)  # Adjust this value as needed
    plt.savefig(save_path, bbox_inches='tight')

# Process first image
hist1, threshold1 = process_image('Documentation/Images/finished/watertesting/original_12_70_glare.png')
if hist1 is not None:
    plot_histogram(hist1, threshold1, 'Documentation/Images/finished/watertesting/original_12_70_glare_hist.png')

# Process second image
hist2, threshold2 = process_image('Documentation/Images/finished/watertesting/original_12_70_square.png')
if hist2 is not None:
    plot_histogram(hist2, threshold2, 'Documentation/Images/finished/watertesting/original_12_70_square_hist.png')
