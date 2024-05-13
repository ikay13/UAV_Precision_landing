import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter

# Load the image in grayscale mode
image = cv2.imread('Documentation/Images/no_circles.png', cv2.IMREAD_GRAYSCALE)

# Check if image has loaded correctly
if image is None:
    print("Error: Image could not be read.")
    exit()

# Calculate histogram
hist = cv2.calcHist([image], [0], None, [256], [0,256])

# Apply Otsu's thresholding
img_cols = image.shape[1]  # count number of pixel columns

# Set size of blur filter relative to image size
size_blur = int(img_cols / 100)
if size_blur % 2 == 0:  # make sure size is odd
    size_blur += 1

blur = cv2.GaussianBlur(image, (size_blur, size_blur), 0)
ret, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Configure plot to mimic MATLAB's appearance
plt.figure(figsize=(15, 7))
plt.grid(True, linestyle='-', linewidth=1.5)
#plt.title("Histogram", fontsize=26)
plt.xlabel("Intensity Value", fontsize=26)
plt.ylabel("Number of Occurences", fontsize=26)
plt.tick_params(axis='both', which='major', labelsize=24)

# Setting up custom ScalarFormatter for scientific notation
formatter = ScalarFormatter(useMathText=True)  # Use math text for consistency in font style
formatter.set_powerlimits((-1,1))  # Limits for switching to scientific notation
formatter.set_scientific(True)  # Enable scientific notation
plt.gca().yaxis.set_major_formatter(formatter)  # Apply formatter to the y-axis

plt.plot(hist, color='black', linewidth=2.5)
plt.axvline(x=ret, color='black', linestyle='--', label='Otsu Threshold: {:.2f}'.format(ret), linewidth=2.5)
plt.legend(loc='upper right', fontsize=22)
# Apply the updated font size for scientific notation separately
plt.gcf().axes[0].yaxis.get_offset_text().set_fontsize(24)  # Adjust this value as needed
plt.savefig('Documentation/Images/finished/otsu_original_histogram.png', bbox_inches='tight')

######################################Same but without the threshold line############################################
# Configure plot to mimic MATLAB's appearance
plt.figure(figsize=(15, 7))
plt.grid(True, linestyle='-', linewidth=1.5)
#plt.title("Histogram", fontsize=26)
plt.xlabel("Intensity Value", fontsize=26)
plt.ylabel("Number of Occurences", fontsize=26)
plt.tick_params(axis='both', which='major', labelsize=24)

# Setting up custom ScalarFormatter for scientific notation
formatter = ScalarFormatter(useMathText=True)  # Use math text for consistency in font style
formatter.set_powerlimits((-1,1))  # Limits for switching to scientific notation
formatter.set_scientific(True)  # Enable scientific notation
plt.gca().yaxis.set_major_formatter(formatter)  # Apply formatter to the y-axis

plt.plot(hist, color='black', linewidth=2.5)
#plt.axvline(x=ret, color='black', linestyle='--', label='Otsu Threshold: {:.2f}'.format(ret), linewidth=2.5)
#plt.legend(loc='upper right', fontsize=22)
# Apply the updated font size for scientific notation separately
plt.gcf().axes[0].yaxis.get_offset_text().set_fontsize(24)  # Adjust this value as needed
plt.savefig('Documentation/Images/finished/histogram_no_threshold.png', bbox_inches='tight')
######################################################################################################################


# Show the original image and the thresholded image
# For the original image
plt.figure(figsize=(5, 5))
plt.imshow(image, cmap='gray')
plt.axis('off')  # Hides the axis
plt.savefig('Documentation/Images/finished/otsu_original_img.png', bbox_inches='tight')

# For the thresholded image
plt.figure(figsize=(5, 5))
plt.imshow(thresh, cmap='gray')
plt.axis('off')  # Hides the axis

plt.savefig('Documentation/Images/finished/otsu_original_threshold.png', bbox_inches='tight')

plt.show()
