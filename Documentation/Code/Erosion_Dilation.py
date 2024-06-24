import cv2
import numpy as np
from matplotlib import pyplot as plt


def adaptive_otsu_thresholding(img, alt):
    assert img is not None, "Image could not be read, check with os.path.exists()"
    img_cols = img.shape[1]  # count number of pixel columns

    # Set size of blur filter relative to image size
    size_blur = int(img_cols / 100)
    if size_blur % 2 == 0:  # make sure size is odd
        size_blur += 1

    blur = cv2.GaussianBlur(img, (size_blur, size_blur), 0)

    # Calculate normalized histogram and its cumulative distribution function
    hist = cv2.calcHist([blur], [0], None, [256], [0, 256])
    hist_norm = hist.ravel() / hist.sum()
    Q = hist_norm.cumsum()
    bins = np.arange(256)

    # Set Otsu's method modification factor based on altitude
    if 5 < alt < 10:
        otsu_factor = 49 / 5 * alt - 48
    elif alt < 5:
        otsu_factor = 1
    else:
        otsu_factor = 50

    fn_min = np.inf
    thresh = -1
    for i in range(1, 256):
        p1, p2 = np.hsplit(hist_norm, [i])  # probabilities
        q1, q2 = Q[i], Q[255] - Q[i]  # cum sum of classes
        if q1 < 1.e-6 or q2 < 1.e-6:
            continue
        b1, b2 = np.hsplit(bins, [i])  # weights
        m1, m2 = np.sum(p1 * b1) / q1, np.sum(p2 * b2) / q2
        v1, v2 = np.sum(((b1 - m1)**2) * p1) / q1, np.sum(((b2 - m2)**2) * p2) / q2
        fn = v1 * q1 + v2 * q2 * otsu_factor  # minimization function adjusted
        if fn < fn_min:
            fn_min = fn
            thresh = i

    # Use the calculated threshold to convert the image to binary
    ret, th_adj = cv2.threshold(blur, thresh, 255, cv2.THRESH_BINARY)

    # Morphological operations to improve image
    size_dilation = int(img_cols / 100)
    dilation = cv2.dilate(th_adj, np.ones((size_dilation, size_dilation), np.uint8), iterations=1)
    size_erode = int(img_cols / 30)
    erosion = cv2.erode(dilation, np.ones((size_erode, size_erode), np.uint8), iterations=1)
    size_dilation_2 = int(img_cols / 38)
    dilation_2 = cv2.dilate(erosion, np.ones((size_dilation_2, size_dilation_2), np.uint8), iterations=1)

    return th_adj, dilation_2  # return the final image for further processing

# Load the image in grayscale mode
image_path = 'Documentation/Images/no_circles.png'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Check if image has loaded correctly
if image is None:
    print("Error: Image could not be read.")
    exit()

# Perform adaptive Otsu thresholding with altitude information
altitude = 8  # Example altitude value in meters
processed_image, final_image = adaptive_otsu_thresholding(image, altitude)

# Perform additional erosion and dilation
colums_img = image.shape[1]
size_kernel = int(colums_img/38)
erosion = cv2.erode(processed_image, np.ones((size_kernel, size_kernel), np.uint8), iterations=1)
dilation = cv2.dilate(erosion, np.ones((size_kernel, size_kernel), np.uint8), iterations=1)

# Save the original image
plt.figure(figsize=(10, 10))
plt.imshow(processed_image, cmap='gray')
plt.axis('off')  # Hides the axis
plt.savefig('Documentation/Images/finished/erosion_dialtion_og_img.png', bbox_inches='tight')

# Save the erosion image
plt.figure(figsize=(10, 10))
plt.imshow(erosion, cmap='gray')
plt.axis('off')  # Hides the axis
plt.savefig('Documentation/Images/finished/erosion_dialtion_erosion.png', bbox_inches='tight')

# Save the dilation image
plt.figure(figsize=(10, 10))
plt.imshow(dilation, cmap='gray')
plt.axis('off')  # Hides the axis
plt.savefig('Documentation/Images/finished/erosion_dialtion_dilation.png', bbox_inches='tight')

# # Save the final processed image
# plt.figure(figsize=(10, 10))
# plt.imshow(final_image, cmap='gray')
# plt.axis('off')  # Hides the axis
# plt.savefig('Documentation/Images/finished/erosion_dialtion_final.png', bbox_inches='tight')
