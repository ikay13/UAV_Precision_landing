import cv2
import numpy as np
from matplotlib import pyplot as plt

def checkIfSquare(cnt, approx_poly, altitude, image_centerX, image_centerY, size_square, image_width_px, cam_hfov):
    """Check if the contour is a square by checking the angles and line lengths
    if any check fails, return False and continue to the next contour"""
    if len(approx_poly) != 4:
        print("len(approx_poly): ", len(approx_poly))
        return False # Not 4 corners
    tolerance = 5 #Tolerance in meters (+/-)
    altitude_with_tol = [altitude+tolerance, altitude-tolerance]

    print("contour area: ", cv2.contourArea(cnt))
    if not (100 < cv2.contourArea(cnt)):
        print("Contour area too small")
        return False # Not the right size
    

    # Calculate the difference between the lengths of the lines
    line_lengths = []
    for lineIdx in range(len(approx_poly)-1):
        line = approx_poly[lineIdx] - approx_poly[(lineIdx+1)]
        line_length = np.linalg.norm(line)
        line_lengths.append(line_length)

    max_line_length = max(line_lengths)
    min_line_length = min(line_lengths)

    # Ratio of longest line length to the difference between the longest and shortest line length
    lineDiffRatio = (max_line_length - min_line_length) / max_line_length 

    if lineDiffRatio > 0.5:
        print("lineDiffRatio: ", lineDiffRatio)
        return False # Line lengths are too different
    

    # Calculate the angles between the lines
    angles = []
    for cornerIdx in range(len(approx_poly)):
        prevCorner = approx_poly[cornerIdx-1]
        currentCorner = approx_poly[cornerIdx]
        nextCorner = approx_poly[(cornerIdx+1) % len(approx_poly)]
        
        prevVector = prevCorner - currentCorner
        prevVector = np.squeeze(prevVector)
        nextVector = nextCorner - currentCorner
        nextVector =  np.squeeze(nextVector)
        
        prevVectorLength = np.linalg.norm(prevVector)
        nextVectorLength = np.linalg.norm(nextVector)
        
        dotProduct = np.dot(prevVector, nextVector)
        angle = np.arccos(dotProduct / (prevVectorLength * nextVectorLength))
        
        angles.append(angle)
    
    max_angle = max(angles)
    min_angle = min(angles)
    angle_diff = max_angle - min_angle # This is the maximum difference between the angles
    


    # Calculate the distance from the center of the contour to the middle of the image
    # This is imprortant as the shape of the contour is more distorted towards the edges of the image
    # center of contour
    M = cv2.moments(cnt)
    cX = int(M["m10"] / M["m00"]+1e-5)
    cY = int(M["m01"] / M["m00"]+1e-5)

    # get distance from center (1 is the edge, 0 the center)
    distX = abs(cX - image_centerX)
    distY = abs(cY - image_centerY)
    relative_distance = np.sqrt(distX**2 + distY**2)
    #As the angles are more distored towards the edges of the image, the acceptable angle increases with distance from the center
    max_accepted_angle = 0.3 * relative_distance + 0.1

    if angle_diff > max_accepted_angle:
        print("angle_diff: ", angle_diff)
        return False # Angles are too different
    
    # Calculate area of contour and compare to area of bounding box
    area_cnt = cv2.contourArea(cnt)
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    area_box = cv2.contourArea(box)

    areaRatio = area_cnt / area_box
    if areaRatio < 0.8:
        print("areaRatio: ", areaRatio)
        return False # Area of contour is too small compared to the bounding box

    return True # All checks passed, the object is a square

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
    #size_dilation = int(img_cols / 100)
    #dilation = cv2.dilate(th_adj, np.ones((size_dilation, size_dilation), np.uint8), iterations=1)
    size_erode = int(img_cols / 70)
    erosion = cv2.erode(th_adj, np.ones((size_erode, size_erode), np.uint8), iterations=1)
    size_dilation_2 = int(img_cols / 70)
    dilation_2 = cv2.dilate(erosion, np.ones((size_dilation_2, size_dilation_2), np.uint8), iterations=1)

    return dilation_2, ret # return the final image for further processing

# Load the image in grayscale mode
image_path = 'Documentation/Images/finished/watertesting/original_12_70s.png'
#Read as color image not grayscale
image = cv2.imread(image_path)

# Check if image has loaded correctly
if image is None:
    print("Error: Image could not be read.")
    exit()

# Perform adaptive Otsu thresholding with altitude information
# altitude = 10  # Example altitude value in meters
# processed_image, ret = adaptive_otsu_thresholding(image, altitude)
# cv2.imshow('processed_image', processed_image)
# cv2.waitKey(0)
#Change to color image



#Load processed image from file
#processed_image = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2BGR)
processed_image = cv2.imread('Documentation/Images/finished/watertesting/threshold_12_70_only_thr.png', cv2.IMREAD_GRAYSCALE)
contour = cv2.findContours(processed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contour = contour[0] if len(contour) == 2 else contour[1]
#image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
for c in contour:
    if cv2.contourArea(c) < 100:
        print("Contour area too small2")
        continue
    #Move contour slightly  to the top right
    c = c + np.array([42, -12])
    approx_temp = cv2.approxPolyDP(c, 0.05*cv2.arcLength(c, True), True)
    #Draw the contour
    cv2.drawContours(image, [approx_temp], 0, (0, 0, 0), 3)

    


# For the original image
plt.figure(figsize=(10, 10))
plt.imshow(image, cmap='gray')
plt.axis('off')  # Hides the axis
plt.savefig('Documentation/Images/finished/watertesting/original_12_70s_w_contours.png', bbox_inches='tight')

# # For the thresholded image
# plt.figure(figsize=(5, 5))
# plt.imshow(processed_image, cmap='gray')
# plt.axis('off')  # Hides the axis
# plt.savefig('Documentation/Images/finished/contour_thr.png', bbox_inches='tight')

# # Calculate histogram
# hist = cv2.calcHist([image], [0], None, [256], [0,256])

# # For the histogram with Otsu's threshold
# plt.figure(figsize=(15, 7))
# plt.grid(True, linestyle='-', linewidth=1.5)
# #plt.title("Histogram", fontsize=26)
# plt.xlabel("Intensity Value", fontsize=26)
# plt.ylabel("Number of Occurrences", fontsize=26)
# plt.tick_params(axis='both', which='major', labelsize=24)
# plt.plot(hist, color='black', linewidth=2.5)
# plt.axvline(x=ret, color='black', linestyle='--', label='Otsu Threshold: {:.2f}'.format(ret), linewidth=2.5)
# plt.legend(loc='upper right', fontsize=22)
# plt.savefig('Documentation/Images/finished/otsu_modified_histogram.png', bbox_inches='tight')
# plt.show()
