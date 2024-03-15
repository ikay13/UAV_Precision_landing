import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter
from coordinate_transform import calculate_size_in_px
def thresholding(img):
    """Threshold the image using Otsu's method and apply dilation and erosion to remove noise and close gaps in the landing pad"""
    #img = cv.imread(path_to_image, cv.IMREAD_GRAYSCALE) #read as grayscale
    assert img is not None, "file could not be read, check with os.path.exists()"

    img_cols = img.shape[1] #count number of pixel columns

    size_blur = int(img_cols / 100) #set size of blur filter relative to image size
    if size_blur % 2 == 0: #make sure size is odd
        size_blur += 1

    blur = cv.GaussianBlur(img,(size_blur,size_blur),0) 
    

    # find normalized_histogram, and its cumulative distribution function
    hist = cv.calcHist([blur],[0],None,[256],[0,256])
    hist_norm = hist.ravel()/hist.sum()

    #Calculate the optimal threshold using Otsu's method (modified to set different threshold value)
    ####Parameter########
    otsu_factor = 200 #a higher value will result in a higher threshold
    #####################
    Q = hist_norm.cumsum()
    bins = np.arange(256)
    fn_min = np.inf
    thresh = -1
    for i in range(1,256):
        p1,p2 = np.hsplit(hist_norm,[i]) # probabilities
        q1,q2 = Q[i],Q[255]-Q[i] # cum sum of classes
        if q1 < 1.e-6 or q2 < 1.e-6:
            continue
        b1,b2 = np.hsplit(bins,[i]) # weights
        # finding means and variances
        m1,m2 = np.sum(p1*b1)/q1, np.sum(p2*b2)/q2
        v1,v2 = np.sum(((b1-m1)**2)*p1)/q1,np.sum(((b2-m2)**2)*p2)/q2
        # calculates the minimization function
        fn = v1*q1 + v2*q2*otsu_factor #adjust to set different threshold
        if fn < fn_min:
            fn_min = fn
            thresh = i

    #Use this threshold to convert the image to binary
    ret, th_adj = cv.threshold(blur, thresh,255,cv.THRESH_BINARY)

    #dilate to close gaps and circles in landing pad
    size_dilation = int(img_cols / 100)
    dilation = cv.dilate(th_adj, np.ones((size_dilation,size_dilation),np.uint8), iterations=1)

    #erode to remove noise
    size_erode = int(img_cols / 30)
    erosion = cv.erode(dilation, np.ones((size_erode,size_erode),np.uint8), iterations=1)

    #dilate to restore size of landing pad
    size_dilation_2 = int(img_cols / 38)
    dilation_2 = cv.dilate(erosion, np.ones((size_dilation_2,size_dilation_2),np.uint8), iterations=1)

    return dilation_2 #return the final image for further processing

    # plt.subplot(331), plt.plot(hist_norm), plt.xlabel('Pixel Value')
    # plt.axvline(x=thresh, color='r', linestyle='--')
    # plt.ylabel('Normalized Frequency'), plt.title('Histogram')

    # plt.subplot(332), plt.imshow(th_adj, 'gray'), plt.title('Otsu Thresholding adjusted')
    # plt.subplot(333), plt.imshow(otsu, 'gray'), plt.title('Otsu original')
    # plt.subplot(334), plt.imshow(blur, 'gray'), plt.title('blurred image')
    # plt.subplot(335), plt.imshow(img, 'gray'), plt.title('original image')
    # plt.subplot(336), plt.imshow(dilation, 'gray'), plt.title('dilated image')
    # plt.subplot(337), plt.imshow(erosion, 'gray'), plt.title('dilated eroded')
    # plt.subplot(338), plt.imshow(dilation_2, 'gray'), plt.title('dilated eroded dilated')

    # plt.show()
 

def checkIfSquare(cnt, approx_poly, altitude, image_centerX, image_centerY, size_square, image_width_px, cam_hfov):
    """Check if the contour is a square by checking the angles and line lengths
    if any check fails, return False and continue to the next contour"""
    if len(approx_poly) != 4:
        return False # Not 4 corners
    
    expected_size = calculate_size_in_px(altitude=altitude, size_object_m=size_square, cam_hfov=cam_hfov, image_width=image_width_px)
    expexted_area = expected_size**2
    tolerance = 0.5
    min_area = expexted_area * (1-tolerance)
    max_area = expexted_area * (1+tolerance)

    if not (min_area < cv.contourArea(cnt) < max_area):
        #print("expte_area: ", expexted_area, "contourArea: ", cv.contourArea(cnt))
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

    if lineDiffRatio > 0.15:
        #print("lineDiffRatio: ", lineDiffRatio)
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
    M = cv.moments(cnt)
    cX = int(M["m10"] / M["m00"]+1e-5)
    cY = int(M["m01"] / M["m00"]+1e-5)

    # get distance from center (1 is the edge, 0 the center)
    distX = abs(cX - image_centerX)
    distY = abs(cY - image_centerY)
    relative_distance = np.sqrt(distX**2 + distY**2)
    #As the angles are more distored towards the edges of the image, the acceptable angle increases with distance from the center
    max_accepted_angle = 0.3 * relative_distance + 0.1

    if angle_diff > max_accepted_angle:
        #print("angle_diff: ", angle_diff)
        return False # Angles are too different
    
    # Calculate area of contour and compare to area of bounding box
    area_cnt = cv.contourArea(cnt)
    rect = cv.minAreaRect(cnt)
    box = cv.boxPoints(rect)
    box = np.int0(box)
    area_box = cv.contourArea(box)

    areaRatio = area_cnt / area_box
    if areaRatio < 0.85:
        #print("areaRatio: ", areaRatio)
        return False # Area of contour is too small compared to the bounding box

    return True # All checks passed, the object is a square


def findContours(threshold_img, grayscale_img, altitude, size_square, cam_hfov):
    """Find contours in the thresholded image and filter using checkIfSquare function"""
    # Find contours and filter using threshold area
    cnts = cv.findContours(threshold_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    approx = None
    for c in cnts:
        approx_temp = cv.approxPolyDP(c, 0.1*cv.arcLength(c, True), True)

        # Get the center of the image
        image_centerX = grayscale_img.shape[1] // 2
        image_centerY = grayscale_img.shape[0] // 2
        image_width_px = grayscale_img.shape[1]
        bounding_box = checkIfSquare(c, approx_temp, altitude, image_centerX, image_centerY, size_square, image_width_px, cam_hfov)
        if bounding_box is not False:
            cv.drawContours(grayscale_img, [approx_temp], -1, (0, 255, 0), 2)    
            approx = approx_temp      

    # plt.subplot(131), plt.imshow(grayscale_img, 'gray'), plt.title('Original Image')
    # plt.subplot(132), plt.imshow(threshold_img, 'gray'), plt.title('Thresholded Image')
    # plt.show()
    disp_img = cv.hconcat([grayscale_img, threshold_img])
    cv.imshow('Grayscale and threshold', disp_img)
    cv.waitKey(1)
    return approx #return the contour of the square or none if no square is found

def calculate_error_image (square_contour, img_width, img_height): #return error relative to the center of the image
    """Calculate the error in the x and y direction of the center of the square relative to the center of the image"""
    if square_contour is None:
        return None
    
    M = cv.moments(square_contour)
    cX = int(M["m10"] / (M["m00"]+1e-5))  # x-coordinate of the center of contour
    cY = int(M["m01"] / (M["m00"]+1e-5))  # y-coordinate of the center of contour
    error_xy = ((cX / img_width-0.5)*2, (cY / img_height-0.5)*-1.5)  # calculate relative error in x and y direction
    return error_xy
    

################################### 
#######Start of main program#######
###################################

def detect_square_main(frame, altitude, size_square, cam_hfov):
    """Main function to detect a square in the frame, return the error in the x and y direction if a square is found, else return None"""
    grayscale_img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    threshold_img = thresholding(grayscale_img)
    square_contour = findContours(threshold_img, grayscale_img,altitude, size_square, cam_hfov)
    if square_contour is not None: #Implement check to see if only one square is detected???????????????
        area_ratio = cv.contourArea(square_contour)/ (grayscale_img.shape[1] * grayscale_img.shape[0]) #ratio of area of contour to area of image
        
        error = calculate_error_image(square_contour, grayscale_img.shape[1], grayscale_img.shape[0])
        return error, area_ratio
    else:
        return None, None


def calculate_target_error(errors_xy):
    """Calculate the mean error in the x and y direction from a list of errors"""
    distances = np.linalg.norm(errors_xy, axis=1)
    errors_xy= np.mean(errors_xy, axis=0)
    standard_deviation = np.std(distances)
    #print("Standard deviation: ", standard_deviation)
    #implement logic to check if deviation is too high for one platform
    return errors_xy

def check_for_time(frame, altitude,duration,ratio_detected, size_square, cam_hfov):
    """check if a square is detected in the frame for 3 seconds, if it is, return coordinates, if not, return None"""
    #Use function variables to store values between function calls (only executed once)
    if check_for_time.start_time is None:
        check_for_time.start_time = perf_counter()
        check_for_time.errors_xy = []
        check_for_time.not_detected_cnt = 0
    
    err_square, _ = detect_square_main(frame, altitude, size_square, cam_hfov)
    if err_square is not None: #If a square is detected, add the error to the list
        check_for_time.errors_xy.append(err_square)
    else:
        check_for_time.not_detected_cnt += 1

    if perf_counter() - check_for_time.start_time > duration: #If 3 seconds have passed, check if a square was detected in more than half of the frames
        if len(check_for_time.errors_xy )/(check_for_time.not_detected_cnt+1e-5) > ratio_detected: #If more than half of the frames have a square, get target error
            calculated_target_err =  calculate_target_error(check_for_time.errors_xy)
            return calculated_target_err
        else:
            return False
    else:
        return None

    
    



























# altitude = 10
# cam = cv.VideoCapture("code_project/images/cut.mp4")

# while(True):
#    ret,frame = cam.read()
#    if ret:
#       # if video is still left continue creating images
#         #path_to_image = 'code_project/images/out.png'
#         #grayscale_img = cv.imread(path_to_image, cv.IMREAD_GRAYSCALE)
#         #threshold_img = thresholding(path_to_image)
#         frame = cv.resize(frame, (640, 480))
#         grayscale_img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
#         threshold_img = thresholding(grayscale_img)
#         square_contour = findContours(threshold_img, grayscale_img,altitude)
#         positions = calculatePositions(square_contour, grayscale_img.shape[1], grayscale_img.shape[0])
#         if positions is not None:
#             print("position_x: {:.2f} position_y: {:.2f}".format(positions[0][0], positions[0][1]))
#    else:
#       break

# cam.release()
# cv.destroyAllWindows()







