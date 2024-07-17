import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter
from coordinate_transform import calculate_size_in_px, calculate_altitude
import diptest
from math import sqrt
def thresholding(img,alt):
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
    if 5 < alt < 10: #Between 5 and 10m gradually decrease otsu_factor from 50 to 1
        otsu_factor = 49/5*alt-48
    elif alt < 5: #below 5 meter normal otsu thresholding
        otsu_factor = 1
    else: #Keep factor constant above 10m
        otsu_factor = 50
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

    # #dilate to close gaps and circles in landing pad
    # size_dilation = int(img_cols / 100)
    # dilation = cv.dilate(th_adj, np.ones((size_dilation,size_dilation),np.uint8), iterations=1)

    #erode to remove noise
    size_erode = int(img_cols / 30)
    erosion = cv.erode(th_adj, np.ones((size_erode,size_erode),np.uint8), iterations=1)

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
        print("len(approx_poly): ", len(approx_poly))
        return False # Not 4 corners
    tolerance = 3.5 #Tolerance in meters (+/-)
    altitude_with_tol = [altitude+tolerance, altitude-tolerance]
    expected_sizes = []
    expected_sizes.append(calculate_size_in_px(altitude=altitude_with_tol[0], size_object_m=size_square, cam_hfov=cam_hfov, image_width=image_width_px))
    expected_sizes.append(calculate_size_in_px(altitude=altitude_with_tol[1], size_object_m=size_square, cam_hfov=cam_hfov, image_width=image_width_px))

    min_area = expected_sizes[0]**2
    max_area = expected_sizes[1]**2

    if not (min_area < cv.contourArea(cnt) < max_area):
        print("min_area: ", min_area, "max_area", max_area, "contourArea: ", cv.contourArea(cnt))
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

    if lineDiffRatio > 0.25:
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
        print("angle_diff: ", angle_diff)
        return False # Angles are too different
    
    # Calculate area of contour and compare to area of bounding box
    area_cnt = cv.contourArea(cnt)
    rect = cv.minAreaRect(cnt)
    box = cv.boxPoints(rect)
    box = np.int0(box)
    area_box = cv.contourArea(box)

    areaRatio = area_cnt / area_box
    if areaRatio < 0.75:
        print("areaRatio: ", areaRatio)
        return False # Area of contour is too small compared to the bounding box

    return True # All checks passed, the object is a square


def findContours(threshold_img, grayscale_img, altitude, size_square, cam_hfov):
    """Find contours in the thresholded image and filter using checkIfSquare function"""
    # Find contours and filter using threshold area
    cnts = cv.findContours(threshold_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    approx = []
    for c in cnts:
        approx_temp = cv.approxPolyDP(c, 0.1*cv.arcLength(c, True), True)

        # Get the center of the image
        image_centerX = grayscale_img.shape[1] // 2
        image_centerY = grayscale_img.shape[0] // 2
        image_width_px = grayscale_img.shape[1]
        bounding_box = checkIfSquare(c, approx_temp, altitude, image_centerX, image_centerY, size_square, image_width_px, cam_hfov)
        if bounding_box is not False:
            # cv.drawContours(grayscale_img, [approx_temp], -1, (0, 255, 0), 2)    
            approx.append(approx_temp)

    return approx #return the contour of the square or none if no square is found

def calculate_error_image (square_contour, img_width, img_height): #return error relative to the center of the image
    """Calculate the error in the x and y direction of the center of the square relative to the center of the image. X is ranging from -1 to 1 and y from -0.75 to 0.75"""
    if square_contour is None:
        return None
    
    M = cv.moments(square_contour)
    cX = int(M["m10"] / (M["m00"]+1e-5))  # x-coordinate of the center of contour
    cY = int(M["m01"] / (M["m00"]+1e-5))  # y-coordinate of the center of contour
    error_xy = ((cX / img_width-0.5)*2, (cY / img_height-0.5)*-1.5)  # calculate relative error in x and y direction
    return error_xy
    

def detect_square_main(frame, altitude, size_square, cam_hfov):
    """Main function to detect a square in the frame, return the error in the x and y direction if a square is found, else return None"""
    # print("Frame: {}, Alt: {}, Sq. size: {}, HFOV: {}".format(frame.shape, altitude,size_square,cam_hfov))
    error = []
    perimeter_max = 0
    # hsv_img = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    # h,saturation,v = cv.split(hsv_img)
    grayscale_img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # cv.imshow("saturation", saturation)
    # cv.waitKey(1)
    threshold_img = thresholding(grayscale_img, altitude)
    # cv.imshow("Disp1",frame)
    # cv.imshow("Disp2",threshold_img)
    square_contour = findContours(threshold_img, grayscale_img,altitude, size_square, cam_hfov)
    
    if square_contour != []: #At least one square is detected
        cv.drawContours(frame, square_contour, -1, (0, 0, 0), 2)
        for cnt in square_contour:
            error_tmp = calculate_error_image(cnt, grayscale_img.shape[1], grayscale_img.shape[0])
            error.append(error_tmp)
            perimeter = cv.arcLength(cnt, True)
            if perimeter > perimeter_max:
                perimeter_max = perimeter
                # print("Edge length is: {}.\n".format(perimeter/4))
        alt_from_contour = calculate_altitude(perimeter_max/4, cam_hfov, grayscale_img.shape[1], size_square)
        # print("Image_Alt: {}".format(alt_from_contour))
        return error, alt_from_contour, threshold_img
    else:
        return None, None, threshold_img


def calculate_target_error(errors_xy, frame):
    """Calculate the mean error in the x and y direction from a list of errors. If the data is bimodal, return the two most likely targets, else return the mean error."""
    calc_error_xy = []
    for error_set in errors_xy:
        size = np.array(error_set).shape
        if size[0] == 1: #One square detected
            calc_error_xy.append((error_set[0][0], error_set[0][1]))
        else: ##Two squares or more detected (most likely two so just take the first two)
            calc_error_xy.append((error_set[0][0], error_set[0][1]))
            calc_error_xy.append((error_set[1][0], error_set[1][1]))
    x_values = [coord[0] for coord in calc_error_xy]
    y_values = [coord[1] for coord in calc_error_xy]
    dip_x, pval_x = diptest.diptest(np.array(x_values))
    dip_y, pval_y = diptest.diptest(np.array(y_values))

    if pval_x < 0.05 or pval_y < 0.05 and len(calc_error_xy) > 1:
        #Data is bimodal
        #Seperate data into a AxA grid. A is the number of bins (length of the array)
        num_bins = len(calc_error_xy)
        num_occurences_xy = [[0 for _ in range(num_bins+1)] for _ in range(num_bins+1)]
        bins_x = np.linspace(min(x_values), max(x_values), num_bins)
        bins_y = np.linspace(min(y_values), max(y_values), num_bins)
        for coord in calc_error_xy:
            x_bin = np.digitize(coord[0], bins_x)
            y_bin = np.digitize(coord[1], bins_y)
            num_occurences_xy[x_bin][y_bin] += 1

        #Convert the data to an image
        max_value = max(max(sub_arr) for sub_arr in num_occurences_xy)
        adjusted_arr = [[int(value/max_value*255) for value in sub_arr] for sub_arr in num_occurences_xy]
        image_of_occurences = np.asarray(adjusted_arr, dtype=np.uint8) 
        #Blur the image to connect single close peaks
        kernel_size = num_bins//10 if num_bins//10%2 == 1 else num_bins//10+1 #Kernel size must be odd
        blurred_image = cv.GaussianBlur(image_of_occurences, (kernel_size,kernel_size), 0)
        new_max = max(max(sub_arr) for sub_arr in blurred_image)
        _, thr_img = cv.threshold(blurred_image, new_max*0.2, 255, cv.THRESH_BINARY)
        # copied_img = thr_img.copy()
        # copied_img = cv.cvtColor(copied_img, cv.COLOR_GRAY2BGR)
        # cv.imwrite("thresholded_img.png", thr_img)
        # cv.imwrite("blurred_img.png", blurred_image)
        # cv.imwrite("image_of_occurences.png", image_of_occurences)
        # print("max_value: ", max_value, "")
        
        #Find the contours of the image and get the center of the two peaks
        cnt = cv.findContours(thr_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        # cv.drawContours(copied_img, cnt[0], -1, (0, 255, 0), 2)#Draw contours
        # cv.imwrite("contoursbefore.png", copied_img)
        # print("contours: ", cnt)
        cnt = cnt[0] if len(cnt) == 2 else cnt[1]
        # print("contours after selection: ", cnt)
        # copied_img = thr_img.copy()
        # copied_img = cv.cvtColor(copied_img, cv.COLOR_GRAY2BGR)
        # cv.drawContours(copied_img, cnt, -1, (0, 255, 0), 2)#Draw contours
        # cv.imwrite("contoursafter.png", copied_img)

        if len(cnt) < 2: #Only one peak
            print("Not bimodal (only one peak Nr1)")
            x_mean = np.mean(x_values)
            y_mean = np.mean(y_values)
            return (x_mean, y_mean), False
        
        real_centers = []
        # copied_img = thr_img.copy()
        # copied_img = cv.cvtColor(copied_img, cv.COLOR_GRAY2BGR)
        for c in cnt if len(cnt) < 3 else cnt[0:1]:
            M = cv.moments(c)
            cx = int(M['m10']/(M['m00']+1e-5))
            cy = int(M['m01']/(M['m00']+1e-5))
            real_centers.append((bins_x[cx], bins_y[cy]))
            # cv.circle(copied_img, (cx, cy), 5, (0, 0, 255), -1)
        # print("real_centers: ", real_centers)
        # cv.imwrite("centers.png", copied_img)
        
        if len(real_centers) < 2: #Only one peak
            print("Not bimodal (only one peak)")
            x_mean = np.mean(x_values)
            y_mean = np.mean(y_values)
            return (x_mean, y_mean), False
        


        #Check if the two peaks are too close together to be two seperate platforms
        distance = np.linalg.norm(np.array(real_centers[0]) - np.array(real_centers[1]))
        min_distance = 0.5 #This is just realtive to the image. The two peaks should be at least 0.5 of the image apart
        if distance < min_distance:
            #The two peaks are too close together, they should be at least half the image apart
            print("Not bimodal (peaks too close)")
            dist_to_target_1 = np.linalg.norm(np.array(real_centers[0]) - np.array([0,0]))
            dist_to_target_2 = np.linalg.norm(np.array(real_centers[1]) - np.array([0,0]))
            if dist_to_target_2<dist_to_target_1:
                return real_centers[1], False
            return real_centers[0], False
        else:
            print("Data is bimodal")
            dist_to_target_1 = np.linalg.norm(np.array(real_centers[0]) - np.array([0,0]))
            dist_to_target_2 = np.linalg.norm(np.array(real_centers[1]) - np.array([0,0]))
            if dist_to_target_2<dist_to_target_1:
                real_centers[0], real_centers[1] = real_centers[1], real_centers[0]#The closer target should be first

            return real_centers, True



    #     location_targets = [] #Where the two squares are at
    #     np_arr_occurences = np.array(num_occurences_xy, dtype=np.int16)
    #     # print("np_arr_occurences: ", np_arr_occurences)
    #     # max_1 = max(max(num_occurences_xy))
    #     # max_1 
    #     # print("max_1: ", max_1)
    #     # coord = np.where(num_occurences_xy == max_1)

    #     #Get first maximum
    #     coord = np.unravel_index(np_arr_occurences.argmax(), np_arr_occurences.shape)
    #     np_arr_occurences[coord] = 0
    #     location_targets.append((coord[0], coord[1]))

    #     #Get second maximum only if it is not too close to the first maximum (at least 5m away)
    #     min_relative_dist = 5
    #     num_set_to_0 = 0
    #     total_length = sum(len(sub_arr) for sub_arr in np_arr_occurences)
    #     print("total_length: ", total_length)
    #     while True:
    #         coord = np.unravel_index(np_arr_occurences.argmax(), np_arr_occurences.shape)
    #         print("coord: ", coord)
    #         x_m_current_idx = [coord[1]-1 if coord[1]-1 >= 0 else 0][0]
    #         y_m_current_idx = [coord[0]-1 if coord[0]-1 >= 0 else 0][0]
    #         x_m_max_idx = [location_targets[0][1]-1 if location_targets[0][1]-1 >= 0 else 0][0]
    #         y_m_max_idx = [location_targets[0][0]-1 if location_targets[0][0]-1 >= 0 else 0][0]
    #         xy_m_current = (bins_x[x_m_current_idx], bins_y[y_m_current_idx])
    #         xy_m_max = (bins_x[x_m_max_idx], bins_y[y_m_max_idx])
    #         distance = np.linalg.norm(np.array(xy_m_current) - np.array(xy_m_max))
    #         print("distance: ", distance)
    #         if distance > min_relative_dist:
    #             location_targets.append((coord[0], coord[1]))
    #             break
    #         else:
    #             np_arr_occurences[coord] = 0
    #             num_set_to_0 += 1
    #         print("count: ", sum(len(sub_arr) for sub_arr in np_arr_occurences))

    #         if num_set_to_0 > total_length*0.05:
    #             #The second peak should be within the top 5% of the data
    #             break
        
    #     #Change to x/y format
    #     for i in range(sum(len(sub_arr) for sub_arr in location_targets)//2):
    #         location_targets[i] = (bins_x[location_targets[i][1]-1], bins_y[location_targets[i][0]-1])

    #     print("location_targets: ", location_targets)
    #     # plt.imshow(num_occurences_xy)
    #     # plt.colorbar()
    #     # plt.show()
    #     if(sum(len(sub_arr) for sub_arr in location_targets)//2 != 2):
    #         two_squares = False
    #         location_targets = location_targets[0] #Remove useless dimension
    #         print("Looks bimodal but is not")
    #     else:
    #         print("Data is bimodal")
    #         two_squares = True
    #         dist_to_target_1 = np.linalg.norm(np.array(location_targets[0]) - np.array([0,0]))
    #         dist_to_target_2 = np.linalg.norm(np.array(location_targets[1]) - np.array([0,0]))

    #         plt.hist(x_values, bins=num_bins, label='x', color='r')
    #         plt.hist(y_values, bins=num_bins, label='y', color='b')
    #         plt.show()
    #         cv.waitKey(0)
    #         if dist_to_target_2<dist_to_target_1:
    #             location_targets[0], location_targets[1] = location_targets[1], location_targets[0]#The closer target should be first
    #     return location_targets, two_squares
    else:
        print("Data is unimodal")
        #Data is unimodal
        x_mean = np.mean(x_values)
        y_mean = np.mean(y_values)
        return (x_mean, y_mean), False
        



    cv.waitKey(0)

    distances = np.linalg.norm(calc_error_xy, axis=1)
    errors_xy= np.mean(calc_error_xy, axis=0)
    standard_deviation = np.std(distances)
    #print("Standard deviation: ", standard_deviation)
    #implement logic to check if deviation is too high for one platform
    print("final error: ", errors_xy)
    return errors_xy

def check_for_time(frame, altitude,duration,ratio_detected, size_square, cam_hfov):
    """check if a square is detected in the frame for 3 seconds, if it is, return coordinates, if not, return None"""
    #Use function variables to store values between function calls (only executed once)
    if check_for_time.start_time is None:
        check_for_time.start_time = perf_counter()
        check_for_time.errors_xy = []
        check_for_time.not_detected_cnt = 0
    
    err_square, _, threshold_img = detect_square_main(frame, altitude, size_square, cam_hfov)
    if err_square != None: #If a square is detected, add the error to the list
        check_for_time.errors_xy.append(err_square)
    else:
        check_for_time.not_detected_cnt += 1

    if perf_counter() - check_for_time.start_time > duration: #If 3 seconds have passed, check if a square was detected in more than half of the frames
        if len(check_for_time.errors_xy )/(check_for_time.not_detected_cnt+1e-5) > ratio_detected: #If more than half of the frames have a square, get target error
            calculated_target_err, is_bimodal =  calculate_target_error(check_for_time.errors_xy, frame)
            return calculated_target_err, is_bimodal, threshold_img
        else:
            return False, False, threshold_img
    else:
        return None, False, threshold_img

    
    



























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







