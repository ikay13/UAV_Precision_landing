# Python code for Multiple Color Detection 


import numpy as np 
import cv2 as cv
import matplotlib.pyplot as plt
from coordinate_transform import calculate_size_in_px, transform_to_ground_xy
from time import perf_counter




class target_parameters:
    def __init__(self):
        self.diameter_big = 0.72
        self.diameter_small = 0.24
        self.canny_max_threshold = 50 #param1 of hough circle transform
        self.hough_circle_detect_thr = 30 #param2 of hough circle transform
        self.factor = self.diameter_big/self.diameter_small #How much bigger the big circle is compared to the small circle (diameter)
        self.tin_diameter = 0.084 #Diameter of the tins in meters
        self.size_square = 2 #Size of the square in meters

class tin_colours:
    def __init__(self):
        self.green_hue = 95
        self.blue_hue = 105
        self.red_hue = 170

def sort_tins(avg_h_val, tin_colours_obj):
    """This function is used to sort the tins in the order green, blue, red. If ony one of the blue/green tins is in the picture, both tins are 
    assigned to that index."""
    for idx in range(len(avg_h_val)):
        if avg_h_val[idx] < 30: #If the hue is less than 30, it is red
            avg_h_val[idx] = 180 + avg_h_val[idx] #Change the hue so that red is always the highest value
    avg_h_val = np.array(avg_h_val)
    colour_order = ("G", "B", "R")
    tins_gbr_idx = [None for _ in range(3)]

    number_tins = len(avg_h_val)
    if number_tins == 3:
        sort_idx = np.argsort(avg_h_val) #Sort the hue values (Green, Blue, Red)
        tins_gbr_idx = sort_idx #The index of the tins in the order green, blue, red
    elif number_tins == 2:
        threshold_blue_to_red = (tin_colours_obj.blue_hue + tin_colours_obj.red_hue) // 2 #The threshold for the hue value to be blue or red
        if max(avg_h_val) > threshold_blue_to_red: #Red tin exists
            #Both green and blue will be assigned the same index as it is too difficult to distinguish between them
            tins_gbr_idx[2] = np.argmax(avg_h_val) #The red tin is the one with the highest hue value
            tins_gbr_idx[0:2] = [np.argmin(avg_h_val) for _ in range(2)] #The green and blue tin are the one with the lowest hue value
        else: #No red tin exists
            sort_idx = np.argsort(avg_h_val) #Sort the hue values (Green, Blue)
            tins_gbr_idx[0:2] = sort_idx #The index of the tins in the order green, blue
    elif number_tins == 1:
        threshold_blue_to_red = (tin_colours_obj.blue_hue + tin_colours_obj.red_hue) // 2 #The threshold for the hue value to be blue or red
        if avg_h_val > threshold_blue_to_red: #Tin is red
            tins_gbr_idx[2] = 0 #Only the red tin has a valid index
        else: #Tin is green or blue
            tins_gbr_idx[0:2] = [0 for _ in range(2)] #Green and blue tin have a valid index (not distinguishable)
    return tins_gbr_idx
            
        
def tin_detection(frame, altitude, cam_hfov, circle_parameters_obj, tin_colours_obj):
    """This function is used to detect the tins in the frame. It returns the centers of the tins in the order green, blue, red. 
    If a tin is not detected, the center is None. If no tins are detected, None is returned. 
    If the tins are not detected, the frame is displayed and the function waits for a key press. 
    If the tins are detected, the frame is displayed for 1 ms."""
    frame_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    blur_gray = cv.GaussianBlur(frame_gray, (11, 11), 0)
    blur_hsv = cv.GaussianBlur(frame_hsv, (11, 11), 0)
    diameter_tin_px = calculate_size_in_px(altitude=altitude, size_object_m=circle_parameters_obj.tin_diameter, cam_hfov=cam_hfov, image_width=frame.shape[1])
    #print("Diameter of the tins in pixels: ", diameter_tin_px)
    ### Get circles using hough circles
    cannyEdgeMaxThr = circle_parameters_obj.canny_max_threshold//2 #Max Thr for canny edge detection
    circleDetectThr = circle_parameters_obj.hough_circle_detect_thr//2 #Threshold for circle detection
    tolerance = 0.25    #This is the tolarance the circles (of the tins) are expected to be in
    diameter_tin_max = int(diameter_tin_px*(1+tolerance)) #Max diameter of the tin
    diameter_tin_min = int(diameter_tin_px*(1-tolerance)) #Min diameter of the tin
    print("Diameter of the tins in pixels: ", diameter_tin_max, diameter_tin_min)
    circles = cv.HoughCircles(blur_gray, cv.HOUGH_GRADIENT, 1, 50,
                                param1=cannyEdgeMaxThr,param2=circleDetectThr,minRadius=diameter_tin_min//2,maxRadius=diameter_tin_max//2)
    canny_edges = cv.Canny(blur_gray, 0.5*cannyEdgeMaxThr, cannyEdgeMaxThr)

    if circles is None:
        canny_edges = cv.cvtColor(canny_edges, cv.COLOR_GRAY2BGR)
        frame = cv.hconcat([frame, canny_edges])
        #cv.imshow("Circles", frame)
        #cv.waitKey(0)
        print("No circles found")
        return None
    
    circles = np.int16(np.around(circles))
    circles = circles[0] #remove redundant dimensions

    
    #Check for how many circles are found (array is 1D if only one circle is found, 2D if multiple are found)
    array_dimensions = len(circles.shape)
    if array_dimensions == 1: #If only one circle is found
        num_circles = 1
    else:
        num_circles = circles.shape[0]
    
    ###Create masks for all tins
    masks = [np.zeros((frame.shape[0], frame.shape[1]), np.uint8) for _ in range(num_circles)] #List of masks for each circle
    centers = [] #List of the centers of the circles in same order as the masks
    if num_circles <= 3:	#If 3 circles or less are found, draw the masks
        radius_mask = int(diameter_tin_px/4)
        for idx in range(num_circles):
            center = (circles[idx][0], circles[idx][1])
            centers.append(center)
            cv.circle(masks[idx], center, radius_mask, (255, 255, 255), -1) #Draw the masks

    average_colors = []
    for current_mask in masks: #For each hsv mask (blue and green)
        average_colors.append(cv.mean(frame_hsv, mask=current_mask)) #Find the average color of the masked image

    avg_h_val = [avg_color[0] for avg_color in average_colors] #only the hue values
    sort_tins_idx = sort_tins(avg_h_val, tin_colours_obj)

    colors_to_diplay = ((0, 255, 0), (255, 0, 0), (0, 0, 255))
    gbr_centers = [[None, None] for _ in range(3)]
    for current_color_idx in range(3): #For each color (green=0, blue=1, red=2) draw the circle
        gbr_idx = sort_tins_idx[current_color_idx]
        if gbr_idx is None:
            continue    
        diameter_to_draw = (diameter_tin_max+diameter_tin_min)//4 #convert avg to radius
        current_center = centers[sort_tins_idx[current_color_idx]] 
        gbr_centers[current_color_idx] = current_center
        current_color = colors_to_diplay[current_color_idx]

        cv.circle(frame, current_center, diameter_to_draw, current_color, 2)
        cv.circle(frame, current_center, 2, current_color, 3)



    # masks_combined = np.zeros((frame.shape[0], frame.shape[1]), np.uint8)
    # for mask in masks:
    #     masks_combined = cv.bitwise_or(masks_combined, mask)
    #masks_combined = cv.cvtColor(masks_combined, cv.COLOR_GRAY2BGR)
    # bgr_frame = cv.bitwise_and(frame, frame, mask=masks_combined)
    # cv.imshow("Masks", bgr_frame)
         
    # canny_edges = cv.cvtColor(canny_edges, cv.COLOR_GRAY2BGR)
    # frame = cv.hconcat([frame, canny_edges])
    #cv.imshow("Circles", frame)
    #cv.waitKey(1)
    return gbr_centers

def calculate_error_in_image (coordinates, img_width, img_height): #return error relative to the center of the image
    """Calculate the error in the image plane (-1 to 1 in x and -0.75 to 0.75 in y)"""
    #coordinates is structured [[(g_x,g_y),(b_x,b_y),(r_x,r_y)],[(g_x,g_y),(b_x,b_y),(r_x,r_y)],...]
    coordinates_img = [[None,None] for _ in range(3)]
    for color_idx in range(3):
        if coordinates[color_idx][0] is not None and coordinates[color_idx][1] is not None:
            error_x = (coordinates[color_idx][0] / img_width-0.5)*2
            error_y = (coordinates[color_idx][1] / img_height-0.5)*-1.5
            coordinates_img[color_idx] = (error_x, error_y)
    return coordinates_img

def tin_detection_for_time(frame, uav_inst, circle_parameters_obj, tin_colours_obj, altitude):
    """This function is used to run the tin detection for a certain amount of time and return the errors in gbr format. 
    Returns the errors in gbr format if the time is up, returns False if not enough tins are detected and returns None if not done yet."""
    cam_hfov = uav_inst.cam_hfov
    angle_uav_xy = (uav_inst.angle_x, uav_inst.angle_y)
    time_to_run = 0.5 #Time to run the function in seconds
    if tin_detection_for_time.start_time is None:
        tin_detection_for_time.start_time = perf_counter()
        tin_detection_for_time.errors_xy = []
        tin_detection_for_time.not_detected_cnt = 0
    gbr_centers = tin_detection(frame, altitude, cam_hfov, circle_parameters_obj, tin_colours_obj)
    if gbr_centers is None:
        tin_detection_for_time.not_detected_cnt += 1
    else: #append the current errors of the tin to the list
        error_img = calculate_error_in_image(coordinates=gbr_centers, img_width=frame.shape[1], img_height=frame.shape[0])
        error_ground = [[None, None] for _ in range(3)]
        for idx in range(3):
            if error_img[idx][0] is not None or error_img[idx][1] is not None:
                error_ground[idx] = transform_to_ground_xy(error_img_xy=error_img[idx], altitude=altitude, fov_hv=(uav_inst.cam_hfov, uav_inst.cam_vfov))
        tin_detection_for_time.errors_xy.append(error_ground)
    if perf_counter() - tin_detection_for_time.start_time > time_to_run and len(tin_detection_for_time.errors_xy) > tin_detection_for_time.not_detected_cnt:
        return tin_detection_for_time.errors_xy #Return the errors in gbr format
    elif tin_detection_for_time.not_detected_cnt > time_to_run:
        return False #Not enough tins detected
    else:
        return None #Not done yet
        # print("Tins not detected: ", tin_detection_for_time.not_detected_cnt)
        # tin_detection_for_time.start_time = None
        # print("Errors: ", tin_detection_for_time.errors_xy)
        # tin_detection_for_time.errors_xy = []
        # tin_detection_for_time.not_detected_cnt = 0





def tins_error_bin_mode(error_ground_gbr_xy, uav_inst, frame_width, frame_height):
    
    # #Convert the error in px to the error in the ground plane in m (accoounts for tilt of the UAV and altitude)
    # error_img_plane = calculate_error_in_image(coordinates=list_errors_gbr, img_width=frame_width, img_height=frame_height)
    # error_ground_gbr_xy = [[[None, None] for __ in range(3)] for _ in range(len(list_errors_gbr))]
    # for error in error_img_plane:
    #     for idx in range(3):
    #         error_ground_gbr_xy[error_img_plane.index(error)][idx] = transform_to_ground_xy(error_img_xy=error[idx], angle_uav_xy=(uav_inst.angle_x, uav_inst.angle_y), altitude=uav_inst.altitude)

    #Calculate the min and max of the errors
    min_xy = [0,0]
    max_xy = [0,0]
    for error in error_ground_gbr_xy:
        for idx in range(3):
            current_xy = [0,0]
            for x_y_idx in range(2):
                if error[idx][x_y_idx] is None:
                    current_xy[x_y_idx] = 0
                else:
                    current_xy[x_y_idx] = error[idx][x_y_idx]
                if error[idx][x_y_idx] < min_xy[x_y_idx]:
                    min_xy[x_y_idx] = error[idx][x_y_idx] 
                if error[idx][x_y_idx] > max_xy[x_y_idx]:
                    max_xy[x_y_idx] = error[idx][x_y_idx]

    #Divides the field in which bins where found into num_bins equally sized squares
    coords_gbr_final_xy = []
    num_bins = 50
    # offset_x = (max_xy[0]-min_xy[0])/(num_bins*2+0.5)
    # offset_y = (max_xy[1]-min_xy[1])/(num_bins*2+0.5)
    bins_x = np.linspace(min_xy[0], max_xy[0], num_bins)
    bins_y = np.linspace(min_xy[1], max_xy[1], num_bins)

    #Sort the errors into bins
    for idx in range(3):
        binned_errors_x = []
        binned_errors_y = []
        for error in error_ground_gbr_xy:
            if error[idx][0] is None or error[idx][1] is None:
                continue
            bin_x = np.digitize(error[idx][0], bins_x)
            bin_y = np.digitize(error[idx][1], bins_y)
            binned_errors_x.append(bin_x)
            binned_errors_y.append(bin_y)
        #Find the most common bin
        max_x_idx = max(set(binned_errors_x), key = binned_errors_x.count)
        max_y_idx = max(set(binned_errors_y), key = binned_errors_y.count)
        if max_x_idx < len(bins_x)-1:
            x_coord = (bins_x[max_x_idx]+bins_x[max_x_idx+1])/2
        else:
            x_coord = bins_x[max_x_idx]
        if max_y_idx < len(bins_y)-1:
            y_coord = (bins_y[max_y_idx]+bins_y[max_y_idx+1])/2
        else:
            y_coord = bins_y[max_y_idx]
        coords_gbr_final_xy.append((x_coord, y_coord))
    #print("Coords: ", coords_gbr_final_xy)
    return coords_gbr_final_xy
        
    


# target_parameters_obj = target_parameters()
# tin_colours_obj = tin_colours()
# cam = cv.VideoCapture("images/tins_sun.mp4")
# while(True):
# 	ret,frame = cam.read()
# 	if ret:
# 		# Reading the video from the 
# 		# webcam in image frames 
# 		imageFrame = cv.resize(frame, (360, 540))
# 		tin_detection(imageFrame, 0.1, 65, target_parameters_obj, tin_colours_obj)



