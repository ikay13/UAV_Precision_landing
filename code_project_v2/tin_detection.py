# Python code for Multiple Color Detection 


import numpy as np 
import cv2 as cv
import matplotlib.pyplot as plt
from coordinate_transform import calculate_size_in_px


cam = cv.VideoCapture("images/tins_1.mp4")

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
    for idx in range(len(avg_h_val)):
        if avg_h_val[idx] < 30: #If the hue is less than 30, it is red
            avg_h_val[idx] = 180 + avg_h_val[idx] #Change the hue so that red is always the highest value
    avg_h_val = np.array(avg_h_val)
    colour_order = ("G", "B", "R")
    tins_gbr_idx = [None for _ in range(3)]

    number_tins = len(avg_h_val)
    match number_tins:
        case 3:
            sort_idx = np.argsort(avg_h_val) #Sort the hue values (Green, Blue, Red)
            tins_gbr_idx = sort_idx #The index of the tins in the order green, blue, red
        case 2:
            threshold_blue_to_red = (tin_colours_obj.blue_hue + tin_colours_obj.red_hue) // 2 #The threshold for the hue value to be blue or red
            if max(avg_h_val) > threshold_blue_to_red: #Red tin exists
                #Both green and blue will be assigned the same index as it is too difficult to distinguish between them
                tins_gbr_idx[2] = np.argmax(avg_h_val) #The red tin is the one with the highest hue value
                tins_gbr_idx[0:2] = [np.argmin(avg_h_val) for _ in range(2)] #The green and blue tin are the one with the lowest hue value
            else: #No red tin exists
                sort_idx = np.argsort(avg_h_val) #Sort the hue values (Green, Blue)
                tins_gbr_idx[0:2] = sort_idx #The index of the tins in the order green, blue
        case 1:
            threshold_blue_to_red = (tin_colours_obj.blue_hue + tin_colours_obj.red_hue) // 2 #The threshold for the hue value to be blue or red
            if avg_h_val > threshold_blue_to_red: #Tin is red
                tins_gbr_idx[2] = 0 #Only the red tin has a valid index
            else: #Tin is green or blue
                tins_gbr_idx[0:2] = [0 for _ in range(2)] #Green and blue tin have a valid index (not distinguishable)
    return tins_gbr_idx
            
        
def tin_detection(frame, altitude, cam_hfov, circle_parameters_obj, tin_colours_obj):
    frame_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    blur_gray = cv.GaussianBlur(frame_gray, (11, 11), 0)
    blur_hsv = cv.GaussianBlur(frame_hsv, (11, 11), 0)
    diameter_tin_px = calculate_size_in_px(altitude=altitude, size_object_m=circle_parameters_obj.tin_diameter, cam_hfov=cam_hfov, image_width=frame.shape[1])

    ### Get circles using hough circles
    cannyEdgeMaxThr = circle_parameters_obj.canny_max_threshold #Max Thr for canny edge detection
    circleDetectThr = circle_parameters_obj.hough_circle_detect_thr #Threshold for circle detection
    tolerance = 0.25    #This is the tolarance the circles (of the tins) are expected to be in
    diameter_tin_max = int(diameter_tin_px*(1+tolerance)) #Max diameter of the tin
    diameter_tin_min = int(diameter_tin_px*(1-tolerance)) #Min diameter of the tin
    circles = cv.HoughCircles(blur_gray, cv.HOUGH_GRADIENT, 1, 50,
                                param1=cannyEdgeMaxThr,param2=circleDetectThr,minRadius=diameter_tin_min//2,maxRadius=diameter_tin_max//2)

    if circles is None:
        print("No circles found")
        return None, None
    
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

    print("Average color of the tins: ", avg_h_val)
    print("Sorted index: ", sort_tins_idx)
    for current_color_idx in range(3): #For each color (green=0, blue=1, red=2) draw the circle
        gbr_idx = sort_tins_idx[current_color_idx]
        if gbr_idx is None:
            continue    
        diameter_to_draw = (diameter_tin_max+diameter_tin_min)//4 #convert avg to radius
        current_center = centers[sort_tins_idx[current_color_idx]] 
        current_color = colors_to_diplay[current_color_idx]
        cv.circle(frame, current_center, diameter_to_draw, current_color, 2)
        cv.circle(frame, current_center, 2, current_color, 3)

    masks_combined = np.zeros((frame.shape[0], frame.shape[1]), np.uint8)
    for mask in masks:
        masks_combined = cv.bitwise_or(masks_combined, mask)
    #masks_combined = cv.cvtColor(masks_combined, cv.COLOR_GRAY2BGR)
    bgr_frame = cv.bitwise_and(frame, frame, mask=masks_combined)
    cv.imshow("Masks", bgr_frame)
         


    # print("Average color of the tins: ", avg_h_val)
    # print("Sorted index: ", sort_idx)

    # for circle in circles:
    #     print("diameter: ", circle[2]*2)
    #     cv.circle(frame,(circle[0],circle[1]),circle[2],(0,255,0),2)
    #     # draw the center of the circle
    #     cv.circle(frame,(circle[0],circle[1]),2,(0,0,255),3)
    cv.imshow("Circles", frame)
    cv.waitKey(0)

target_parameters_obj = target_parameters()
tin_colours_obj = tin_colours()
while(True):
	ret,frame = cam.read()
	if ret:
		# Reading the video from the 
		# webcam in image frames 
		imageFrame = cv.resize(frame, (360, 540))
		tin_detection(imageFrame, 0.15, 65, target_parameters_obj, tin_colours_obj)



# def small_circle(frame, altitude, cam_hfov, circle_parameters_obj):
#     """Detects concentric circles in the image using altitude"""
#     ###Parameters
#     cannyEdgeMaxThr = circle_parameters_obj.canny_max_threshold #Max Thr for canny edge detection
#     circleDetectThr = circle_parameters_obj.hough_circle_detect_thr #Threshold for circle detection
#     factor = circle_parameters_obj.factor #Factor big circle diameter / small circle diameter
#     tolerance = 0.25     #This is the tolarance the circles are expected to be in

#     calculated_altitude = None #This is the altitude calculated from the image (As the circel dimensions are known)

#     ###Calculate the size of the small circle relative to altitude and camera hfov
#     dist_img_on_ground = tan(cam_hfov/2)*2*altitude
#     rel_size = 0.72/dist_img_on_ground #This is the size of the bigger circle compared to the overall frame
#     radius_big_pixel = rel_size*frame.shape[1]/2 #This is the radius of the bigger circle in pixels
#     radius_small_pixel = radius_big_pixel/factor #This is the radius of the smaller circle in pixels
#     radii_small = [int(radius_small_pixel*(1-tolerance)), int(radius_small_pixel*(1+tolerance))] #Min and max radius for the smaller circle

#     frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
#     blur = cv.medianBlur(frame,11)
    
#     ###Find big circles
#     circles = cv.HoughCircles(blur,cv.HOUGH_GRADIENT,1,50,
#                                 param1=cannyEdgeMaxThr,param2=circleDetectThr,minRadius=radii_small[0],maxRadius=radii_small[1])
    
#     if circles is None:
#         return None, None
    
#     circles = np.int16(np.around(circles))
#     circles = circles[0] #remove redundant dimensions
    
#     if len(circles) == 1:
#         circle = circles[0]
#         radius_imaginary_big = circle[2]*factor
#         alt = calculate_altitude(radius_big_circle=radius_imaginary_big)
#         calculated_altitude = alt
#         #print("Altitude: ", alt)

#         error_xy = calculate_error_image(circles=circles, img_width=frame.shape[1], img_height=frame.shape[0],num_of_circles=1)
        
#         cv.circle(frame,(circle[0],circle[1]),circle[2],(0,255,0),2)
#         # draw the center of the circle
#         cv.circle(frame,(circle[0],circle[1]),2,(0,0,255),3)

#     ##only for visual representation
#     edges = cv.Canny(blur,0.5*cannyEdgeMaxThr,cannyEdgeMaxThr)
#     combinedImage = np.concatenate((frame, edges), axis=1) #Combine canny edge detection and gray image
#     cv.imshow('Circles and Canny', combinedImage) #Display the combined image
#     cv.waitKey(1)

#     return calculated_altitude, error_xy










# def find_red_tin(hsvFrame, minsize):
# 	# Set range for red color and 
# 	# define mask 
# 	red_lower = np.array([330//2, 30, 50], np.uint8) 
# 	red_upper = np.array([186, 255, 255], np.uint8) 
# 	red_mask = cv.inRange(hsvFrame, red_lower, red_upper) 
	
	
# 	# Morphological Transform, Dilation 
# 	# for each color and bitwise_and operator 
# 	# between imageFrame and mask determines 
# 	# to detect only that particular color
# 	kernel = np.ones((5, 5), "uint8") 
	
# 	#####Search for the red tin#####
# 	# For red color 
# 	red_mask = cv.dilate(red_mask, kernel) 
# 	# Creating contour to track red color 
# 	cnts, hierarchy = cv.findContours(red_mask, 
# 										cv.RETR_TREE, 
# 										cv.CHAIN_APPROX_SIMPLE) 
# 	min_circle_red = [] #This will be the minimum enclosing circle [x, y, radius] of the red tin
# 	red_tin_exists = False
# 	for cnt in cnts: 
# 		area = cv.contourArea(cnt) 
# 		if(area > 1000): #Checks if the contour is big enough to be a tin
# 				temp_circle = cv.minEnclosingCircle(cnt)
# 				center = (int(temp_circle[0][0]), int(temp_circle[0][1]))
# 				radius = int(temp_circle[1])
# 				min_circle_red = [center, radius]
# 				# cv.circle(imageFrame, center, 5, (0, 0, 255), -1)
# 				# cv.circle(imageFrame, center, radius, (0, 0, 255), 2)
# 				red_tin_exists = True

# 				cv.drawContours(imageFrame, [cnt], -1, (0, 0, 255), 2)
# 	#Return the minimum enclosing circle of the red tin and a bool indicating if the red tin exists
# 	return min_circle_red, red_tin_exists 













# red_found_cnt = 0
# red_notfound_cnt = 0
# while(True):
	
# 	ret,frame = cam.read()
# 	if ret:
# 		# Reading the video from the 
# 		# webcam in image frames 
# 		imageFrame = cv.resize(frame, (360, 540))
# 		imageFrame = cv.GaussianBlur(imageFrame, (15, 15), 0)

# 		# Convert the imageFrame in 
# 		# BGR to HSV color space
# 		hsvFrame = cv.cvtColor(imageFrame, cv.COLOR_BGR2HSV) 

# 		min_circle_red, red_tin_exists = find_red_tin(hsvFrame, 1000)
# 		if red_tin_exists: 
# 			red_found_cnt += 1
# 			###################################################
# 			########Landing pad for picking up the tin#########
# 			###################################################

# 			img_gray = cv.cvtColor(imageFrame, cv.COLOR_BGR2GRAY)

# 			#####find tins using HoughCircles#####
# 			#min and max radius of the circle based on the radius of the red tin
# 			rows = img_gray.shape[0]
# 			min_radius = min_circle_red[1] - rows//50
# 			max_radius = min_circle_red[1] + rows//50
# 			circles = cv.HoughCircles(img_gray, cv.HOUGH_GRADIENT, 1, rows / 8,
# 								param1=35, param2=30,
# 								minRadius=min_radius, maxRadius=max_radius)
			
# 			####Create masks for the blue and green tins######
# 			masks = [np.zeros((img_gray.shape[0], img_gray.shape[1]), np.uint8) for _ in range(2)] #List of masks for each circle
# 			centers = [[],[]] #List of the centers of the circles (green and blue tins) in same order as masks
# 			if circles is not None and len(circles[0]) == 3:	#If 3 circles are found, draw the masks
# 				mask_idx = 0
# 				circles = np.uint16(np.around(circles)) #Convert the coordinates and radius to integers
# 				for idx in range(len(circles[0])): #For each circle
# 					center = (circles[0][idx][0], circles[0][idx][1]) 
# 					# distance from the center of the red tin to the center of the circle
# 					dist_to_red = np.sqrt((center[0] - min_circle_red[0][0])**2 + (center[1] - min_circle_red[0][1])**2)
# 					if dist_to_red > min_circle_red[1]: #This is not the red circle
# 						radius = circles[0][idx][2]
# 						centers[mask_idx] = center
# 						cv.circle(masks[mask_idx], center, radius-15, (255, 255, 255), -1) ######Modify -15 to a better value
# 						mask_idx += 1


# 				#####Apply the masks to the image#####
# 				average_colors = []
# 				for current_mask in masks: #For each hsv mask (blue and green)
# 					average_colors.append(cv.mean(hsvFrame, mask=current_mask)) #Find the average color of the masked image

# 				#for idx in range(len(average_colors)):
# 					#print("Average color of tin", idx, ":", average_colors[idx])

# 				#If the average hue of the first tin is greater than the second tin, the first tin is blue
# 				if average_colors[0][0] > average_colors[1][0]: 
# 					cv.circle(imageFrame, centers[0], min_circle_red[1], (255, 0, 0), 2)
# 					cv.circle(imageFrame, centers[1], min_circle_red[1], (0, 255, 0), 2)
# 				else:
# 					cv.circle(imageFrame, centers[0], min_circle_red[1], (0, 255, 0), 2)
# 					cv.circle(imageFrame, centers[1], min_circle_red[1], (255, 0, 0), 2)

# 				mask = masks[0] + masks[1] #combine the masks for display
# 				circles_coords = [[]*3] #List of the coordinates of the circles in x,y format
				
				

# 				# Display the image
# 				mask = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
# 				concat_img = cv.hconcat([mask, imageFrame])
# 				cv.imshow("Multiple Color Detection in Real-TIme", concat_img) 
# 		else:
# 			red_notfound_cnt += 1

		
# 		#####Program Termination#####
# 		if cv.waitKey(10) & 0xFF == ord('q'): 
# 			cam.release() 
# 			cv.destroyAllWindows() 
# 			break

# 	else:
# 		break
# print("Red found: ", red_found_cnt, "Red not found: ", red_notfound_cnt)
# cam.release()


