import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from time import time
from math import atan2, tan
from coordinate_transform import calculate_altitude

# ###Parameters
# cannyEdgeMaxThr = 40 #Max Thr for canny edge detection
# circleDetectThr = 35 #Threshold for circle detection
# size = 30           #Size of the circles (to be calculated)
# factor = 3.2          #Factor big circle diameter / small circle diameter
# rangePerc = 1.5     #This is the range the circles are expected to be in

#cap = cv.VideoCapture(0)
#plt.ion()

def calculate_error_image(circles, img_width, img_height, num_of_circles):
    """Calculate the error in the x and y direction from the center of the image"""
    if num_of_circles == 2:
        cnt_x = (circles[0][0][0] + circles[0][1][0]) / 2
        cnt_y = (circles[0][0][1] + circles[0][1][1]) / 2
        center_xy = (cnt_x, cnt_y)
    else: #only one circle
        center_xy = (circles[0][0], circles[0][1])
    error_xy = ((center_xy[0] / img_width-0.5)*2, (center_xy[1] / img_height-0.5)*-1.5)  # calculate relative error in x and y direction
    return error_xy


def concentric_circles(frame, altitude, cam_hfov, circle_parameters_obj):
    """Detects concentric circles in the image using altitude"""
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    blur = cv.medianBlur(frame_gray,3)

    #Threshold image to get only the area inside square plus other bright spots (reduces edges in image)
    blur_otsu = cv.GaussianBlur(blur, (11, 11), 0)
    _, thr = cv.threshold(blur_otsu, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)

    blur = cv.bitwise_and(blur, thr)


    ###Parameters
    cannyEdgeMaxThr = circle_parameters_obj.canny_max_threshold #Max Thr for canny edge detection
    circleDetectThr = circle_parameters_obj.hough_circle_detect_thr #Threshold for circle detection
    factor = circle_parameters_obj.factor #Factor big circle diameter / small circle diameter
    tolerance = 1.5     #This is the tolarance the circles are expected to be in

    calculated_altitude = None #This is the altitude calculated from the image (As the circel dimensions are known)

    ###Calculate the size of the circles relative to altitude and camera hfov
    dist_img_on_ground = tan(cam_hfov/2)*2*altitude
    actual_radius = circle_parameters_obj.diameter_big/2
    rel_size = actual_radius/dist_img_on_ground #This is the size of the bigger circle compared to the overall frame
    radius_big_pixel = rel_size*frame_gray.shape[1] #This is the radius of the bigger circle in pixels
    radius_small_pixel = radius_big_pixel/factor #This is the radius of the smaller circle in pixels
    radii_big = [int(radius_big_pixel/tolerance), int(radius_big_pixel*tolerance)] #Min and max radius for the bigger circle
    radii_small = [int(radius_small_pixel/tolerance), int(radius_small_pixel*tolerance)] #Min and max radius for the smaller circle

    edges = cv.Canny(blur,0.5*cannyEdgeMaxThr,cannyEdgeMaxThr) #Only for visual representation (hough already does this)
    
    ###Find big circles
    circles_big = cv.HoughCircles(blur,cv.HOUGH_GRADIENT,1,50,
                                param1=cannyEdgeMaxThr,param2=circleDetectThr,minRadius=radii_big[0],maxRadius=radii_big[1])
    
    if circles_big is not None:
        circles_big = np.int16(np.around(circles_big))
        circles_big = circles_big[0] #remove redundant dimension


    ###Find small circles
    circles_small = cv.HoughCircles(blur,cv.HOUGH_GRADIENT,1,50,
                                param1=cannyEdgeMaxThr,param2=circleDetectThr//2,minRadius=radii_small[0],maxRadius=radii_small[1])
    if circles_small is not None:
        circles_small = np.int16(np.around(circles_small))
        circles_small = circles_small[0] #remove redundant dimension


    ###Check if both circles are found
    if circles_big is not None and circles_small is not None: #Circles have been found in both sizes
        circles = []
        for big_c in circles_big:
            for small_c in circles_small:
                #Check if the circles are concentric by calculating the distance between the centers and the ratio of the radii
                distanceX = abs(big_c[0] - small_c[0])
                distanceY = abs(big_c[1] - small_c[1])
                #ratio_big_small = big_c[2] / small_c[2]
                
                if np.sqrt(distanceX**2 + distanceY**2) < 10: # and ratio_big_small > factor-0.4 and ratio_big_small < factor+0.4
                    #print("Ratio: ", ratio_big_small)
                    #print("big_c: ", big_c, " small_c: ", small_c)
                    circles.append(np.concatenate(([big_c], [small_c]), axis=0))
                    #print("match")
                    break
        if len(circles) > 0:
            error_xy = calculate_error_image(circles=circles, img_width=frame_gray.shape[1], img_height=frame_gray.shape[0], num_of_circles=2)
            #print("Center: ", error_xy)
            for i in circles[0]:
                if radii_big[0] < i[2] < radii_big[1]:
                    actual_radius = circle_parameters_obj.diameter_big/2
                    alt = calculate_altitude(length_px=i[2], cam_hfov=cam_hfov, img_width=frame_gray.shape[1], actual_length=actual_radius)
                    calculated_altitude = alt
                #print("Altitude: ", alt)
                #This is drawn on orignal frame image passed to function and not a copy
                # draw the outer circle
                cv.circle(frame,(i[0],i[1]),i[2],(0,0,0),2)
                # draw the center of the circle
                cv.circle(frame,(i[0],i[1]),2,(0,0,0),3)
        else:
            #No concentric circles found
            return None, None, edges
    else:
        #No circles found (either big or small)
        
        # combinedImage = np.concatenate((frame, edges), axis=1) #Combine canny edge detection and gray image
        # cv.imshow('Circles and Canny', combinedImage) #Display the combined image
        # cv.waitKey(1)
        return None, None, edges
        

    # combinedImage = np.concatenate((frame, edges), axis=1) #Combine canny edge detection and gray image
    # cv.imshow('Circles and Canny', combinedImage) #Display the combined image
    # cv.waitKey(1)

    return calculated_altitude, error_xy, edges

def small_circle(frame, altitude, cam_hfov, circle_parameters_obj):
    """Detects concentric circles in the image using altitude"""
    ###Parameters
    cannyEdgeMaxThr = circle_parameters_obj.canny_max_threshold *1.2#Max Thr for canny edge detection
    circleDetectThr = circle_parameters_obj.hough_circle_detect_thr*1.2#Threshold for circle detection
    factor = circle_parameters_obj.factor #Factor big circle diameter / small circle diameter
    tolerance = 1.3     #This is the tolarance the circles are expected to be in

    calculated_altitude = None #This is the altitude calculated from the image (As the circel dimensions are known)

    ###Calculate the size of the small circle relative to altitude and camera hfov
    dist_img_on_ground = tan(cam_hfov/2)*2*altitude
    actual_radius = circle_parameters_obj.diameter_small/2
    rel_size = actual_radius/dist_img_on_ground #This is the size of the circle compared to the overall frame
    radius_small_pixel = rel_size*frame.shape[1] #This is the radius of the smaller circle in pixels
    radii_small = [int(radius_small_pixel/tolerance), int(radius_small_pixel*tolerance)] #Min and max radius for the smaller circle

    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    blur = cv.medianBlur(frame_gray,3)
    edges = cv.Canny(blur,0.5*cannyEdgeMaxThr,cannyEdgeMaxThr) #Only for visual representation (hough already does this)
    
    ###Find big circles
    circles = cv.HoughCircles(blur,cv.HOUGH_GRADIENT,1,50,
                                param1=cannyEdgeMaxThr,param2=circleDetectThr,minRadius=radii_small[0],maxRadius=radii_small[1])
    
    if circles is None:
        # combinedImage = np.concatenate((frame, edges), axis=1) #Combine canny edge detection and gray image
        # cv.imshow('Circles and Canny', combinedImage) #Display the combined image
        # cv.waitKey(1)
        return None, None, edges
    
    circles = np.int16(np.around(circles))
    circles = circles[0] #remove redundant dimensions
    
    if len(circles) == 1:
        circle = circles[0]
        actual_radius = circle_parameters_obj.diameter_small/2
        alt = calculate_altitude(length_px=circle[2], cam_hfov=cam_hfov, img_width=frame_gray.shape[1], actual_length=actual_radius)
        calculated_altitude = alt
        #print("Altitude: ", alt)

        error_xy = calculate_error_image(circles=circles, img_width=frame_gray.shape[1], img_height=frame_gray.shape[0],num_of_circles=1)
        
        #This is drawn on orignal frame image passed to function and not a copy
        cv.circle(frame,(circle[0],circle[1]),circle[2],(0,0,0),2)
        # draw the center of the circle
        cv.circle(frame,(circle[0],circle[1]),2,(0,0,0),3)
    else:
        return None, None, edges

    ##only for visual representation
    # combinedImage = np.concatenate((frame, edges), axis=1) #Combine canny edge detection and gray image
    # cv.imshow('Circles and Canny', combinedImage) #Display the combined image
    # cv.waitKey(1)

    return calculated_altitude, error_xy, edges



