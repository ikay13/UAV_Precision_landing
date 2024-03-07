import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from time import time
from math import atan2, tan

# ###Parameters
# cannyEdgeMaxThr = 40 #Max Thr for canny edge detection
# circleDetectThr = 35 #Threshold for circle detection
# size = 30           #Size of the circles (to be calculated)
# factor = 3.2          #Factor big circle diameter / small circle diameter
# rangePerc = 1.5     #This is the range the circles are expected to be in

#cap = cv.VideoCapture(0)
#plt.ion()

def calculate_error_image(circles, img_width, img_height):
    """Calculate the error in the x and y direction from the center of the image"""
    center_xy = (np.mean(circles[0][0:1][0], axis=0),np.mean(circles[0][0:1][0], axis=0)) # calculate the average center of the circles
    error_xy = ((center_xy[0] / img_width-0.5)*2, (center_xy[1] / img_height-0.5)*-1.5)  # calculate relative error in x and y direction
    return error_xy

def calculate_altitude(radius_big_circle):
    """Calculate the altitude from the diameter of the big circle"""
    angle_per_px = 1.83e-3 #This is the angle per pixel in radians
    radius_meters = 0.72*0.5 #This is the radius of the big circle in meters
    angle_circle = radius_big_circle*angle_per_px
    altitude = radius_meters/(tan(angle_circle))
    return altitude


def concentric_circles(frame, altitude, cam_hfov, circle_parameters_obj):
    """Detects concentric circles in the image using altitude"""
    ###Parameters
    cannyEdgeMaxThr = circle_parameters_obj.canny_max_threshold #Max Thr for canny edge detection
    circleDetectThr = circle_parameters_obj.hough_circle_detect_thr #Threshold for circle detection
    factor = circle_parameters_obj.factor #Factor big circle diameter / small circle diameter
    tolerance = 0.25     #This is the tolarance the circles are expected to be in

    calculated_altitude = None #This is the altitude calculated from the image (As the circel dimensions are known)

    ###Calculate the size of the circles relative to altitude and camera hfov
    dist_img_on_ground = atan2(cam_hfov,2)*2*altitude
    rel_size = 0.72/dist_img_on_ground #This is the size of the bigger circle compared to the overall frame
    radius_big_pixel = rel_size*frame.shape[0]/2 #This is the radius of the bigger circle in pixels
    radius_small_pixel = radius_big_pixel/factor #This is the radius of the smaller circle in pixels
    radii_big = [int(radius_big_pixel*(1-tolerance)), int(radius_big_pixel*(1+tolerance))] #Min and max radius for the bigger circle
    radii_small = [int(radius_small_pixel*(1-tolerance)), int(radius_small_pixel*(1+tolerance))] #Min and max radius for the smaller circle

    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    blur = cv.medianBlur(frame,11)

    
    
    ###Find big circles
    circles_big = cv.HoughCircles(blur,cv.HOUGH_GRADIENT,1,50,
                                param1=cannyEdgeMaxThr,param2=circleDetectThr,minRadius=radii_big[0],maxRadius=radii_big[1])
    
    if circles_big is not None:
        circles_big = np.int16(np.around(circles_big))
        circles_big = circles_big[0] #remove redundant dimension

    
    
    ###Find small circles
    circles_small = cv.HoughCircles(blur,cv.HOUGH_GRADIENT,1,50,
                                param1=cannyEdgeMaxThr,param2=circleDetectThr,minRadius=radii_small[0],maxRadius=radii_small[1])
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
            error_xy = calculate_error_image(circles, frame.shape[0], frame.shape[1])
            print("Center: ", error_xy)
            for i in circles[0]:
                if radii_big[0] < i[2] < radii_big[1]:
                    alt = calculate_altitude(radius_big_circle=i[2])
                    calculated_altitude = alt
                print("Altitude: ", alt)
                # draw the outer circle
                cv.circle(frame,(i[0],i[1]),i[2],(0,255,0),2)
                # draw the center of the circle
                cv.circle(frame,(i[0],i[1]),2,(0,0,255),3)
    else:
        print("No circles found (either big or small)")
        # print("big: ", circles_big)
        # print("small: ", circles_small)
        # print("radii_big: ", radii_big)

    ##only for visual representation
    edges = cv.Canny(blur,0.5*cannyEdgeMaxThr,cannyEdgeMaxThr)
    combinedImage = np.concatenate((frame, edges), axis=1) #Combine canny edge detection and gray image
    cv.imshow('Circles and Canny', combinedImage) #Display the combined image
    cv.waitKey(1)

    return calculated_altitude

def small_circle(frame, altitude, cam_hfov, circle_parameters_obj):
    """Detects concentric circles in the image using altitude"""
    ###Parameters
    cannyEdgeMaxThr = circle_parameters_obj.canny_max_threshold #Max Thr for canny edge detection
    circleDetectThr = circle_parameters_obj.hough_circle_detect_thr #Threshold for circle detection
    factor = circle_parameters_obj.factor #Factor big circle diameter / small circle diameter
    tolerance = 0.25     #This is the tolarance the circles are expected to be in

    calculated_altitude = None #This is the altitude calculated from the image (As the circel dimensions are known)

    ###Calculate the size of the small circle relative to altitude and camera hfov
    dist_img_on_ground = atan2(cam_hfov,2)*2*altitude
    rel_size = 0.72/dist_img_on_ground #This is the size of the bigger circle compared to the overall frame
    radius_big_pixel = rel_size*frame.shape[0]/2 #This is the radius of the bigger circle in pixels
    radius_small_pixel = radius_big_pixel/factor #This is the radius of the smaller circle in pixels
    radii_small = [int(radius_small_pixel*(1-tolerance)), int(radius_small_pixel*(1+tolerance))] #Min and max radius for the smaller circle

    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    blur = cv.medianBlur(frame,11)
    
    ###Find big circles
    circles = cv.HoughCircles(blur,cv.HOUGH_GRADIENT,1,50,
                                param1=cannyEdgeMaxThr,param2=circleDetectThr,minRadius=radii_small[0],maxRadius=radii_small[1])
    
    if circles is not None and len(circles[0]) == 1:
        circles = np.int16(np.around(circles))
        circles = circles[0][0] #remove redundant dimensions

        radius_imaginary_big = circles[2]*factor
        alt = calculate_altitude(radius_big_circle=radius_imaginary_big)
        calculated_altitude = alt
        print("Altitude: ", alt)
        cv.circle(frame,(circles[0],circles[1]),circles[2],(0,255,0),2)
        # draw the center of the circle
        cv.circle(frame,(circles[0],circles[1]),2,(0,0,255),3)

    ##only for visual representation
    edges = cv.Canny(blur,0.5*cannyEdgeMaxThr,cannyEdgeMaxThr)
    combinedImage = np.concatenate((frame, edges), axis=1) #Combine canny edge detection and gray image
    cv.imshow('Circles and Canny', combinedImage) #Display the combined image
    cv.waitKey(1)














# avgTime = []
# while True:
#     start = time()
#     ret, frame = cap.read()

#     frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
#     img = cv.medianBlur(frame,5)

#     edges = cv.Canny(img,0.5*cannyEdgeMaxThr,cannyEdgeMaxThr)
    
#     circles_big = cv.HoughCircles(img,cv.HOUGH_GRADIENT,1,50,
#                                 param1=cannyEdgeMaxThr,param2=circleDetectThr,minRadius=int(size*factor),maxRadius=int(size*factor*rangePerc))
    
#     if circles_big is not None:
#         circles_big = np.int16(np.around(circles_big))
#         circles_big = circles_big[0] #remove redundant dimension

    
    
    
#     circles_small = cv.HoughCircles(img,cv.HOUGH_GRADIENT,1,50,
#                                 param1=cannyEdgeMaxThr,param2=circleDetectThr,minRadius=size,maxRadius=int(size*rangePerc))
#     if circles_small is not None:
#         circles_small = np.int16(np.around(circles_small))
#         circles_small = circles_small[0] #remove redundant dimension


#     if circles_big is not None and circles_small is not None:
#         testCirc = []
#         for big_c in circles_big:
#             for small_c in circles_small:
#                 distanceX = abs(big_c[0] - small_c[0])
#                 distanceY = abs(big_c[1] - small_c[1])
#                 ratio_big_small = big_c[2] / small_c[2]
                
#                 if np.sqrt(distanceX**2 + distanceY**2) < 10 and ratio_big_small > factor-0.4 and ratio_big_small < factor+0.4:
#                     print("Ratio: ", ratio_big_small)
#                     print("big_c: ", big_c, " small_c: ", small_c)
#                     testCirc.append(np.concatenate(([big_c], [small_c]), axis=0))
#                     print("match")
#                 else:
#                     print("ratio: ", ratio_big_small)
#                     print("distance: ", np.sqrt(distanceX**2 + distanceY**2))
#         if len(testCirc) > 0:
#             circles = testCirc
#             for i in circles[0]:
#                 # draw the outer circle
#                 cv.circle(frame,(i[0],i[1]),i[2],(0,255,0),2)
#                 # draw the center of the circle
#                 cv.circle(frame,(i[0],i[1]),2,(0,0,255),3)
#     else:
#         print("No circles found (either big or small)")

#     combinedImage = np.concatenate((frame, edges), axis=1) #Combine canny edge detection and gray image

#     cv.imshow('Circles and Canny', combinedImage) #Display the combined image
#     if cv.waitKey(1)  == ord('q'):
#         print("AVGTime: ", sum(avgTime)/len(avgTime))
#         break
#     end = time()
#     avgTime.append(end-start)
#     print("Time: ", end - start)
    
# cap.release()
# cv.destroyAllWindows()

