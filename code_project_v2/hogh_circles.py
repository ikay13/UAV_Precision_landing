import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from time import time
from math import atan2

# ###Parameters
# cannyEdgeMaxThr = 40 #Max Thr for canny edge detection
# circleDetectThr = 35 #Threshold for circle detection
# size = 30           #Size of the circles (to be calculated)
# factor = 3.2          #Factor big circle diameter / small circle diameter
# rangePerc = 1.5     #This is the range the circles are expected to be in

#cap = cv.VideoCapture(0)
#plt.ion()

def concentric_circles(frame, altitude, cam_hfov):
    ###Parameters
    cannyEdgeMaxThr = 55 #Max Thr for canny edge detection
    circleDetectThr = 40 #Threshold for circle detection
    factor = 3.2          #Factor big circle diameter / small circle diameter
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

    edges = cv.Canny(blur,0.5*cannyEdgeMaxThr,cannyEdgeMaxThr)
    
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
            for i in circles[0]:
                print("radius big: ", i[2]) #Print the radius of the big circle
                # draw the outer circle
                cv.circle(frame,(i[0],i[1]),i[2],(0,255,0),2)
                # draw the center of the circle
                cv.circle(frame,(i[0],i[1]),2,(0,0,255),3)
    # elif circles_big is not None:
    #     print("No small circles found")
    #     print("big: ", circles_big)
    #     for i in circles_big:
    #         # draw the outer circle
    #         cv.circle(frame,(i[0],i[1]),i[2],(0,255,0),2)
    #         # draw the center of the circle
    #         cv.circle(frame,(i[0],i[1]),2,(0,0,255),3)
    else:
        print("No circles found (either big or small)")
        # print("big: ", circles_big)
        # print("small: ", circles_small)
        # print("radii_big: ", radii_big)

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

