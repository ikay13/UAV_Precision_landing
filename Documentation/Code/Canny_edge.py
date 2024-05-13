import cv2
import numpy as np
from matplotlib import pyplot as plt
import math

def concentric_circles(frame, altitude=1.5, cam_hfov=65):
    """Detects concentric circles in the image using altitude"""
    frame_gray = frame
    blur = cv2.medianBlur(frame_gray,3)

    #Threshold image to get only the area inside square plus other bright spots (reduces edges in image)
    blur_otsu = cv2.GaussianBlur(blur, (11, 11), 0)
    _, thr = cv2.threshold(blur_otsu, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    blur = cv2.bitwise_and(blur, thr)


    ###Parameters
    cannyEdgeMaxThr = 50 #Max Thr for canny edge detection
    circleDetectThr = 20#Threshold for circle detection
    factor = 1.7 #Factor big circle diameter / small circle diameter
    tolerance = 1.5     #This is the tolarance the circles are expected to be in

    calculated_altitude = None #This is the altitude calculated from the image (As the circel dimensions are known)

    ###Calculate the size of the circles relative to altitude and camera hfov
    dist_img_on_ground = math.tan(cam_hfov/2)*2*altitude
    actual_radius = 0.72/2
    rel_size = actual_radius/dist_img_on_ground #This is the size of the bigger circle compared to the overall frame
    radius_big_pixel = rel_size*frame_gray.shape[1] #This is the radius of the bigger circle in pixels
    radius_small_pixel = radius_big_pixel/factor #This is the radius of the smaller circle in pixels
    radii_big = [int(radius_big_pixel/tolerance), int(radius_big_pixel*tolerance)] #Min and max radius for the bigger circle
    radii_small = [int(radius_small_pixel/tolerance), int(radius_small_pixel*tolerance)] #Min and max radius for the smaller circle
    print(radii_big)
    print(radii_small)

    edges = cv2.Canny(blur,0.5*cannyEdgeMaxThr,cannyEdgeMaxThr) #Only for visual representation (hough already does this)
    
    ###Find big circles
    circles_big = cv2.HoughCircles(blur,cv2.HOUGH_GRADIENT,1,50,
                                param1=cannyEdgeMaxThr,param2=circleDetectThr,minRadius=radii_big[0],maxRadius=radii_big[1])
    
    if circles_big is not None:
        circles_big = np.int16(np.around(circles_big))
        circles_big = circles_big[0] #remove redundant dimension


    ###Find small circles
    circles_small = cv2.HoughCircles(blur,cv2.HOUGH_GRADIENT,1,50,
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
            #print("Center: ", error_xy)
            for i in circles[0]:

                #print("Altitude: ", alt)
                #This is drawn on orignal frame image passed to function and not a copy
                # draw the outer circle
                cv2.circle(frame,(i[0],i[1]),i[2],(0,0,0),2)
                # draw the center of the circle
                cv2.circle(frame,(i[0],i[1]),2,(0,0,0),3)
        else:
            #No concentric circles found
            return None, edges
    else:
        #No circles found (either big or small)
        
        # combinedImage = np.concatenate((frame, edges), axis=1) #Combine canny edge detection and gray image
        # cv2.imshow('Circles and Canny', combinedImage) #Display the combined image
        # cv2.waitKey(1)
        return None, edges
        

    # combinedImage = np.concatenate((frame, edges), axis=1) #Combine canny edge detection and gray image
    # cv2.imshow('Circles and Canny', combinedImage) #Display the combined image
    # cv2.waitKey(1)

    return calculated_altitude, edges

# Load the image in grayscale mode
image = cv2.imread('Documentation/Images/concentric_scaled_water.png', cv2.IMREAD_GRAYSCALE)

# Check if image has loaded correctly
if image is None:
    print("Error: Image could not be read.")
    exit()

alt, edges = concentric_circles(frame=image)

# Display and save the original image
plt.figure(figsize=(5, 5))
plt.imshow(image, cmap='gray')
plt.axis('off')  # Hides the axis
plt.savefig('Documentation/Images/finished/concentric_original_fitted_img.png', bbox_inches='tight')

# Display and save the edge-detected image
plt.figure(figsize=(5, 5))
plt.imshow(edges, cmap='gray')
plt.axis('off')  # Hides the axis
plt.savefig('Documentation/Images/finished/canny_edges.png', bbox_inches='tight')

plt.show()
