import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from time import time

###Parameters
cannyEdgeMaxThr = 40 #Max Thr for canny edge detection
circleDetectThr = 35 #Threshold for circle detection
size = 20           #Size of the circles (to be calculated)
factor = 2.25          #Factor big circle diameter / small circle diameter
rangePerc = 1.5     #This is the range the circles are expected to be in

cap = cv.VideoCapture(0)
plt.ion()

def findCircle(img, hough_gradient, dp, minDist, param1, param2, minRadius, maxRadius):
    circles = cv.HoughCircles(img, hough_gradient, dp, minDist, param1, param2, minRadius, maxRadius)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        circles = circles[0] #remove redundant dimension
    return circles

avgTime = []
while True:
    start = time()
    ret, frame = cap.read()

    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    img = cv.medianBlur(frame,5)

    edges = cv.Canny(img,0.5*cannyEdgeMaxThr,cannyEdgeMaxThr)
    
    circles_big = cv.HoughCircles(img,cv.HOUGH_GRADIENT,1,50,
                                param1=cannyEdgeMaxThr,param2=circleDetectThr,minRadius=int(size*factor),maxRadius=int(size*factor*rangePerc))
    
    if circles_big is not None:
        circles_big = np.uint16(np.around(circles_big))
        circles_big = circles_big[0] #remove redundant dimension

    
    
    
    circles_small = cv.HoughCircles(img,cv.HOUGH_GRADIENT,1,50,
                                param1=cannyEdgeMaxThr,param2=circleDetectThr,minRadius=size,maxRadius=int(size*rangePerc))
    if circles_small is not None:
        circles_small = np.uint16(np.around(circles_small))
        circles_small = circles_small[0] #remove redundant dimension


    if circles_big is not None and circles_small is not None:
        testCirc = []
        for big_c in circles_big:
            for small_c in circles_small:
                distanceX = abs(big_c[0] - small_c[0])
                distanceY = abs(big_c[1] - small_c[1])
                ratio_big_small = big_c[2] / small_c[2]
                
                if np.sqrt(distanceX**2 + distanceY**2) < 10 and ratio_big_small > 2.1 and ratio_big_small < 2.4:
                    print("Ratio: ", ratio_big_small)
                    print("big_c: ", big_c, " small_c: ", small_c)
                    testCirc.append(np.concatenate(([big_c], [small_c]), axis=0))
                    print("match")
        if len(testCirc) > 0:
            circles = testCirc
            for i in circles[0]:
                # draw the outer circle
                cv.circle(frame,(i[0],i[1]),i[2],(0,255,0),2)
                # draw the center of the circle
                cv.circle(frame,(i[0],i[1]),2,(0,0,255),3)
        

    combinedImage = np.concatenate((frame, edges), axis=1) #Combine canny edge detection and gray image

    cv.imshow('Circles and Canny', combinedImage) #Display the combined image
    if cv.waitKey(1)  == ord('q'):
        print("AVGTime: ", sum(avgTime)/len(avgTime))
        break
    end = time()
    avgTime.append(end-start)
    print("Time: ", end - start)
    
cap.release()
cv.destroyAllWindows()

