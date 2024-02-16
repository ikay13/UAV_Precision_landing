import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
img = cv.imread('pad.jpeg', cv.IMREAD_GRAYSCALE)
assert img is not None, "file could not be read, check with os.path.exists()"
img = cv.medianBlur(img,5)
cimg = cv.cvtColor(img,cv.COLOR_GRAY2BGR)

print("test1")

circles = cv.HoughCircles(img,cv.HOUGH_GRADIENT,1,50,
                            param1=80,param2=60,minRadius=300,maxRadius=400)
circles = np.uint16(np.around(circles))

print("test2")



for i in circles[0,:]:
    # draw the outer circle
    cv.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the center of the circle
    cv.circle(cimg,(i[0],i[1]),2,(0,0,255),3)

print("test3")

plt.imshow(cimg)
plt.show()
