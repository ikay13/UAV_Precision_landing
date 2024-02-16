import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('landingPad.jpeg')
assert img is not None, "file could not be read, check with os.path.exists()"




imgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

ret, thresh = cv.threshold(imgray, 127, 255, 0)


contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

# Calculate the lengths of the contours
contour_lengths = [cv.arcLength(cnt, True) for cnt in contours]

# Get the indices of the three longest contours
longest_contours_indices = sorted(range(len(contour_lengths)), key=lambda i: contour_lengths[i], reverse=True)[:3]

for idx in range(3):
    cnt = contours[longest_contours_indices[idx]]
    cv.drawContours(img, [cnt], 0, (0,255,0), 3)
    
    (x,y),radius = cv.minEnclosingCircle(cnt)
    center = (int(x),int(y))
    radius = int(radius)
    cv.circle(img,center,radius,(0,255,0),2)



plt.imshow(img)
plt.show()
