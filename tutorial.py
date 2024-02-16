import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
img = cv.imread('landingPad.jpeg', cv.IMREAD_GRAYSCALE)
assert img is not None, "file could not be read, check with os.path.exists()"
#laplacian = cv.Laplacian(img,cv.CV_64F)
sobelx = cv.Sobel(img,cv.CV_64F,1,0,ksize=5)
sobely = cv.Sobel(img,cv.CV_64F,0,1,ksize=5)

addedImg = np.hypot(sobelx, sobely)
addedImg = addedImg - addedImg.min()
addedImg = addedImg / addedImg.max() * 255

addedImg = np.uint8(addedImg)

# Otsu's thresholding after Gaussian filtering
blur = cv.GaussianBlur(addedImg,(15,15),0)
#ret3,th3 = cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
th3 = cv.adaptiveThreshold(blur,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV,11,2)

kernel = np.ones((3,3),np.uint8)
erosion = cv.erode(th3,kernel,iterations = 1)
dilation = cv.dilate(erosion,kernel,iterations = 1)

plt.subplot(2,2,1),plt.imshow(addedImg,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,2),plt.imshow(erosion,cmap = 'gray')
plt.title('sobelx'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,3),plt.imshow(dilation,cmap = 'gray')
plt.title('sobel added'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,4),plt.imshow(dilation,cmap = 'gray')
plt.title('threshhold'), plt.xticks([]), plt.yticks([])
plt.show()