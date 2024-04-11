import cv2 as cv
import numpy as np

import matplotlib.pyplot as plt




def thresholding(path_to_image):
    img = cv.imread(path_to_image, cv.IMREAD_GRAYSCALE) #read as grayscale
    assert img is not None, "file could not be read, check with os.path.exists()"

    img_cols = img.shape[1] #count number of pixel columns

    size_blur = int(img_cols / 100) #set size of blur filter relative to image size
    if size_blur % 2 == 0: #make sure size is odd
        size_blur += 1

    blur = cv.GaussianBlur(img,(size_blur,size_blur),0)

    # find normalized_histogram, and its cumulative distribution function
    hist = cv.calcHist([blur],[0],None,[256],[0,256])
    hist_norm = hist.ravel()/hist.sum()

    #Calculate the optimal threshold using Otsu's method (modified to set different threshold value)
    ####Parameter########
    otsu_factor = 200 #a higher value will result in a higher threshold
    #####################
    Q = hist_norm.cumsum()
    bins = np.arange(256)
    fn_min = np.inf
    thresh = -1
    for i in range(1,256):
        p1,p2 = np.hsplit(hist_norm,[i]) # probabilities
        q1,q2 = Q[i],Q[255]-Q[i] # cum sum of classes
        if q1 < 1.e-6 or q2 < 1.e-6:
            continue
        b1,b2 = np.hsplit(bins,[i]) # weights
        # finding means and variances
        m1,m2 = np.sum(p1*b1)/q1, np.sum(p2*b2)/q2
        v1,v2 = np.sum(((b1-m1)**2)*p1)/q1,np.sum(((b2-m2)**2)*p2)/q2
        # calculates the minimization function
        fn = v1*q1 + v2*q2*otsu_factor #adjust to set different threshold
        if fn < fn_min:
            fn_min = fn
            thresh = i

    #Use this threshold to convert the image to binary
    ret, th_adj = cv.threshold(blur, thresh,255,cv.THRESH_BINARY)

    #dilate to close gaps and circles in landing pad
    size_dilation = int(img_cols / 100)
    dilation = cv.dilate(th_adj, np.ones((size_dilation,size_dilation),np.uint8), iterations=1)

    #erode to remove noise
    size_erode = int(img_cols / 30)
    erosion = cv.erode(dilation, np.ones((size_erode,size_erode),np.uint8), iterations=1)

    #dilate to restore size of landing pad
    size_dilation_2 = int(img_cols / 50)
    dilation_2 = cv.dilate(erosion, np.ones((size_dilation_2,size_dilation_2),np.uint8), iterations=1)

    return dilation_2 #return the final image for further processing

    """ plt.subplot(331), plt.plot(hist_norm), plt.xlabel('Pixel Value')
    plt.axvline(x=thresh, color='r', linestyle='--')
    plt.ylabel('Normalized Frequency'), plt.title('Histogram')

    plt.subplot(332), plt.imshow(th_adj, 'gray'), plt.title('Otsu Thresholding adjusted')
    plt.subplot(333), plt.imshow(otsu, 'gray'), plt.title('Otsu original')
    plt.subplot(334), plt.imshow(blur, 'gray'), plt.title('blurred image')
    plt.subplot(335), plt.imshow(img, 'gray'), plt.title('original image')
    plt.subplot(336), plt.imshow(dilation, 'gray'), plt.title('dilated image')
    plt.subplot(337), plt.imshow(erosion, 'gray'), plt.title('dilated eroded')
    plt.subplot(338), plt.imshow(dilation_2, 'gray'), plt.title('dilated eroded dilated')

    plt.show()
 """
thresholding("code_project/images/out_insun.png")