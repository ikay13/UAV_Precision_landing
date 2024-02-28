import sys
import cv2 as cv
import numpy as np
def main(argv):
    
    default_file = 'code_project/images/tins.jpg'
    filename = argv[0] if len(argv) > 0 else default_file
    # Loads an image
    src = cv.imread(cv.samples.findFile(filename), cv.IMREAD_COLOR)
    # Check if image is loaded fine
    if src is None:
        print ('Error opening image!')
        print ('Usage: hough_circle.py [image_name -- default ' + default_file + '] \n')
        return -1
    
    
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    
    
    gray = cv.medianBlur(gray, 5)
    
    
    rows = gray.shape[0]
    circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, rows / 8,
                               param1=50, param2=40,
                               minRadius=10, maxRadius=200)
    
    mask = np.zeros((src.shape[0], src.shape[1]), np.uint8)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            # circle center
            cv.circle(src, center, 1, (0, 100, 100), 3)
            # circle outline
            radius = i[2]
            cv.circle(src, center, radius, (255, 0, 255), 3)
            print("radius: ", radius)
            mask = cv.circle(mask, center, radius, (255, 255, 255), -1)
    
    cv.imshow("detected circles", mask)
    cv.waitKey(0)
    
    return 0
if __name__ == "__main__":
    main(sys.argv[1:])