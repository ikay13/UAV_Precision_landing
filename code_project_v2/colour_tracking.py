# Python code for Multiple Color Detection 


import numpy as np 
import cv2 as cv
import matplotlib.pyplot as plt


cam = cv.VideoCapture("images/tins_1.mp4")

def find_red_tin(hsvFrame, minsize):
	# Set range for red color and 
	# define mask 
	red_lower = np.array([330//2, 30, 50], np.uint8) 
	red_upper = np.array([186, 255, 255], np.uint8) 
	red_mask = cv.inRange(hsvFrame, red_lower, red_upper) 
	
	
	# Morphological Transform, Dilation 
	# for each color and bitwise_and operator 
	# between imageFrame and mask determines 
	# to detect only that particular color
	kernel = np.ones((5, 5), "uint8") 
	
	#####Search for the red tin#####
	# For red color 
	red_mask = cv.dilate(red_mask, kernel) 
	# Creating contour to track red color 
	cnts, hierarchy = cv.findContours(red_mask, 
										cv.RETR_TREE, 
										cv.CHAIN_APPROX_SIMPLE) 
	min_circle_red = [] #This will be the minimum enclosing circle [x, y, radius] of the red tin
	red_tin_exists = False
	for cnt in cnts: 
		area = cv.contourArea(cnt) 
		if(area > 1000): #Checks if the contour is big enough to be a tin
				temp_circle = cv.minEnclosingCircle(cnt)
				center = (int(temp_circle[0][0]), int(temp_circle[0][1]))
				radius = int(temp_circle[1])
				min_circle_red = [center, radius]
				# cv.circle(imageFrame, center, 5, (0, 0, 255), -1)
				# cv.circle(imageFrame, center, radius, (0, 0, 255), 2)
				red_tin_exists = True

				cv.drawContours(imageFrame, [cnt], -1, (0, 0, 255), 2)
	#Return the minimum enclosing circle of the red tin and a bool indicating if the red tin exists
	return min_circle_red, red_tin_exists 













red_found_cnt = 0
red_notfound_cnt = 0
while(False):
	
	ret,frame = cam.read()
	if ret:
		# Reading the video from the 
		# webcam in image frames 
		imageFrame = cv.resize(frame, (360, 540))
		imageFrame = cv.GaussianBlur(imageFrame, (15, 15), 0)

		# Convert the imageFrame in 
		# BGR to HSV color space
		hsvFrame = cv.cvtColor(imageFrame, cv.COLOR_BGR2HSV) 

		min_circle_red, red_tin_exists = find_red_tin(hsvFrame, 1000)
		if red_tin_exists: 
			red_found_cnt += 1
			###################################################
			########Landing pad for picking up the tin#########
			###################################################

			img_gray = cv.cvtColor(imageFrame, cv.COLOR_BGR2GRAY)

			#####find tins using HoughCircles#####
			#min and max radius of the circle based on the radius of the red tin
			rows = img_gray.shape[0]
			min_radius = min_circle_red[1] - rows//50
			max_radius = min_circle_red[1] + rows//50
			circles = cv.HoughCircles(img_gray, cv.HOUGH_GRADIENT, 1, rows / 8,
								param1=35, param2=30,
								minRadius=min_radius, maxRadius=max_radius)
			
			####Create masks for the blue and green tins######
			masks = [np.zeros((img_gray.shape[0], img_gray.shape[1]), np.uint8) for _ in range(2)] #List of masks for each circle
			centers = [[],[]] #List of the centers of the circles (green and blue tins) in same order as masks
			if circles is not None and len(circles[0]) == 3:	#If 3 circles are found, draw the masks
				mask_idx = 0
				circles = np.uint16(np.around(circles)) #Convert the coordinates and radius to integers
				for idx in range(len(circles[0])): #For each circle
					center = (circles[0][idx][0], circles[0][idx][1]) 
					# distance from the center of the red tin to the center of the circle
					dist_to_red = np.sqrt((center[0] - min_circle_red[0][0])**2 + (center[1] - min_circle_red[0][1])**2)
					if dist_to_red > min_circle_red[1]: #This is not the red circle
						radius = circles[0][idx][2]
						centers[mask_idx] = center
						cv.circle(masks[mask_idx], center, radius-15, (255, 255, 255), -1) ######Modify -15 to a better value
						mask_idx += 1


				#####Apply the masks to the image#####
				average_colors = []
				for current_mask in masks: #For each hsv mask (blue and green)
					average_colors.append(cv.mean(hsvFrame, mask=current_mask)) #Find the average color of the masked image

				#for idx in range(len(average_colors)):
					#print("Average color of tin", idx, ":", average_colors[idx])

				#If the average hue of the first tin is greater than the second tin, the first tin is blue
				if average_colors[0][0] > average_colors[1][0]: 
					cv.circle(imageFrame, centers[0], min_circle_red[1], (255, 0, 0), 2)
					cv.circle(imageFrame, centers[1], min_circle_red[1], (0, 255, 0), 2)
				else:
					cv.circle(imageFrame, centers[0], min_circle_red[1], (0, 255, 0), 2)
					cv.circle(imageFrame, centers[1], min_circle_red[1], (255, 0, 0), 2)

				mask = masks[0] + masks[1] #combine the masks for display
				circles_coords = [[]*3] #List of the coordinates of the circles in x,y format
				
				

				# Display the image
				mask = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
				concat_img = cv.hconcat([mask, imageFrame])
				cv.imshow("Multiple Color Detection in Real-TIme", concat_img) 
		else:
			red_notfound_cnt += 1

		
		#####Program Termination#####
		if cv.waitKey(10) & 0xFF == ord('q'): 
			cam.release() 
			cv.destroyAllWindows() 
			break

	else:
		break
print("Red found: ", red_found_cnt, "Red not found: ", red_notfound_cnt)
cam.release()


