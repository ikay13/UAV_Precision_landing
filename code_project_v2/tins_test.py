# Python code for Multiple Color Detection 


import numpy as np 
import cv2 as cv

#ting_white_rot
#blue_back_full
video = cv.VideoCapture("images/blue_back_full.mp4")

#only to not play the entire video
fps = video.get(cv.CAP_PROP_FPS)
start_time_video = 8
start_frame_num = int(start_time_video * fps)
video.set(cv.CAP_PROP_POS_FRAMES, start_frame_num)

# Start a while loop 
while(video.isOpened()): 
	
	# Reading the video from the 
	# webcam in image frames 
	_, imageFrame = video.read() 
	if not _:  #If the frame is empty, break immediately
		break
	
    # Convert the imageFrame in 
	# BGR(RGB color space) to 
	# HSV(hue-saturation-value) 
	# color space 
	hsvFrame = cv.cvtColor(imageFrame, cv.COLOR_BGR2HSV) 

	# Set range for red color and 
	# define mask 
	red_lower = np.array([136, 87, 111], np.uint8) 
	red_upper = np.array([180, 255, 255], np.uint8) 
	red_mask = cv.inRange(hsvFrame, red_lower, red_upper) 

	# # Set range for green color and 
	# # define mask 
	# green_lower = np.array([25, 52, 72], np.uint8) 
	# green_upper = np.array([102, 255, 255], np.uint8) 
	# green_mask = cv.inRange(hsvFrame, green_lower, green_upper) 

	# Set range for blue color and 
	# define mask 
	blue_lower = np.array([25, 52, 2], np.uint8) 
	blue_upper = np.array([120, 255, 255], np.uint8) 
	blue_mask = cv.inRange(hsvFrame, blue_lower, blue_upper) 
	
	# Morphological Transform, Dilation 
	# for each color and bitwise_and operator 
	# between imageFrame and mask determines 
	# to detect only that particular color 
	kernel = np.ones((5, 5), "uint8") 
	
	# For red color 
	red_mask = cv.dilate(red_mask, kernel) 
	res_red = cv.bitwise_and(imageFrame, imageFrame, 
							mask = red_mask) 
	
	# # For green color 
	# green_mask = cv.dilate(green_mask, kernel) 
	# res_green = cv.bitwise_and(imageFrame, imageFrame, 
	# 							mask = green_mask) 
	
	# For blue color 
	blue_mask = cv.dilate(blue_mask, kernel) 
	res_blue = cv.bitwise_and(imageFrame, imageFrame, 
							mask = blue_mask) 

	# Creating contour to track red color 
	contours, hierarchy = cv.findContours(red_mask, 
										cv.RETR_TREE, 
										cv.CHAIN_APPROX_SIMPLE) 
	
	for pic, contour in enumerate(contours): 
		area = cv.contourArea(contour) 
		if(area > 300): 
			x, y, w, h = cv.boundingRect(contour) 
			imageFrame = cv.rectangle(imageFrame, (x, y), 
									(x + w, y + h), 
									(0, 0, 255), 2) 
			
			cv.putText(imageFrame, "Red Colour", (x, y), 
						cv.FONT_HERSHEY_SIMPLEX, 1.0, 
						(0, 0, 255))	 

	# # Creating contour to track green color 
	# contours, hierarchy = cv.findContours(green_mask, 
	# 									cv.RETR_TREE, 
	# 									cv.CHAIN_APPROX_SIMPLE) 
	
	# for pic, contour in enumerate(contours): 
	# 	area = cv.contourArea(contour) 
	# 	if(area > 300): 
	# 		x, y, w, h = cv.boundingRect(contour) 
	# 		imageFrame = cv.rectangle(imageFrame, (x, y), 
	# 								(x + w, y + h), 
	# 								(0, 255, 0), 2) 
			
	# 		cv.putText(imageFrame, "Green Colour", (x, y), 
	# 					cv.FONT_HERSHEY_SIMPLEX, 
	# 					1.0, (0, 255, 0)) 

	# Creating contour to track blue color 
	contours, hierarchy = cv.findContours(blue_mask, 
										cv.RETR_TREE, 
										cv.CHAIN_APPROX_SIMPLE) 
	for pic, contour in enumerate(contours): 
		area = cv.contourArea(contour) 
		if(area > 300): 
			x, y, w, h = cv.boundingRect(contour) 
			imageFrame = cv.rectangle(imageFrame, (x, y), 
									(x + w, y + h), 
									(255, 0, 0), 2) 
			
			cv.putText(imageFrame, "Blue Colour", (x, y), 
						cv.FONT_HERSHEY_SIMPLEX, 
						1.0, (255, 0, 0)) 
			
	# Program Termination 
	cv.imshow("Multiple Color Detection in Real-TIme", imageFrame) 
	if cv.waitKey(10) & 0xFF == ord('q'): 
		cap.release() 
		cv.destroyAllWindows() 
		break
