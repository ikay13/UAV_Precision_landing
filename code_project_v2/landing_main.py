from square_detect import detect_square_main, check_for_time
from coordinate_transform import transform_to_ground_xy, calculate_new_coordinate
from hogh_circles import concentric_circles, small_circle
from colour_tracking import find_red_tin

import cv2 as cv
import numpy as np
from enum import Enum

from pynput import keyboard

def on_press(key):
    if key == keyboard.Key.tab:
        global skip
        skip = True

def on_release(key):
    if key == keyboard.Key.esc:
        # Stop listener
        return False

# ...or, in a non-blocking fashion:
listener = keyboard.Listener(
    on_press=on_press,
    on_release=on_release)
listener.start()

class state(Enum):
    initial = 0
    fly_to_waypoint = 1
    detect_square = 2
    fly_to_target = 3
    check_for_target = 4
    descend_square = 5
    descend_concentric = 6
    descend_inner_circle = 7
    detect_tins = 8
    land_tins = 9
    return_to_launch = 10

class error_estimation_image:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.z = 0
        self.timeout = 0

class uav:
    def __init__(self):
        self.angle_x = 0
        self.angle_y = 0
        self.altitude = 7
        self.heading = 0
        self.latitude = 29.183972
        self.longitude = -81.043251
        self.state = state.initial
        self.cam_hfov = 65
        self.cam_vfov = 52

class target_parameters:
    def __init__(self):
        self.diameter_big = 0.72
        self.diameter_small = 0.24
        self.canny_max_threshold = 55 #param1 of hough circle transform
        self.hough_circle_detect_thr = 45 #param2 of hough circle transform
        self.factor = self.diameter_big/self.diameter_small #How much bigger the big circle is compared to the small circle (diameter)
        self.tin_diameter = 0.084 #Diameter of the tins in meters
        self.size_square = 2 #Size of the square in meters

def main():
    global skip
    skip = False
    video = cv.VideoCapture("images/cut.mp4")

    #only to not play the entire video
    fps = video.get(cv.CAP_PROP_FPS)
    start_time_video = 4
    start_frame_num = int(start_time_video * fps)
    video.set(cv.CAP_PROP_POS_FRAMES, start_frame_num)

    err_estimation = error_estimation_image()  # Create an instance of the error_estimation class
    uav_inst = uav()  # Create an instance of the uav_angles class
    target_parameters_obj = target_parameters()

    uav_inst.state = state.descend_square


    check_for_time.start_time = None
    reached_waypoint = "n"
    reached_target = "n"
    while video.isOpened():
        ret, frame = video.read()
        if ret:
            
            if uav_inst.state == state.descend_concentric or uav_inst.state == state.descend_inner_circle or uav_inst.state == state.land_tins:
                frame = cv.resize(frame, (850, 478))
                #frame = cv.resize(frame, (9*50, 16*50))
            else:
                frame = cv.resize(frame, (640, 480))
            match uav_inst.state:
                case state.initial:
                    uav_inst.state = state.fly_to_waypoint

                case state.fly_to_waypoint:
                    #############################################
                    ###Fly to waypoint using lat and lon#########
                    #############################################

                    ##This check will be done with mavros data
                    if reached_waypoint == "n":
                        reached_waypoint = input("Have you reached the waypoint? (y/n)")
                    if reached_waypoint == "y":
                        uav_inst.state = state.detect_square

                case state.detect_square:   
                    print("Detecting square")                 
                    #Check for 3s if square is detected more than 50% of the time
                    square_detected_err = check_for_time(frame=frame, altitude=uav_inst.altitude,duration=3,
                                                         ratio_detected=0.5, size_square=target_parameters_obj.size_square, cam_hfov=uav_inst.cam_hfov)
                    if square_detected_err is None:
                        pass #time not over yet
                    elif square_detected_err is False:
                        uav_inst.state = state.fly_to_waypoint
                        reached_waypoint = "n"
                        print("Square not detected. Flying to next waypoint.")
                        ################################################
                        ####Fly to next waypoint, no square detected####
                        ################################################
                        break
                    else:
                        uav_inst.state = state.fly_to_target
                        print("Square detected: " + str(square_detected_err))

                        #Transform the error to target to lat and lon
                        err_estimation.x = square_detected_err[0]
                        err_estimation.y = square_detected_err[1]
                        error_ground_xy = transform_to_ground_xy([err_estimation.x, err_estimation.y], [uav_inst.angle_x, uav_inst.angle_y], uav_inst.altitude)
                        target_lat, target_lon = calculate_new_coordinate(uav_inst.latitude, uav_inst.longitude, error_ground_xy, uav_inst.heading)
                    
                case state.fly_to_target:
                    #############################################
                    #Fly to estimated waypoint using lat and lon#
                    #############################################


                    #Detect square again
                    #Check for 2s if square is detected
                    #check will be done using mavros data
                    if reached_target == "n":
                        reached_target = input("Have you reached the target? (y/n)")
                    if reached_target == "y":
                        check_for_time.start_time = None
                        print("restart time")
                        uav_inst.state = state.check_for_target
                        
                case state.check_for_target:
                    #check for 2s if square is detected more than 80% of the time
                    square_detected_err = check_for_time(frame=frame, altitude=uav_inst.altitude,duration=2,ratio_detected=0.9)

                    if square_detected_err is None:
                        continue #time not over yet
                    elif square_detected_err is False:
                        uav_inst.state = state.return_to_launch
                        reached_target = "over"
                        print("Returning to launch")
                    else:
                        #Set mode to using square as reference
                        uav_inst.state = state.descend_square
                        reached_target = "over"
                        print("Target detected: " + str(square_detected_err) + "now descending")

                case state.descend_square:
                    error_xy, area_ratio = detect_square_main(frame, uav_inst.altitude, target_parameters_obj.size_square, uav_inst.cam_hfov)
                    if error_xy is not None:
                        err_estimation.x = error_xy[0]
                        err_estimation.y = error_xy[1]
                        error_ground_xy = transform_to_ground_xy([err_estimation.x, err_estimation.y], [uav_inst.angle_x, uav_inst.angle_y], uav_inst.altitude)
                        print("Area ratio " + str(area_ratio))
                    if skip: #should instead be a check whether area ratio is bigger than threshold
                        uav_inst.state = state.descend_concentric
                        ############################################
                        uav_inst.altitude = 2 ###################################to be removed###################''!!!!!!!!
                        #############################################
                        skip = False
                        video = cv.VideoCapture("images/concentric_to_single_rot.mp4")
                        fps = video.get(cv.CAP_PROP_FPS)
                        start_time_video = 0
                        start_frame_num = int(start_time_video * fps)
                        video.set(cv.CAP_PROP_POS_FRAMES, start_frame_num)
                                        

                case state.descend_concentric:
                    #####/3 should be removed later
                    alt, error_xy = concentric_circles(frame=frame, altitude=uav_inst.altitude/3, cam_hfov=uav_inst.cam_hfov, circle_parameters_obj=target_parameters_obj) 
                    if alt is not None:
                        uav_inst.altitude = alt
                        uav_inst.error_xy = error_xy
                    if uav_inst.altitude < 1.5:
                        uav_inst.state = state.descend_inner_circle
                        print("Switching to inner circle")

                case state.descend_inner_circle:
                    #####/3 should be removed later
                    alt,error_xy = small_circle(frame=frame, altitude=uav_inst.altitude/3, cam_hfov=uav_inst.cam_hfov, circle_parameters_obj=target_parameters_obj)
                    if alt is not None:
                        uav_inst.altitude = alt
                        uav_inst.error_xy = error_xy
                    if uav_inst.altitude < 0.8:
                        uav_inst.state = state.land_tins
                        print("Switching to detectig tins")
                        uav_inst.state = state.detect_tins
                        video = cv.VideoCapture("images/tin_white_rot.mp4")
                        fps = video.get(cv.CAP_PROP_FPS)
                        start_time_video = 0
                        start_frame_num = int(start_time_video * fps)
                        video.set(cv.CAP_PROP_POS_FRAMES, start_frame_num)
                case state.detect_tins:
                    pass

                case state.land_tins:

                    pass
                case state.return_to_launch:
                    pass
        else:
            break

      

if __name__ == "__main__":
    main()
