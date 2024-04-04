from square_detect import detect_square_main, check_for_time
from coordinate_transform import transform_to_ground_xy, calculate_new_coordinate, transform_ground_to_img_xy
from hogh_circles import concentric_circles, small_circle
from tin_detection import tin_detection_for_time, tins_error_bin_mode
from debugging_code import display_error_and_text
import navigate_pixhawk
import cv2 as cv
import numpy as np
from enum import Enum
from math import atan2, pi, sqrt, exp
import time

from pynput import keyboard

#Only for debugging
def on_press(key):
    if key == keyboard.Key.tab:
        global skip
        skip = True

def on_release(key):
    if key == keyboard.Key.esc:
        # Stop listener
        return False

listener = keyboard.Listener(
    on_press=on_press,
    on_release=on_release)
listener.start()

class state:
    def __init__(self):
        self.initial = 0
        self.fly_to_waypoint = 1
        self.detect_square = 2
        self.fly_to_target = 3
        self.check_for_target = 4
        self.descend_square = 5
        self.descend_concentric = 6
        self.descend_inner_circle = 7
        self.detect_tins = 8
        self.fly_over_tins = 9
        self.land = 10
        self.return_to_launch = 11
        self.climb_to_altitude = 12
        self.done = 13
    

class error_estimation:
    def __init__(self):
        self.x_img = 0
        self.y_img = 0
        self.x_m = 0
        self.y_m = 0

        self.x_m_filt = []
        self.y_m_filt = []
        self.err_distances_filt = []
        self.err_times_filt = []
        self.altitude_m_filt = []

        self.x_m_avg = 0
        self.y_m_avg = 0
        self.altitude_m_avg = 0

        self.altitude_m = 0
        self.target_lat = 0
        self.target_lon = 0
        self.list_length = 5
        self.err_px_x = 0
        self.err_px_y = 0
        self.p_value = 0.95
        self.time_last_detection = time.time()

    def update_errors(self, x_img, y_img, altitude_m, uav_lat, uav_lon, heading, cam_fov_hv, image_size):
        """Update the errors in the image and ground plane and the target lat and lon"""     
        #Assign current errors
        self.x_img = x_img
        self.y_img = y_img
        self.altitude_m = altitude_m
        #Calculate the errors in the ground plane (dependent on altitude of UAV)
        error_ground_xy = transform_to_ground_xy([x_img, y_img],altitude_m, cam_fov_hv)
        self.x_m = error_ground_xy[0]
        self.y_m = error_ground_xy[1]

        #Keep list a maximum of self.list_length elements
        if len(self.x_m_filt) == self.list_length:
            self.x_m_filt.pop(0)
            self.y_m_filt.pop(0)
            self.err_distances_filt.pop(0)
            self.err_times_filt.pop(0)
            self.altitude_m_filt.pop(0)

        #Add the current errors to the list
        self.x_m_filt.append(self.x_m)
        self.y_m_filt.append(self.y_m)
        self.err_distances_filt.append(sqrt(self.x_m**2 + self.y_m**2 + self.altitude_m**2))
        self.err_times_filt.append(time.time())
        self.altitude_m_filt.append(altitude_m)
        
        #Reset average values
        self.x_m_avg = 0
        self.y_m_avg = 0
        self.altitude_m_avg = 0
        weight_total = 0

        #Calculate the weighted average of the errors
        for idx in range(len(self.x_m_filt)):
            time_diff = time.time() - self.err_times_filt[idx]
            weight_time = exp(-time_diff)
            weight_dist = 10-self.err_distances_filt[idx]
            if weight_dist < 2:
                weight_dist = 2
            weight_result = weight_time*weight_dist
            weight_total += weight_result
            self.x_m_avg += self.x_m_filt[idx]*weight_result
            self.y_m_avg += self.y_m_filt[idx]*weight_result
            self.altitude_m_avg += self.altitude_m_filt[idx]*weight_result
        self.x_m_avg = self.x_m_avg/weight_total
        self.y_m_avg = self.y_m_avg/weight_total
        self.altitude_m_avg = self.altitude_m_avg/weight_total

        #Calculate the target lat and lon using the avg value
        self.target_lat, self.target_lon = calculate_new_coordinate(uav_lat, uav_lon, [self.x_m_avg, self.y_m_avg], heading)

        #Calculate the error in pixels using the avg value
        curr_err_px = transform_ground_to_img_xy((self.x_m_avg, self.y_m_avg), self.altitude_m_avg, cam_fov_hv, image_size)
        self.err_px_x = curr_err_px[0]
        self.err_px_y = curr_err_px[1]
        self.time_last_detection = time.time()
    
    def clear_errors(self):
        """Clears all errors from list (used when drone moved to new point)"""
        self.x_m_filt = []
        self.y_m_filt = []
        self.err_distances_filt = []
        self.err_times_filt = []
        self.altitude_m_filt = []

        self.x_m_avg = 0
        self.y_m_avg = 0
        self.altitude_m_avg = 0

    def check_for_timeout(self):
        """Check if the time since last detection of an object is more than 3s"""
        if time.time() - self.time_last_detection > 15:################################change later
            return True
        return False


class uav(state):
    def __init__(self):
        state.__init__(self)
        self.altitude = 6 #Altitude of the fc in meters
        self.heading = 0 #Heading of the UAV in radians
        self.latitude = 29.183972 #Current of the UAV
        self.longitude = -81.043251 #Current longitude of the UAV
        self.state = self.initial
        self.cam_hfov = 60*pi/180 #Input by user
        self.cam_vfov = 34*pi/180 #Input by user
        self.image_size = [850, 478] #Automatically updated by the code

class target_parameters:
    def __init__(self):
        #Change these values if the size of the target changes
        self.diameter_big = 0.72
        self.diameter_small = 0.24
        self.canny_max_threshold = 45 #param1 of hough circle transform
        self.hough_circle_detect_thr = 40 #param2 of hough circle transform
        self.factor = self.diameter_big/self.diameter_small #How much bigger the big circle is compared to the small circle (diameter)
        self.tin_diameter = 0.084 #Diameter of the tins in meters
        self.size_square = 2 #Size of the square in meters

class waypoints:
    def __init__(self):
        self.waypoints = []


class tin_colours:
    def __init__(self):
        #Change this if the colour of the tins changes (hue value from 0 to 180)
        self.green_hue = 95
        self.blue_hue = 105
        self.red_hue = 170

def main():
    global skip
    skip = False
    #blue_back_full.mp4
    #two_platforms.mp4
    #cut.mp4
    video = cv.VideoCapture("images/test_data2.mp4")

    #only to not play the entire video
    fps = video.get(cv.CAP_PROP_FPS)
    start_time_video = 27
    start_frame_num = int(start_time_video * fps)
    video.set(cv.CAP_PROP_POS_FRAMES, start_frame_num)

    err_estimation = error_estimation()  # Create an instance of the error_estimation class
    uav_inst = uav()  # Create an instance of the uav_angles class
    target_parameters_obj = target_parameters()
    tin_colours_obj = tin_colours()
    waypoints_obj = waypoints()
    state_inst = state()

    ################Edit before testing#########################################################
    waypoints_obj.waypoints.append((29.183972, -81.043251))
    ################Edit before testing#########################################################

    uav_inst.state = state_inst.detect_square


    check_for_time.start_time = None
    reached_waypoint = "n"
    while video.isOpened():
        ret, frame = video.read()
        if ret:
            
            # if uav_inst.state == state.descend_concentric or uav_inst.state == state.descend_inner_circle:
            #     frame = cv.resize(frame, (850, 478))
            #     #frame = cv.resize(frame, (9*50, 16*50))
            # elif uav_inst.state == state.detect_tins:
            #     frame = cv.resize(frame, (850, 478))
            # else:
            #     frame = cv.resize(frame, (640, 480))
            frame = cv.resize(frame, (640, 480))
            # frame = cv.resize(frame, (850, 478))
            uav_inst.image_size = (frame.shape[1], frame.shape[0])
        

            match uav_inst.state:
                case state_inst.initial:
                    uav_inst.state = state_inst.fly_to_waypoint

                case state_inst.fly_to_waypoint:
                    if len(waypoints_obj.waypoints) == 0:
                        uav_inst.state = state_inst.done
                        print("No waypoints left")
                    current_waypoint = waypoints_obj.waypoints[-1]
                    
                    navigate_pixhawk.navigate_to_target_coordinates(current_waypoint)

                    ##This check will be done with mavros data
                    if reached_waypoint == "n":
                        reached_waypoint = input("Have you reached the waypoint? (y/n)")
                    if reached_waypoint == "y":
                        uav_inst.state = state_inst.detect_square
                        waypoints_obj.waypoints.remove(current_waypoint)
                        reached_waypoint = "n"
                        check_for_time.start_time = None

                case state_inst.detect_square:              
                    #Check for 1.5s if square is detected more than 50% of the time
                    square_detected_err, is_bimodal, threshold_img = check_for_time(frame=frame, altitude=uav_inst.altitude,duration=1.5,
                                                         ratio_detected=0.5, size_square=target_parameters_obj.size_square, cam_hfov=uav_inst.cam_hfov)
                    debugging_frame = cv.hconcat([frame, cv.cvtColor(threshold_img, cv.COLOR_GRAY2BGR)])
                    if square_detected_err is None:
                        pass #time not over yet
                    elif square_detected_err is False:
                        uav_inst.state = state_inst.fly_to_waypoint
                        reached_waypoint = "n"
                        print("Square not detected. Flying to next waypoint.")
                    else: #Done and detected
                        if is_bimodal:
                            for idx in range(2):
                                current_err = square_detected_err[idx-1] #Reverse order so that closer waypoint is last in list
                                err_estimation.update_errors(current_err[0], current_err[1], uav_inst.altitude, uav_inst.latitude, 
                                                     uav_inst.longitude, uav_inst.heading, [uav_inst.cam_hfov, uav_inst.cam_vfov], uav_inst.image_size)
                                err_estimation.clear_errors() #Clear the errors so that the average is not used (two differnt platforms)
                            waypoints_obj.waypoints.append((err_estimation.target_lat, err_estimation.target_lon))
                            
                        else:
                            err_estimation.update_errors(square_detected_err[0], square_detected_err[1], uav_inst.altitude, uav_inst.latitude, 
                                                     uav_inst.longitude, uav_inst.heading, [uav_inst.cam_hfov, uav_inst.cam_vfov], uav_inst.image_size)
                            waypoints_obj.waypoints.append((err_estimation.target_lat, err_estimation.target_lon))
                        
                        error_in_m = sqrt(err_estimation.x_m**2 + err_estimation.y_m**2)
                        if error_in_m < 1.5:
                            err_estimation.time_last_detection = time.time()
                            uav_inst.state = state_inst.descend_square
                            print("Target detected and now descending")
                        else:
                            print("Target detected but too far away. Flying closer. Distance: " + str(error_in_m) + "m")
                            uav_inst.state = state_inst.fly_to_waypoint


                case state_inst.descend_square:
                    error_xy, altitude, threshold_img = detect_square_main(frame, uav_inst.altitude, target_parameters_obj.size_square, uav_inst.cam_hfov)
                    debugging_frame = cv.hconcat([frame, cv.cvtColor(threshold_img, cv.COLOR_GRAY2BGR)])
                    if error_xy != None: #Target detected
                        err_estimation.update_errors(error_xy[0][0], error_xy[0][1], altitude, uav_inst.latitude, 
                                                     uav_inst.longitude, uav_inst.heading, [uav_inst.cam_hfov, uav_inst.cam_vfov], uav_inst.image_size)
                        navigate_pixhawk.descend_using_error_m(err_estimation.x_m, err_estimation.y_m)
                    elif err_estimation.check_for_timeout(): #No target detected for 3s
                        print("Timeout")
                        uav_inst.state = state_inst.return_to_launch
                    else: #Target not detected but was detected recently
                        navigate_pixhawk.lost_target_ascend()

                    if (altitude is not None and altitude < 6) or skip: #remove skip later
                        # print("Switching to concentric circles")
                        uav_inst.state = state_inst.descend_concentric
                        skip = False
                        # video = cv.VideoCapture("images/concentric_to_single_rot.mp4")
                        # fps = video.get(cv.CAP_PROP_FPS)
                        # start_time_video = 0
                        # start_frame_num = int(start_time_video * fps)
                        # video.set(cv.CAP_PROP_POS_FRAMES, start_frame_num)

                                        

                case state_inst.descend_concentric:
                    alt, error_xy, edges = concentric_circles(frame=frame, altitude=err_estimation.altitude_m_avg, cam_hfov=uav_inst.cam_hfov, circle_parameters_obj=target_parameters_obj) 
                    debugging_frame = cv.hconcat([frame, cv.cvtColor(edges, cv.COLOR_GRAY2BGR)])
                    if alt is not None: #Target detected
                        err_estimation.update_errors(error_xy[0], error_xy[1], alt, uav_inst.latitude, 
                                                     uav_inst.longitude, uav_inst.heading, [uav_inst.cam_hfov, uav_inst.cam_vfov], uav_inst.image_size)
                        navigate_pixhawk.descend_using_error_m(err_estimation.x_m, err_estimation.y_m)
                    elif err_estimation.check_for_timeout(): #No target detected for 3s
                        print("Timeout")
                        uav_inst.state = state_inst.return_to_launch
                    else: #Target not detected but was detected recently
                        navigate_pixhawk.lost_target_ascend()

                    if err_estimation.altitude_m < 2 or skip:
                        skip = False
                        uav_inst.state = state_inst.descend_inner_circle
                        # print("Switching to inner circle")
                                        

                case state_inst.descend_inner_circle:
                    alt,error_xy, edges = small_circle(frame=frame, altitude=err_estimation.altitude_m_avg, cam_hfov=uav_inst.cam_hfov, circle_parameters_obj=target_parameters_obj)
                    debugging_frame = cv.hconcat([frame, cv.cvtColor(edges, cv.COLOR_GRAY2BGR)])
                    if alt is not None:
                        err_estimation.update_errors(error_xy[0], error_xy[1], alt, uav_inst.latitude, 
                                                     uav_inst.longitude, uav_inst.heading, [uav_inst.cam_hfov, uav_inst.cam_vfov], uav_inst.image_size)
                        navigate_pixhawk.descend_using_error_m(err_estimation.x_m, err_estimation.y_m)
                    elif err_estimation.check_for_timeout() and err_estimation.altitude_m > 0.5:
                        print("Timeout")
                        uav_inst.state = state_inst.return_to_launch
                    elif err_estimation.altitude_m < 0.5: #Target not detected but low enough to land
                        navigate_pixhawk.descend_without_error()
                    else:
                        navigate_pixhawk.lost_target_ascend()

                    # if err_estimation.altitude_m < 0.8 or skip:
                    #     skip = False
                    #     # print("Switching to detectig tins. Altitude: " + str(uav_inst.altitude))
                    #     uav_inst.state = state_inst.detect_tins
                    #     tin_detection_for_time.start_time = None

                    #     # video = cv.VideoCapture("images/tin_white_rot.mp4")
                    #     # fps = video.get(cv.CAP_PROP_FPS)
                    #     # start_time_video = 0
                    #     # start_frame_num = int(start_time_video * fps)
                    #     # video.set(cv.CAP_PROP_POS_FRAMES, start_frame_num)
                

                # case state_inst.land:
                #     print("Landing")
                #     pass
                
                # case state_inst.done:
                #     print("Done")
                #     pass

                # case state_inst.return_to_launch:
                #     print("Returning to launch")
                #     pass

                # case state_inst.detect_tins:
                #     uav_inst.state = state_inst.land_tins##tin detection not working yet
                #     continue
                #     err_ground_xy_gbr = tin_detection_for_time(frame=frame, uav_inst=uav_inst, circle_parameters_obj=target_parameters_obj, 
                #                                                tin_colours_obj=tin_colours_obj, altitude=err_estimation.altitude_m)
                                                            
                #     if err_ground_xy_gbr is False:
                #         print("Not enough tins detected")
                #         ###change state here
                #     elif err_ground_xy_gbr is not None:
                #         #print("Errors: ", err_ground_xy_gbr)
                #         err_mode_xy = tins_error_bin_mode(error_ground_gbr_xy=err_ground_xy_gbr, uav_inst=uav_inst, frame_width=frame.shape[1], frame_height=frame.shape[0])
                #         err_img = [[None, None] for _ in range(3)]
                #         for idx in range(3):
                #             err_img[idx] = transform_ground_to_img_xy(err_mode_xy[idx], uav_inst.altitude, [uav_inst.cam_hfov, uav_inst.cam_vfov], [frame.shape[1], frame.shape[0]])
                #         cv.waitKey(0)
                #         ####Fly to the tins
                #         uav_inst.state = state.land_tins
            
            # print("Altitute FC: " + str(uav_inst.altitude) + "Altitude from image: " + str(err_estimation.altitude_m))
            # print("Error ground: " + str((err_estimation.x_m, err_estimation.y_m)))
            # print("Error pixels: " + str((err_estimation.err_px_x, err_estimation.err_px_y)))
            debug_img = display_error_and_text(debugging_frame, (err_estimation.err_px_x, err_estimation.err_px_y), err_estimation.altitude_m_avg,uav_inst)
            cv.imshow("Debugging", debug_img)
            if cv.waitKey(10) & 0xFF == ord('q'):
                break
        else:
            break

      

if __name__ == "__main__":
    main()
