#!/usr/bin/env python3

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

import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import *
from mavros_msgs.msg import *
from mavros_msgs.srv import *
from geometry_msgs.msg import *
from std_msgs.msg import *
from tf.transformations import quaternion_from_euler,euler_from_quaternion

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
        self.altitude = 4 #Altitude of the fc in meters
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

# class waypoints:
#     def __init__(self):
#         self.waypoints = []


class tin_colours:
    def __init__(self):
        #Change this if the colour of the tins changes (hue value from 0 to 180)
        self.green_hue = 95
        self.blue_hue = 105
        self.red_hue = 170

class main():
    def __init__(self):
        rospy.init_node("UAV_landing",anonymous=True)

        self.px4_state = State()
        self.target_pose = PoseStamped()
        self.Bridge = CvBridge()
        self.err_estimation = error_estimation()
        self.uav_inst = uav()
        self.target_parameters_obj = target_parameters()
        self.tin_colours_obj = tin_colours()
        self.state_inst = state()
        self.waypoint_pose = PoseStamped()
        self.rate = rospy.Rate(60)
        self.cv_image = None
        self.target_reached_time = None
        self.current_pose = None
        self.angle = (0,0,0)
        self.initialization_time = rospy.Time.now().to_sec()
        self.offboard_switch_time = None
        self.waypoint_pushed = False
        self.radius_of_proximity = np.Inf

        rospy.Subscriber("/iris_downward_depth_camera/camera/rgb/image_raw/compressed",CompressedImage,callback=self.camera)
        rospy.Subscriber("/mavros/local_position/pose",PoseStamped,callback=self.current_position)
        rospy.Subscriber('mavros/state', State, callback=self.monitor_state)
        rospy.wait_for_service('/mavros/set_mode')
        self.set_mode = rospy.ServiceProxy('/mavros/set_mode', SetMode)
        rospy.wait_for_service('/mavros/cmd/arming')
        self.arm = rospy.ServiceProxy('/mavros/cmd/arming',CommandBool)
        self.waypoint_pose_pub = rospy.Publisher("/mavros/setpoint_position/local",PoseStamped,queue_size=50)
        self.img_pub = rospy.Publisher("/iris_downward_depth_camera/camera/landing/rgb/image_raw/compressed",CompressedImage,queue_size=100)

        self.waypoints = [[60.0, 120, 4.0],
                          [48.0, 116.0, 4.0]]                 


        self.waypoint_pose.header.frame_id = 'map'
        self.waypoint_pose.header.stamp = rospy.Time.now()
        a = (np.pi/2 - self.angle[2])
        self.theta = np.arctan2(np.sin(a),np.cos(a))
        self.orientation = quaternion_from_euler(self.angle[0],self.angle[1],self.theta)
        self.waypoint_pose.pose.orientation.x = self.orientation[0]
        self.waypoint_pose.pose.orientation.y = self.orientation[1]
        self.waypoint_pose.pose.orientation.z = self.orientation[2]
        self.waypoint_pose.pose.orientation.w = self.orientation[3]

        check_for_time.start_time = None

        while not rospy.is_shutdown():
            self.landing()
            self.rate.sleep()
            # if self.px4_state.mode == 'AUTO.RTL':
            #     break

    def current_position(self,msg):
        self.current_pose = msg
        self.angle = euler_from_quaternion([msg.pose.orientation.x,msg.pose.orientation.y,msg.pose.orientation.z,msg.pose.orientation.w])

    def camera(self,msg):
        self.cv_image = self.Bridge.compressed_imgmsg_to_cv2(msg,"bgr8")

    def monitor_state(self,msg):
        self.px4_state = msg

    def landing(self):
                
        if self.current_pose != None and self.waypoint_pose != None:

            if self.uav_inst.state == self.state_inst.initial and (rospy.Time.now().to_sec() - self.initialization_time) > 2:
                rospy.loginfo_once("Vehicle is ready to armed...")
                if self.px4_state.armed and (rospy.Time.now().to_sec() - self.initialization_time) > 5:
                    if self.px4_state.mode != 'AUTO.TAKEOFF':
                        mode = self.set_mode(custom_mode='AUTO.TAKEOFF')
                        if mode.mode_sent:
                            rospy.loginfo_once("Vehicle is taking off.")
                    if 2 <= self.current_pose.pose.position.z and self.px4_state.mode == 'AUTO.LOITER':
                         self.uav_inst.state = self.state_inst.fly_to_waypoint
                         self.offboard_switch_time = rospy.Time.now().to_sec()

            if self.uav_inst.state == self.state_inst.fly_to_waypoint:
                if not self.waypoint_pushed:
                    if len(self.waypoints) != 0:  
                        self.waypoint_pose.pose.position.x = self.waypoints[-1][0]
                        self.waypoint_pose.pose.position.y = self.waypoints[-1][1]
                        self.waypoint_pose.pose.position.z = self.waypoints[-1][2]
                        self.waypoints.pop()
                        self.waypoint_pushed = True
                    else:
                        # rospy.loginfo_once("No waypoints. Vehicle will now return to home.")
                        # mode = self.set_mode(custom_mode='AUTO.RTL')
                        # if mode.mode_sent:
                        #     rospy.loginfo_once("Vehicle returning home.")
                        #     rospy.signal_shutdown("END")
                        pass
                
                if self.waypoint_pushed:
                    self.waypoint_pose_pub.publish(self.waypoint_pose)
                    delta_x = (self.waypoint_pose.pose.position.x - self.current_pose.pose.position.x)
                    delta_y = (self.waypoint_pose.pose.position.y - self.current_pose.pose.position.y)
                    self.radius_of_proximity = np.sqrt(delta_x**2 + delta_y**2)

                    if (rospy.Time.now().to_sec() - self.offboard_switch_time) > 5:
                        mode = self.set_mode(custom_mode='OFFBOARD')
                        if mode.mode_sent:
                            rospy.loginfo_once("Waypoint received: x = {}, y = {}, z = {}.\n".format(self.waypoint_pose.pose.position.x,self.waypoint_pose.pose.position.y,self.waypoint_pose.pose.position.z))
                            rospy.loginfo_once("Vehicle is in OFFBOARD mode. Vehicle now flying to the waypoint.")

            if 0 <= self.radius_of_proximity <= 1 and self.uav_inst.state != self.state_inst.detect_square:
                self.uav_inst.state = self.state_inst.detect_square
                mode = self.set_mode(custom_mode='AUTO.LOITER')
                if mode.mode_sent:
                    rospy.loginfo_once("Vehicle is now in LOITER mode.")
                    self.target_reached_time = rospy.Time.now().to_sec()
                
            if self.uav_inst.state == self.state_inst.detect_square and (rospy.Time.now().to_sec() - self.target_reached_time) > 5:
                square_detected_err, is_bimodal, threshold_img = check_for_time(frame=self.cv_image, altitude=self.uav_inst.altitude,duration=1.5,
                                                    ratio_detected=0.5, size_square=self.target_parameters_obj.size_square, cam_hfov=self.uav_inst.cam_hfov)
                debugging_frame = cv.hconcat([self.cv_image, cv.cvtColor(threshold_img, cv.COLOR_GRAY2BGR)])
                if square_detected_err is None:
                    pass #time not over yet
                elif square_detected_err is False:
                    self.uav_inst.state = self.state_inst.fly_to_waypoint
                    self.waypoint_pushed = False
                    self.offboard_switch_time = rospy.Time.now().to_sec()                  
                    rospy.loginfo("Square not detected. Flying to next waypoint.")
                else: #Done and detected
                    rospy.loginfo("Square plate detected.")

                self.cv_image = display_error_and_text(debugging_frame,self.uav_inst)

            cv.imshow("Display",self.cv_image)
            cv.waitKey(1)

                    # if is_bimodal:#two targets in image
                    #     for idx in range(2):
                    #         current_err = square_detected_err[idx-1] #Reverse order so that closer waypoint is last in list
                    #         self.err_estimation.update_errors(current_err[0], current_err[1], self.uav_inst.altitude, self.uav_inst.latitude, 
                    #                                 self.uav_inst.longitude, self.uav_inst.heading, [self.uav_inst.cam_hfov, self.uav_inst.cam_vfov], self.uav_inst.image_size)
                    #         self.err_estimation.clear_errors() #Clear the errors so that the average is not used (two differnt platforms)
                    #     self.waypoints_obj.waypoints.append((self.err_estimation.target_lat, self.err_estimation.target_lon))
                        
                    # else:
                    #     self.err_estimation.update_errors(square_detected_err[0], square_detected_err[1], self.uav_inst.altitude, self.uav_inst.latitude, 
                    #                                 self.uav_inst.longitude, self.uav_inst.heading, [self.uav_inst.cam_hfov, self.uav_inst.cam_vfov], self.uav_inst.image_size)
                    #     self.waypoints_obj.waypoints.append((self.err_estimation.target_lat, self.err_estimation.target_lon))
                    
                    # error_in_m = sqrt(self.err_estimation.x_m**2 + self.err_estimation.y_m**2)
                    # if error_in_m < 1.5:
                    #     self.err_estimation.time_last_detection = time.time()
                    #     self.uav_inst.state = self.state_inst.descend_square
                    #     print("Target detected and now descending")
                    # else:
                    #     print("Target detected but too far away. Flying closer. Distance: " + str(error_in_m) + "m")
                    #     self.uav_inst.state = self.state_inst.fly_to_waypoint



                    



    #     check_for_time.start_time = None

    # def camera(self,msg):
    #     self.cv_image = self.Bridge.compressed_imgmsg_to_cv2(msg,'rgb')
    #     frame = cv.resize(self.cv_image, (640, 480))
    #     # frame = cv.resize(frame, (850, 478))
    #     self.uav_inst.image_size = (frame.shape[1], frame.shape[0])
    

    #     if self.uav_inst.state == self.state_inst.initial:
    #         self.uav_inst.state = self.state_inst.fly_to_waypoint

    #     elif self.uav_inst.state == self.state_inst.fly_to_waypoint:
    #         if len(self.waypoints_obj.waypoints) == 0:
    #             self.uav_inst.state = self.state_inst.done
    #             print("No waypoints left")
    #         self.current_waypoint.current_seq = []
    #         self.current_waypoint.waypoints = self.waypoints_obj.waypoints[-1]
    #         self.waypoint_pub.publish(self.current_waypoint)
            
    #         # navigate_pixhawk.navigate_to_target_coordinates(current_waypoint)

    #         ##This check will be done with mavros data
    #         if self.waypoint_reached == []:
    #             self.uav_inst.state = self.state_inst.detect_square
    #             self.waypoints_obj.waypoints.remove(self.current_waypoint)
    #             check_for_time.start_time = None

    #     elif self.uav_inst.state == self.state_inst.detect_square:              
    #         #Check for 1.5s if square is detected more than 50% of the time
    #         square_detected_err, is_bimodal, threshold_img = check_for_time(frame=frame, altitude=self.uav_inst.altitude,duration=1.5,
    #                                                 ratio_detected=0.5, size_square=self.target_parameters_obj.size_square, cam_hfov=self.uav_inst.cam_hfov)
    #         debugging_frame = cv.hconcat([frame, cv.cvtColor(threshold_img, cv.COLOR_GRAY2BGR)])
    #         if square_detected_err is None:
    #             pass #time not over yet
    #         elif square_detected_err is False:
    #             self.uav_inst.state = self.state_inst.fly_to_waypoint
    #             print("Square not detected. Flying to next waypoint.")
    #         else: #Done and detected
    #             if is_bimodal:#two targets in image
    #                 for idx in range(2):
    #                     current_err = square_detected_err[idx-1] #Reverse order so that closer waypoint is last in list
    #                     self.err_estimation.update_errors(current_err[0], current_err[1], self.uav_inst.altitude, self.uav_inst.latitude, 
    #                                             self.uav_inst.longitude, self.uav_inst.heading, [self.uav_inst.cam_hfov, self.uav_inst.cam_vfov], self.uav_inst.image_size)
    #                     self.err_estimation.clear_errors() #Clear the errors so that the average is not used (two differnt platforms)
    #                 self.waypoints_obj.waypoints.append((self.err_estimation.target_lat, self.err_estimation.target_lon))
                    
    #             else:
    #                 self.err_estimation.update_errors(square_detected_err[0], square_detected_err[1], self.uav_inst.altitude, self.uav_inst.latitude, 
    #                                             self.uav_inst.longitude, self.uav_inst.heading, [self.uav_inst.cam_hfov, self.uav_inst.cam_vfov], self.uav_inst.image_size)
    #                 self.waypoints_obj.waypoints.append((self.err_estimation.target_lat, self.err_estimation.target_lon))
                
    #             error_in_m = sqrt(self.err_estimation.x_m**2 + self.err_estimation.y_m**2)
    #             if error_in_m < 1.5:
    #                 self.err_estimation.time_last_detection = time.time()
    #                 self.uav_inst.state = self.state_inst.descend_square
    #                 print("Target detected and now descending")
    #             else:
    #                 print("Target detected but too far away. Flying closer. Distance: " + str(error_in_m) + "m")
    #                 self.uav_inst.state = self.state_inst.fly_to_waypoint


    #     elif self.uav_inst.state == self.state_inst.descend_square:
    #         error_xy, altitude, threshold_img = detect_square_main(frame, self.uav_inst.altitude, self.target_parameters_obj.size_square, self.uav_inst.cam_hfov)
    #         debugging_frame = cv.hconcat([frame, cv.cvtColor(threshold_img, cv.COLOR_GRAY2BGR)])
    #         if error_xy != None: #Target detected
    #             self.err_estimation.update_errors(error_xy[0][0], error_xy[0][1], altitude, self.uav_inst.latitude, 
    #                                             self.uav_inst.longitude, self.uav_inst.heading, [self.uav_inst.cam_hfov, self.uav_inst.cam_vfov], self.uav_inst.image_size)
    #             navigate_pixhawk.descend_using_error_m(self.err_estimation.x_m, self.err_estimation.y_m)
    #         elif self.err_estimation.check_for_timeout(): #No target detected for 3s
    #             print("Timeout")
    #             self.uav_inst.state = self.state_inst.return_to_launch
    #         else: #Target not detected but was detected recently
    #             navigate_pixhawk.lost_target_ascend()

    #         if (altitude is not None and altitude < 6): #remove skip later
    #             self.uav_inst.state = self.state_inst.descend_concentric

                                

    #     elif self.uav_inst.state == self.state_inst.descend_concentric:
    #         alt, error_xy, edges = concentric_circles(frame=frame, altitude=self.err_estimation.altitude_m_avg, cam_hfov=self.uav_inst.cam_hfov, circle_parameters_obj=self.target_parameters_obj) 
    #         debugging_frame = cv.hconcat([frame, cv.cvtColor(edges, cv.COLOR_GRAY2BGR)])
    #         if alt is not None: #Target detected
    #             self.err_estimation.update_errors(error_xy[0], error_xy[1], alt, self.uav_inst.latitude, 
    #                                             self.uav_inst.longitude, self.uav_inst.heading, [self.uav_inst.cam_hfov, self.uav_inst.cam_vfov], self.uav_inst.image_size)
    #             navigate_pixhawk.descend_using_error_m(self.err_estimation.x_m, self.err_estimation.y_m)
    #         elif self.err_estimation.check_for_timeout(): #No target detected for 3s
    #             print("Timeout")
    #             self.uav_inst.state = self.state_inst.return_to_launch
    #         else: #Target not detected but was detected recently
    #             navigate_pixhawk.lost_target_ascend()

    #         if self.err_estimation.altitude_m < 2:
    #             skip = False
    #             self.uav_inst.state = self.state_inst.descend_inner_circle
                                

    #     elif self.uav_inst.state == self.state_inst.descend_inner_circle:
    #         alt,error_xy, edges = small_circle(frame=frame, altitude=self.err_estimation.altitude_m_avg, cam_hfov=self.uav_inst.cam_hfov, circle_parameters_obj=self.target_parameters_obj)
    #         debugging_frame = cv.hconcat([frame, cv.cvtColor(edges, cv.COLOR_GRAY2BGR)])
    #         if alt is not None:
    #             self.err_estimation.update_errors(error_xy[0], error_xy[1], alt, self.uav_inst.latitude, 
    #                                             self.uav_inst.longitude, self.uav_inst.heading, [self.uav_inst.cam_hfov, self.uav_inst.cam_vfov], self.uav_inst.image_size)
    #             navigate_pixhawk.descend_using_error_m(self.err_estimation.x_m, self.err_estimation.y_m)
    #         elif self.err_estimation.check_for_timeout() and self.err_estimation.altitude_m > 0.5:
    #             print("Timeout")
    #             self.uav_inst.state = self.state_inst.return_to_launch
    #         elif self.err_estimation.altitude_m < 0.5: #Target not detected but low enough to land
    #             navigate_pixhawk.descend_without_error()
    #         else:
    #             navigate_pixhawk.lost_target_ascend()

                # debug_img = display_error_and_text(debugging_frame,self.uav_inst)
                # img_to_msg = self.Bridge.cv2_to_compressed_imgmsg(debug_img,'jpg')
                # self.img_pub.publish(img_to_msg)

    # def waypoint_reached(self,msg):
    #     self.reached_waypoint = msg
    #     return self.reached_waypoint.wp_sq
      

if __name__ == "__main__":
    main()
