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


import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import *
from mavros_msgs.msg import *
from mavros_msgs.srv import *
from geometry_msgs.msg import *
from std_msgs.msg import *
from tf.transformations import quaternion_from_euler,euler_from_quaternion



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
        #Errors in the image plane range from -1 to 1 in x and -0.75 to 0.75 in y
        self.x_img = 0
        self.y_img = 0
        #Current err in the ground plane depending on x_img and y_img and altitude of UAV
        self.x_m = 0
        self.y_m = 0

        #Filtered errors and altutude in the ground plane
        self.x_m_filt = []
        self.y_m_filt = []
        self.err_distances_filt = []
        self.err_times_filt = []
        self.altitude_m_filt = []

        #Average errors and altitude in the ground plane (after weighted average filter is applied)
        self.x_m_avg = 0
        self.y_m_avg = 0
        self.altitude_m_avg = 0

        self.altitude_m = 0
        self.ascended_by_m = 0 #How much the UAV has ascended by without reference target
        # self.target_lat = 0
        # self.target_lon = 0
        self.list_length = 5 #Number of elements to keep in the list (filter)
        #Errors in pixels for debugging
        self.err_px_x = 0 
        self.err_px_y = 0
        self.p_value = 0.95
        self.time_last_detection = None

    def update_errors(self, x_img, y_img, altitude_m, cam_fov_hv, image_size):
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
        # self.target_lat, self.target_lon = calculate_new_coordinate(uav_lat, uav_lon, [self.x_m_avg, self.y_m_avg], heading)

        #Calculate the error in pixels using the avg value
        curr_err_px = transform_ground_to_img_xy((self.x_m_avg, self.y_m_avg), self.altitude_m_avg, cam_fov_hv, image_size)
        self.err_px_x = curr_err_px[0]
        self.err_px_y = curr_err_px[1]
        self.time_last_detection = rospy.Time.now().to_sec()

        
    
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
        self.err_px_x = 0
        self.err_px_y = 0

    def check_for_timeout(self):
        """Check if the time since last detection of an object is more than 3s"""
        if rospy.Time.now().to_sec() - self.time_last_detection > 60:################################change later
            return True
        return False


class uav(state):
    def __init__(self):
        state.__init__(self)
        self.altitude = None #Altitude of the fc in meters
        self.heading = 0 #Heading of the UAV in radians
        self.latitude = 29.183972 #Current of the UAV
        self.longitude = -81.043251 #Current longitude of the UAV
        self.state = self.initial
        self.cam_hfov = 65*np.pi/180 #Input by user
        self.cam_vfov = 52*np.pi/180 #Input by user
        self.image_size = [640, 480] #Automatically updated by the code

class target_parameters:
    def __init__(self):
        #Change these values if the size of the target changes
        self.diameter_big = 0.84
        self.diameter_small = 0.29
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
        # Initialize ROS node and necessary variables
        rospy.init_node("UAV_landing", anonymous=True)

        self.px4_state = State()  # PX4 state variable (Loiter, Offboard, etc.)
        self.target_pose = PoseStamped()  # Target pose variable
        self.Bridge = CvBridge()  # OpenCV bridge for image conversion
        self.err_estimation = error_estimation()  # Error estimation object
        self.uav_inst = uav()  # UAV instance object
        self.target_parameters_obj = target_parameters()  # Target parameters object
        self.tin_colours_obj = tin_colours()  # Tin colours object
        self.state_inst = state()  # State instance object
        self.waypoint_pose = PoseStamped()  # Waypoint pose variable
        self.final_vel = TwistStamped()  # Final velocity variable
        self.rate = rospy.Rate(60)  # ROS rate for loop frequency
        self.cv_image = None  # OpenCV image variable
        self.target_reached_time = None  # Time when target is reached
        self.current_pose = None  # Current pose variable
        self.angle = (0, 0, 0)  # Angle variable
        self.initialization_time = rospy.Time.now().to_sec()  # Initialization time
        self.takeoff_time = 0  # Takeoff time
        self.offboard_switch_time = 0  # Offboard switch time
        self.loiter_switch_time = 0  # Loiter switch time
        self.waypoint_pushed = False  # Flag to indicate if waypoint is pushed
        self.radius_of_proximity = np.Inf  # Radius of proximity variable
        self.updated_img = False  # Flag to indicate if image is updated
        self.descend_altitude = None  # Descend altitude variable
        self.ascending_start_time = None  # Ascending start time
        self.ascended_by_m = 0  # Ascended by meters

        #####Subscribers#####
        # Subscribe to the downward depth camera image topic
        rospy.Subscriber("/iris_downward_depth_camera/camera/rgb/image_raw/compressed", CompressedImage, callback=self.camera)
        # Subscribe to the local position topic
        rospy.Subscriber("/mavros/local_position/pose", PoseStamped, callback=self.current_position)
        # Subscribe to the MAVROS state topic
        rospy.Subscriber('mavros/state', State, callback=self.monitor_state)
        
        #####Waiting for services#####
        # Wait for the set_mode service to become available
        rospy.wait_for_service('/mavros/set_mode')
        self.set_mode = rospy.ServiceProxy('/mavros/set_mode', SetMode)
        # Wait for the arm service to become available
        rospy.wait_for_service('/mavros/cmd/arming')
        self.arm = rospy.ServiceProxy('/mavros/cmd/arming', CommandBool)
        
        #####Publishers#####
        # Create a publisher for the waypoint pose topic
        self.waypoint_pose_pub = rospy.Publisher("/mavros/setpoint_position/local", PoseStamped, queue_size=50)
        # Create a publisher for the landing camera image topic
        self.img_pub = rospy.Publisher("/iris_downward_depth_camera/camera/landing/rgb/image_raw/compressed", CompressedImage, queue_size=100)
        # Create a publisher for the velocity topic
        self.velocity_pub = rospy.Publisher("mavros/setpoint_velocity/cmd_vel", TwistStamped, queue_size=100)
        
        # Define the waypoints for the UAV to fly to (relative to map frame)
        self.waypoints = [[-10, 40, 10.0],
                  [4, 30.0, 10.0]]


        # Set the frame ID and timestamp for the waypoint pose
        self.waypoint_pose.header.frame_id = 'map'
        self.waypoint_pose.header.stamp = rospy.Time.now()

        # Calculate the orientation based on the angle variable (always point north)
        a = (np.pi/2 - self.angle[2])
        self.theta = np.arctan2(np.sin(a),np.cos(a))
        self.orientation = quaternion_from_euler(self.angle[0],self.angle[1],self.theta)

        # Set the orientation in the waypoint pose
        self.waypoint_pose.pose.orientation.x = self.orientation[0]
        self.waypoint_pose.pose.orientation.y = self.orientation[1]
        self.waypoint_pose.pose.orientation.z = self.orientation[2]
        self.waypoint_pose.pose.orientation.w = self.orientation[3]

        # Set the frame ID for the final velocity
        self.final_vel.header.frame_id = "map"

        # Initialize the start time for the check_for_time function
        check_for_time.start_time = None

        # Run the landing process in a loop until rospy is shutdown
        while not rospy.is_shutdown():
            self.landing()  # Call the landing function
            self.rate.sleep()  # Sleep to maintain the desired loop frequency

    def current_position(self,msg):
        """Callback function for the current position subscriber updating vehicle position"""
        # Update current pose and altitude
        self.current_pose = msg
        self.uav_inst.altitude = self.current_pose.pose.position.z
        # Calculate Euler angles from quaternion
        self.angle = euler_from_quaternion([msg.pose.orientation.x,msg.pose.orientation.y,msg.pose.orientation.z,msg.pose.orientation.w])

    def camera(self,msg):
        """Callback function for the camera subscriber updating the image"""
        # Convert compressed image message to OpenCV format
        self.cv_image = self.Bridge.compressed_imgmsg_to_cv2(msg,"bgr8")
        # Resize the image to a specific size
        self.cv_image = cv.resize(self.cv_image, (640, 480))
        # Update the image size attribute
        self.uav_inst.image_size = (self.cv_image.shape[1], self.cv_image.shape[0])
        # Set the flag to indicate that the image has been updated
        self.updated_img = True

    def monitor_state(self,msg):
        # Update the PX4 state
        self.px4_state = msg

    def guided_descend(self, is_low_altitude=False):
            """Guided descend function to align the UAV with the target and descend to the target position"""
            self.ascended_by_m = 0
            self.ascending_start_time = None

            # Calculate the distance between the estimated position and the target position
            deltaS = np.sqrt(np.square(self.err_estimation.x_m_avg) + np.square(self.err_estimation.y_m_avg))
            # Calculate the angle between the estimated position and the target position
            theta_horizontal = np.arctan2(self.err_estimation.y_m_avg, self.err_estimation.x_m_avg) #Angle to the target in x-y plane

            theta_vertical = np.arctan2(deltaS, self.err_estimation.altitude_m_avg) #Angle to the target in x-z plane

            # Calculate the linear velocity based on the distance
            self.linear_vel = 0.2 * deltaS
            self.linear_vel = 0.5 if self.linear_vel > 0.5 else self.linear_vel #Avoid going too fast to be safe

            # Publish the correction position if the distance is greater than 0.2
            if theta_vertical > 15*np.pi/180: #Not well aligned in z (more than 15 deg off)
                rospy.loginfo("Aligning")
                
                # Set the linear velocity components in the x and y directions
                self.final_vel.header.stamp = rospy.Time.now()
                self.final_vel.twist.linear.x = self.linear_vel * np.cos(theta_horizontal)
                self.final_vel.twist.linear.y = self.linear_vel * np.sin(theta_horizontal)
                self.final_vel.twist.linear.z = 0
                # Publish the velocity message
                self.velocity_pub.publish(self.final_vel)
            else:
                rospy.loginfo("Descending")
                # Set the linear velocity components in the x and y directions
                self.final_vel.header.stamp = rospy.Time.now()
                self.final_vel.twist.linear.x = self.linear_vel * np.cos(theta_horizontal)
                self.final_vel.twist.linear.y = self.linear_vel * np.sin(theta_horizontal)
                if is_low_altitude:
                    self.final_vel.twist.linear.z = -0.2
                self.final_vel.twist.linear.z = -0.5 #Descend with 0.5 m/s
                # Publish the velocity message
            self.velocity_pub.publish(self.final_vel)

    def slight_ascend(self):
        """Function to ascend slightly if the target is lost"""
        rospy.loginfo("Lost target. Ascending slightly.")
        if self.ascending_start_time is None: #Store time at which ascending started
            self.ascending_start_time = rospy.Time.now().to_sec()

        self.final_vel.header.stamp = rospy.Time.now()
        self.final_vel.twist.linear.x = 0
        self.final_vel.twist.linear.y = 0
        self.final_vel.twist.linear.z = 0.1 #Ascend with 0.1 m/s
        self.velocity_pub.publish(self.final_vel)
        #Calculate how much the UAV has ascended by based on the velocity and time elapsed
        self.ascended_by_m = (rospy.Time.now().to_sec() - self.ascending_start_time) * self.final_vel.twist.linear.z
        


    def landing(self):
                
        if self.current_pose != None and self.waypoint_pose != None: #Check if the current pose and waypoint pose are not None
            # Check the current state of the UAV
            if self.uav_inst.state == self.state_inst.initial and (rospy.Time.now().to_sec() - self.initialization_time) > 2:
                # Arm the UAV manually
                rospy.loginfo_once("Vehicle is ready to armed...")
                #save the time of takeoff
                self.takeoff_time = rospy.Time.now().to_sec() - self.takeoff_time
                if self.px4_state.armed and self.takeoff_time > 3:
                    if self.px4_state.mode != 'AUTO.TAKEOFF':
                        #Make sure the vehicle is in takeoff mode
                        mode = self.set_mode(custom_mode='AUTO.TAKEOFF')
                        if mode.mode_sent:
                            rospy.loginfo_once("Vehicle is taking off.")
                    if 2 <= self.current_pose.pose.position.z and self.px4_state.mode == 'AUTO.LOITER':
                         #If the vehicle is at the desired altitude and correct mode, switch to fly to waypoint mode
                         self.uav_inst.state = self.state_inst.fly_to_waypoint

            elif self.uav_inst.state == self.state_inst.fly_to_waypoint:
                if not self.waypoint_pushed:
                    #Push the waypoint to the UAV if not already done so
                    if len(self.waypoints) != 0:  
                        self.waypoint_pose.pose.position.x = self.waypoints[-1][0]
                        self.waypoint_pose.pose.position.y = self.waypoints[-1][1]
                        self.waypoint_pose.pose.position.z = self.waypoints[-1][2]
                        self.waypoints.pop()
                        self.waypoint_pushed = True
                    else:
                        pass
                
                # Publish the waypoint pose
                self.waypoint_pose_pub.publish(self.waypoint_pose)
                
                # Calculate the delta x and delta y between the current pose and the waypoint pose
                delta_x = (self.waypoint_pose.pose.position.x - self.current_pose.pose.position.x)
                delta_y = (self.waypoint_pose.pose.position.y - self.current_pose.pose.position.y)
                
                # Calculate the radius of proximity
                self.radius_of_proximity = np.sqrt(delta_x**2 + delta_y**2)
                
                # Calculate the time since the offboard switch
                self.offboard_switch_time = rospy.Time.now().to_sec() - self.offboard_switch_time
                
                # Check if enough time has passed since the offboard switch
                if self.offboard_switch_time > 2:
                    # Check if the vehicle is not already in OFFBOARD mode
                    if self.px4_state.mode != 'OFFBOARD':
                        # Switch the vehicle to OFFBOARD mode
                        print("Switching to offboard mode")
                        print(self.px4_state.mode)
                        mode = self.set_mode(custom_mode='OFFBOARD')
                        if mode.mode_sent:
                            rospy.loginfo("Waypoint received: x = {}, delta x = {}, y = {}, delta y = {}, z = {}.\n"
                                          .format(self.waypoint_pose.pose.position.x, delta_x, self.waypoint_pose.pose.position.y, 
                                                  delta_y, self.waypoint_pose.pose.position.z))
                            rospy.loginfo("Vehicle is in OFFBOARD mode. Vehicle now flying to the waypoint.")
                
                # Check if the radius of proximity is less than or equal to 0.2
                if self.radius_of_proximity <= 0.2:
                    # Calculate the time since the loiter switch
                    self.loiter_switch_time = rospy.Time.now().to_sec() - self.loiter_switch_time
                
                # Check if enough time has passed since the loiter switch to be sure the UAV is stable
                if self.loiter_switch_time > 5:
                    # Check if the vehicle is not already in AUTO.LOITER mode
                    if self.px4_state.mode != 'AUTO.LOITER':    
                        # Switch the vehicle to AUTO.LOITER mode
                        mode = self.set_mode(custom_mode='AUTO.LOITER')
                        if mode.mode_sent:
                            rospy.loginfo("Vehicle is now in LOITER mode.")
                            self.target_reached_time = rospy.Time.now().to_sec()
                            self.uav_inst.state = self.state_inst.detect_square
            
            if self.updated_img:    #Only run if new image is available         
                if self.uav_inst.state == self.state_inst.detect_square and (rospy.Time.now().to_sec() - self.target_reached_time) > 2:
                    # Check if square is detected within a certain duration
                    square_detected_err, is_bimodal, threshold_img = check_for_time(frame=self.cv_image, altitude=self.uav_inst.altitude, duration=2,
                                                                                    ratio_detected=0.5, size_square=self.target_parameters_obj.size_square, cam_hfov=self.uav_inst.cam_hfov)
                    # Merge the threshold image with the original image for debugging
                    debugging_frame = cv.hconcat([self.cv_image, cv.cvtColor(threshold_img, cv.COLOR_GRAY2BGR)])
                    self.cv_image = display_error_and_text(debugging_frame, (0, 0), self.err_estimation.altitude_m_avg, self.uav_inst)
                    # Add some text overlay to the image
                    

                    if square_detected_err is None:
                        pass  # Time not over yet
                    elif square_detected_err is False:
                        # Square not detected, switch to flying to next waypoint
                        self.uav_inst.state = self.state_inst.fly_to_waypoint
                        self.waypoint_pushed = False
                        self.offboard_switch_time = rospy.Time.now().to_sec()
                        check_for_time.start_time = None
                        rospy.loginfo("Square not detected. Flying to next waypoint.")
                    else:  # Done and detected
                        if is_bimodal:
                            # If multiple squares are detected, update errors for each square
                            for idx in range(2):
                                current_err = square_detected_err[idx-1]  # Reverse order so that closer waypoint is last in list
                                self.err_estimation.update_errors(current_err[0], current_err[1], self.current_pose.pose.position.z, [self.uav_inst.cam_hfov, self.uav_inst.cam_vfov], self.uav_inst.image_size)
                                self.err_estimation.clear_errors()  # Clear the errors so that the average is not used (two different platforms)

                                # Append waypoints with updated error estimation
                                self.waypoints.append([self.current_pose.pose.position.x + self.err_estimation.x_m,
                                                       self.current_pose.pose.position.y + self.err_estimation.y_m,
                                                       10])

                        else:
                            # If only one square is detected, update errors and append waypoint
                            self.err_estimation.update_errors(square_detected_err[0], square_detected_err[1], self.current_pose.pose.position.z, [self.uav_inst.cam_hfov, self.uav_inst.cam_vfov], self.uav_inst.image_size)
                            self.waypoints.append([self.current_pose.pose.position.x + self.err_estimation.x_m,
                                                   self.current_pose.pose.position.y + self.err_estimation.y_m,
                                                   10])

                        error_in_m = sqrt(self.err_estimation.x_m**2 + self.err_estimation.y_m**2)
                        print("current error: ", error_in_m)
                        if error_in_m < 1:
                            # If error is within threshold, switch to descending state
                            self.uav_inst.state = self.state_inst.descend_square
                            self.waypoints.pop()
                            self.err_estimation.altitude_m_avg = self.current_pose.pose.position.z
                            rospy.loginfo("Target detected and now descending")
                        else:
                            # If error is too far, switch to flying to next waypoint
                            rospy.loginfo("Target detected but too far away. Flying closer. Distance: " + str(error_in_m) + "m")
                            self.uav_inst.state = self.state_inst.fly_to_waypoint
                            self.waypoint_pushed = False
                            self.offboard_switch_time = rospy.Time.now().to_sec()

                        check_for_time.start_time = None

                elif self.uav_inst.state == self.state_inst.descend_square:
                    if self.err_estimation.altitude_m_avg is not None: #Make sure altitude is actually set
                        # Detect the square and update errors, altitude and the threshold image
                        current_alt = self.err_estimation.altitude_m_avg + self.ascended_by_m
                        error_xy, self.descend_altitude, threshold_img = detect_square_main(self.cv_image, current_alt, self.target_parameters_obj.size_square, self.uav_inst.cam_hfov)
                        # rospy.loginfo(self.current_pose.pose.position.z)
                        # Merge the threshold image with the original image for debugging
                        debugging_frame = cv.hconcat([self.cv_image, cv.cvtColor(threshold_img, cv.COLOR_GRAY2BGR)])
                        if error_xy != None: #Target detected
                            # Update the errors using the weighted running average filter
                            self.err_estimation.update_errors(error_xy[0][0], error_xy[0][1], self.descend_altitude, [self.uav_inst.cam_hfov, self.uav_inst.cam_vfov], self.uav_inst.image_size)
                            # rospy.loginfo("x_m_avg: {}, y_m_avg: {}.\n".format(self.err_estimation.x_m_avg,self.err_estimation.y_m_avg))
                            # if 0.1<self.err_estimation.x_m_avg or -0.1>self.err_estimation.x_m_avg:
                            #     rospy.loginfo("Aligning in x.")
                            #     self.waypoint_pose.pose.position.x = self.current_pose.pose.position.x + self.err_estimation.x_m_avg
                            #     self.waypoint_pose.pose.position.y = self.current_pose.pose.position.y
                            #     self.waypoint_pose.pose.position.z = self.current_pose.pose.position.z
                            # elif 0.1<self.err_estimation.y_m_avg or -0.1>self.err_estimation.y_m_avg:
                            #     rospy.loginfo("Aligning in y.")
                            #     self.waypoint_pose.pose.position.x = self.current_pose.pose.position.x
                            #     self.waypoint_pose.pose.position.y = self.current_pose.pose.position.y + self.err_estimation.y_m_avg
                            #     self.waypoint_pose.pose.position.z = self.current_pose.pose.position.z
                            # else:
                            #     rospy.loginfo("Descending.")
                            #     self.waypoint_pose.pose.position.x = self.current_pose.pose.position.x
                            #     self.waypoint_pose.pose.position.y = self.current_pose.pose.position.y
                            #     self.waypoint_pose.pose.position.z = self.current_pose.pose.position.z - 0.3
                            
                            self.guided_descend()
                           
                            # Check if the vehicle is in OFFBOARD mode and switch to OFFBOARD mode if necessary
                            if self.px4_state.mode != 'OFFBOARD':
                                mode = self.set_mode(custom_mode='OFFBOARD')

                            if(self.descend_altitude is not None and self.descend_altitude < 4):
                                self.uav_inst.state = self.state_inst.descend_concentric
                                rospy.loginfo("Descending using concentric circles")
                                    

                        elif self.err_estimation.check_for_timeout(): #No target detected for 3s
                            rospy.loginfo_once("Timeout")
                            # self.uav_inst.state = self.state_inst.return_to_launch
                            if self.px4_state.mode != 'AUTO.LOITER':    
                                mode = self.set_mode(custom_mode='AUTO.LOITER')
                                if mode.mode_sent:
                                    rospy.loginfo_once("Vehicle is now in LOITER mode.")
                        else: #Target not detected but was detected recently
                            # Set the linear velocity components in the x, y, and z directions
                            self.slight_ascend()
                            """ if self.px4_state.mode != 'AUTO.LAND':    
                                mode = self.set_mode(custom_mode='AUTO.LAND')
                                if mode.mode_sent:
                                    rospy.loginfo_once("Target out of FOV. Vehicle will just land at the current position.") """

                        self.cv_image = display_error_and_text(debugging_frame, (self.err_estimation.err_px_x, self.err_estimation.err_px_y), self.err_estimation.altitude_m_avg,self.uav_inst)
                        
                    

                elif self.uav_inst.state == self.state_inst.descend_concentric:
                    current_alt = self.err_estimation.altitude_m_avg + self.ascended_by_m
                    alt, error_xy, edges = concentric_circles(frame=self.cv_image, altitude=current_alt, cam_hfov=self.uav_inst.cam_hfov, circle_parameters_obj=self.target_parameters_obj) 
                    debugging_frame = cv.hconcat([self.cv_image, cv.cvtColor(edges, cv.COLOR_GRAY2BGR)])
                    if alt is not None: #Target detected
                        self.err_estimation.update_errors(error_xy[0], error_xy[1], alt, [self.uav_inst.cam_hfov, self.uav_inst.cam_vfov], self.uav_inst.image_size)
                        self.guided_descend()
                    elif self.err_estimation.check_for_timeout(): #No target detected for 3s
                        print("Timeout")
                        pass
                    else: #Target not detected but was detected recently
                        self.slight_ascend()

                    if self.err_estimation.altitude_m_avg < 2:
                        self.uav_inst.state = self.state_inst.descend_inner_circle
                    self.cv_image = display_error_and_text(debugging_frame, (self.err_estimation.err_px_x, self.err_estimation.err_px_y), self.err_estimation.altitude_m_avg,self.uav_inst)
                    
                elif self.uav_inst.state == self.state_inst.descend_inner_circle:
                    current_alt = self.err_estimation.altitude_m_avg + self.ascended_by_m
                    alt,error_xy, edges = small_circle(frame=self.cv_image, altitude=current_alt, cam_hfov=self.uav_inst.cam_hfov, circle_parameters_obj=self.target_parameters_obj)
                    debugging_frame = cv.hconcat([self.cv_image, cv.cvtColor(edges, cv.COLOR_GRAY2BGR)])
                    if alt is not None:
                        self.err_estimation.update_errors(error_xy[0], error_xy[1], alt, [self.uav_inst.cam_hfov, self.uav_inst.cam_vfov], self.uav_inst.image_size)
                        self.guided_descend(is_low_altitude=True) #Guided descend but slower
                    elif self.err_estimation.check_for_timeout() and self.err_estimation.altitude_m > 0.5:
                        print("Timeout")
                        self.uav_inst.state = self.state_inst.return_to_launch
                    elif self.err_estimation.altitude_m < 0.7: #Target not detected but low enough to land
                        self.uav_inst.state = self.state_inst.land
                    else:
                        self.slight_ascend()
                    self.cv_image = display_error_and_text(debugging_frame, (self.err_estimation.err_px_x, self.err_estimation.err_px_y), self.err_estimation.altitude_m_avg,self.uav_inst)

                elif self.uav_inst.state == self.state_inst.land:
                    if self.px4_state.mode != 'AUTO.LAND':    
                        mode = self.set_mode(custom_mode='AUTO.LAND')
                        if mode.mode_sent:
                            rospy.loginfo_once("Vehicle is now landing.")
                
                
                #Set updated_to false to wait for new img
                self.updated_img = False 

                img_to_msg = self.Bridge.cv2_to_compressed_imgmsg(self.cv_image,'jpg')
                self.img_pub.publish(img_to_msg)
                # cv.imshow("Display",self.cv_image)
                # cv.waitKey(0)




                    



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
