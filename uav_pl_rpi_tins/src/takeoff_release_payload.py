#!/usr/bin/env python

import rospy
from mavros_msgs.srv import CommandBool, SetMode, CommandTOL
from mavros_msgs.msg import State
from geometry_msgs.msg import PoseStamped
import time

class UAVExtendedMission:
    def __init__(self):
        rospy.init_node('uav_extended_mission_node', anonymous=True)

        # Subscribers, Publishers, and Service Proxies
        self.state_sub = rospy.Subscriber("/mavros/state", State, self.state_callback)
        self.local_pos_pub = rospy.Publisher("/mavros/setpoint_position/local", PoseStamped, queue_size=10)

        self.arming_srv = rospy.ServiceProxy("/mavros/cmd/arming", CommandBool)
        self.set_mode_srv = rospy.ServiceProxy("/mavros/set_mode", SetMode)
        self.takeoff_srv = rospy.ServiceProxy("/mavros/cmd/takeoff", CommandTOL)

        self.current_state = State()
        self.rate = rospy.Rate(20)  # 20 Hz control loop

    def state_callback(self, state):
        self.current_state = state

    def arm_and_takeoff(self, altitude=2.0):
        rospy.loginfo("Taking off from the platform to 2 meters altitude")
        self.set_mode("OFFBOARD")
        self.arming_srv(True)  # Arm the UAV

        takeoff_command = CommandTOL(min_pitch=0, yaw=0, latitude=0, longitude=0, altitude=altitude)
        self.takeoff_srv(takeoff_command)

    def loiter(self, altitude, duration=10):
        rospy.loginfo(f"Loitering at {altitude} meters for {duration} seconds")
        pose = PoseStamped()
        pose.pose.position.z = altitude

        start_time = rospy.Time.now().to_sec()
        while rospy.Time.now().to_sec() - start_time < duration:
            self.local_pos_pub.publish(pose)
            self.rate.sleep()

    def jostle(self, roll_amplitude=0.1, duration=1.0):
        rospy.loginfo("Jostling on the roll axis")
        roll_left = True  # Flag to alternate roll direction

        # Perform left-right roll maneuver twice
        for _ in range(2):
            start_time = rospy.Time.now().to_sec()
            while rospy.Time.now().to_sec() - start_time < duration:
                if roll_left:
                    rospy.loginfo("Rolling left")
                    # Placeholder for left roll command to Pixhawk (e.g., setting roll angle to -roll_amplitude)
                else:
                    rospy.loginfo("Rolling right")
                    # Placeholder for right roll command to Pixhawk (e.g., setting roll angle to roll_amplitude)
                roll_left = not roll_left  # Switch direction

    def maintain_loiter(self, altitude):
        rospy.loginfo("Maintaining loiter position indefinitely until RC control is taken")
        pose = PoseStamped()
        pose.pose.position.z = altitude

        # Continuously publish the loiter position until RC control is detected
        while not rospy.is_shutdown():
            self.local_pos_pub.publish(pose)
            self.rate.sleep()

    def run_mission(self):
        # Assuming landing has already occurred

        # Takeoff from the landing platform to 2 meters
        self.arm_and_takeoff(altitude=2.0)
        rospy.sleep(5)  # Ensure stability after takeoff

        # Loiter for 10 seconds at 2 meters altitude
        self.loiter(altitude=2.0, duration=10)

        # Jostle on roll axis
        self.jostle()

        # Maintain loiter indefinitely until RC control is taken
        self.maintain_loiter(altitude=2.0)

if __name__ == "__main__":
    try:
        uav_mission = UAVExtendedMission()
        uav_mission.run_mission()
    except rospy.ROSInterruptException:
        pass
