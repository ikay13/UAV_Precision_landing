#!/usr/bin/env python3

import rospy
import argparse
from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import State
from std_msgs.msg import Int8

class Position():
    """Publishes the target position."""
    def __init__(self):
        rospy.init_node("Target_pub_node",anonymous=True)
        rospy.Subscriber('/mavros/state',State,callback=self.monitor_state)
        rospy.Subscriber("/mavros/uav/state",Int8,callback=self.uav_state)
        rospy.Subscriber("/mavros/uav/next/position",PoseStamped,callback=self.next_target_position)
        self.tar_pose_pub = rospy.Publisher("/mavros/setpoint_position/local",PoseStamped,queue_size=50)
        # self.parser = argparse.ArgumentParser(description='Target Position.')
        # self.parser.add_argument('x', type=float, help='Local x distance in meters.')
        # self.parser.add_argument('y', type=float, help='Local y distance in meters.')
        # self.parser.add_argument('z', type=float, help='Local z distance in meters.')
        # self.args = self.parser.parse_args()
        self.target_position = PoseStamped()
        self.px4_state = State()
        self.uav_state_data = Int8()
        self.rate = rospy.Rate(60)
        self.count = 0

        while not rospy.is_shutdown():
            self.tar_pose_pub.publish(self.target_position)
            rospy.loginfo(self.target_position)
                  
            if self.uav_state_data == 2 and self.px4_state.mode == 'AUTO.LOITER':
                rospy.loginfo_once('Target reached. Stopping target publisher.')
                break

            self.rate.sleep()

    def monitor_state(self,msg):
        self.px4_state = msg

    def uav_state(self,msg):
        self.uav_state_data = msg.data

    def next_target_position(self,msg):
        self.target_position = msg
        
        self.count = self.count+1

    # def target_pose(self):
    #     self.target_position.header.seq = self.count
    #     self.target_position.header.frame_id = 'map'
    #     self.target_position.header.stamp = rospy.Time.now()
    #     self.target_position.pose.position.x = self.args.x
    #     self.target_position.pose.position.y = self.args.y
    #     self.target_position.pose.position.z = self.args.z

    #     self.tar_pose_pub.publish(self.target_position)
    #     rospy.loginfo(self.target_position)
    #     self.count = self.count+1


if __name__ == '__main__':
    try:
        Position()
    except rospy.ROSInterruptException:
        pass
