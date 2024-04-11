#!/usr/bin/env python3

import rospy
from mavros_msgs.msg import *

class Mode():
    """Check the state of px4."""
    def __init__(self):
        rospy.init_node("px4_mode",anonymous=True)
        rospy.Subscriber("/mavros/state",State,callback=self.px4_mode)

        rospy.spin()

    def px4_mode(self,msg):
        self.px4_state = msg
        rospy.loginfo("Arming status: {}, Mode: {}.\n".format(self.px4_state.armed,self.px4_state.mode))

if __name__ == '__main__':
    try:
        Mode()
    except rospy.ROSInterruptException:
        pass