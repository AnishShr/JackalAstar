# !/usr/bin/env python

import rospy
import tf
import tf2_ros
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PointStamped, PoseStamped, Twist
from nav_msgs.msg import Odometry
import tf2_geometry_msgs

import numpy as np

DISTANCE_TO_GOAL_THRESHOLD = 0.5

class LaserRefiner:
    def __init__(self):
        rospy.init_node('laser_refiner_node')

        self.input_topic = '/scan'
        self.output_topic = '/front/scan'            

        # Publishers
        self.scan_pub = rospy.Publisher(self.output_topic, 
                                        LaserScan, 
                                        queue_size=1)       

        # Subscribers
        self.scan_sub = rospy.Subscriber(self.input_topic, LaserScan, self.scan_callback, queue_size=1)


    def scan_callback(self, msg):        
        
        ranges = []
        for laser_range in msg.ranges:
            if np.isinf(laser_range):
                if laser_range == float("-inf"):
                    ranges.append(msg.range_min)
                else:
                    ranges.append(msg.range_max)
            elif np.isnan(laser_range):
                ranges.append(msg.range_max)
            else:
                ranges.append(laser_range)

        # Create a new LaserScan message to store filtered data
        refined_scan_msg = LaserScan()
        refined_scan_msg.header = msg.header        
        refined_scan_msg.angle_min = msg.angle_min
        refined_scan_msg.angle_max = msg.angle_max
        refined_scan_msg.angle_increment = msg.angle_increment
        refined_scan_msg.time_increment = msg.time_increment
        refined_scan_msg.scan_time = msg.scan_time
        refined_scan_msg.range_min = msg.range_min
        refined_scan_msg.range_max = msg.range_max
        refined_scan_msg.ranges = ranges

        self.scan_pub.publish(refined_scan_msg)

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    node = LaserRefiner()
    node.run()
