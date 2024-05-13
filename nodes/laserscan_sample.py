# !/usr/bin/env python

import rospy
import tf
import tf2_ros
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PointStamped, PoseStamped, Twist
from nav_msgs.msg import Odometry
import tf2_geometry_msgs

import numpy as np

class LaserDownsampler:
    def __init__(self):
        rospy.init_node('laser_downsampler_node')

        self.input_topic = '/front/scan'
        self.output_topic = '/scan/filtered'       

        # Publishers
        self.scan_pub = rospy.Publisher(self.output_topic, 
                                        LaserScan, 
                                        queue_size=1)

        # Subscribers
        self.scan_sub = rospy.Subscriber(self.input_topic, LaserScan, self.scan_callback, queue_size=1)


    def scan_callback(self, msg):
        
        filtered_ranges = []
        current_angle = msg.angle_min
        angle_increment = np.deg2rad(10)

        while current_angle <= msg.angle_max:
            index = int((current_angle - msg.angle_min)/msg.angle_increment)
            range_scan = msg.ranges[index]

            filtered_ranges.append(range_scan)
            current_angle += angle_increment
        
        filtered_scan_msg = LaserScan()
        filtered_scan_msg.header = msg.header        
        filtered_scan_msg.angle_min = msg.angle_min
        filtered_scan_msg.angle_max = msg.angle_max
        filtered_scan_msg.angle_increment = angle_increment
        filtered_scan_msg.time_increment = msg.time_increment
        filtered_scan_msg.scan_time = msg.scan_time
        filtered_scan_msg.range_min = msg.range_min
        filtered_scan_msg.range_max = msg.range_max
        filtered_scan_msg.ranges = filtered_ranges

        self.scan_pub.publish(filtered_scan_msg)
    

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    node = LaserDownsampler()
    node.run()
