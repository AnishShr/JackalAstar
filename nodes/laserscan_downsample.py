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

class LaserDownsampler:
    def __init__(self):
        rospy.init_node('laser_downsampler_node')

        self.input_topic = '/front/scan'
        self.output_topic = '/scan/filtered'
        self.odom_topic = '/odometry/filtered'
        self.goal_topic = '/move_base_simple/goal'

        self.angle_min = -np.deg2rad(75.0)
        self.angle_max = np.deg2rad(75.0)
        self.angle_increment = np.deg2rad(10.0)

        # Variables
        self.goal_location = None
        self.current_location = None        

        # TF buffer
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # TF frames
        self.source_frame = 'odom'
        self.target_frame = 'front_laser'

        # Publishers
        self.scan_pub = rospy.Publisher(self.output_topic, 
                                        LaserScan, 
                                        queue_size=1)
        
        self.vel_pub = rospy.Publisher('/cmd_vel',
                                        Twist,
                                        queue_size=1)


        # Subscribers
        self.scan_sub = rospy.Subscriber(self.input_topic, LaserScan, self.scan_callback, queue_size=1)
        self.odom_sub = rospy.Subscriber(self.odom_topic, Odometry, self.odom_callback, queue_size=1)
        self.goal_sub = rospy.Subscriber(self.goal_topic, PoseStamped, self.goal_callback, queue_size=1)

    
    def goal_callback(self, msg):
        self.goal_location = np.array([msg.pose.position.x, msg.pose.position.y])


    def odom_callback(self, msg):
        self.current_location = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y])


    def scan_callback(self, scan_msg):
        
        sampled_ranges = []
        sampled_angles = []
        current_angle = scan_msg.angle_min
        while current_angle <= scan_msg.angle_max:
            if self.angle_min <= current_angle <= self.angle_max:
                index = int((current_angle - scan_msg.angle_min) / scan_msg.angle_increment)
                range_scan = scan_msg.ranges[index]
                if np.isinf(range_scan):
                    range_scan = scan_msg.range_max
                    
                sampled_ranges.append(range_scan)
                sampled_angles.append(current_angle)
            current_angle += self.angle_increment
        
        # sampled_ranges.append(scan_msg.ranges[-1])
        # sampled_angles.append(current_angle)

        # Create a new LaserScan message to store filtered data
        filtered_scan_msg = LaserScan()
        filtered_scan_msg.header = scan_msg.header        
        filtered_scan_msg.angle_min = self.angle_min
        filtered_scan_msg.angle_max = self.angle_max
        filtered_scan_msg.angle_increment = self.angle_increment
        filtered_scan_msg.time_increment = scan_msg.time_increment
        filtered_scan_msg.scan_time = scan_msg.scan_time
        filtered_scan_msg.range_min = scan_msg.range_min
        filtered_scan_msg.range_max = scan_msg.range_max
        filtered_scan_msg.ranges = sampled_ranges

        # for i in range(len(filtered_scan_msg.ranges)):
        #     if filtered_scan_msg.ranges[i] < 2.0:
        #         filtered_scan_msg.ranges[i] = 0.5

        self.scan_pub.publish(filtered_scan_msg)
    

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    node = LaserDownsampler()
    node.run()
