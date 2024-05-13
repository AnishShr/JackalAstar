import os
import yaml

import cv2
import numpy as np

from pyproj import CRS, Transformer
from queue import PriorityQueue
import networkx as nx

import rospy
import tf2_ros
import tf
from nav_msgs.msg import Odometry
from sensor_msgs.msg import NavSatFix, LaserScan, Imu
from geometry_msgs.msg import Twist, Vector3Stamped
from visualization_msgs.msg import Marker, MarkerArray

from shapely.geometry import LineString, Point 
import cvxpy as cp

import time

from global_planner.data.mapping import MapReader
from global_planner.models.global_planner_onnx import GlobalPlannerOnnx
from global_planner.viz.global_planner_viz import GlobalPlannerViz

# Color constants, in RGB order
BLUE = (0, 0, 255)
RED  = (255, 0, 0)
GREEN = (0, 128, 0)
YELLOW = (255, 255, 0)
ORANGE = (255, 102, 0)
CYAN = (0, 255, 255)
FONT_SIZE = 0.3

# Trajectory colors, in BGR order
TRAJ_COLORS = (
    (255, 0, 0),     # BLUE    
    (0, 0, 255),     # RED
    (0, 255, 0),     # GREEN
    (0, 0, 0),       # BLACK
    (255, 0, 255),   # MAGENTA
    (255, 255, 255), # WHITE
    (128, 128, 128), # GRAY
    (255, 255, 0),   # YELLOW
    (0, 0, 128),     # NAVY BLUE
    (255, 165, 0)    # ORANGE   
)


DISTANCE_TO_GOAL_THRESHOLD = 2.0
DISTANCE_TO_TARGET_GPS = 0.6
BUFFER_DISTANCE = 0.4 

class SigmaWaypoints:
    def __init__(self, 
                 wp_node_id,
                 wp_gps, 
                 wp_prob):
        
        self.wp_node_id = wp_node_id                        # waypoint's parent node id in graph
        self.wp_gps = wp_gps                                # waypoint's gps position (lat, lon)
        self.wp_prob = wp_prob                              # waypoint's probability as predicted by global planner
        
        

class JackalAstar:

    def __init__(self):

        # Fetch parameters
        
        self.config_dir_path = rospy.get_param('~config_dir_path')
        self.global_model_path = rospy.get_param('~global_model_path')
        self.map_path = rospy.get_param('~map_path')
        self.map_name = rospy.get_param('~map_name')
        self.base_link_frame = rospy.get_param('~base_link_frame', 'base_link_frame')
        self.left_camera_frame = rospy.get_param('~left_camera_frame', 'zed_left_camera_frame')
        self.left_camera_optical_frame = rospy.get_param('~left_camera_optical_frame', 'zed_left_camera_optical_frame')
        self.record_video = rospy.get_param('~record_video')        
        self.timer_frequency = rospy.get_param('~timer_frequency')
        
        # Load global planner map
        map_type = self.global_planner_config["map_type"]
        map_file_path = os.path.join(self.map_path, f"{self.map_name}_{map_type}.tif")
        self.map_reader = MapReader(map_file_path, self.global_planner_config["map_size"])
        rospy.loginfo(f"Loaded global planner map: {map_file_path}")
        rospy.loginfo(f"Map resolution: {self.map_reader.map_resolution}")

        # Load global planner configration
        # global_planner_config_file = os.path.join(self.config_dir_path, 'distance_segment.yaml')
        global_planner_config_file = os.path.join(self.config_dir_path, 'default_segment.yaml')
        with open(global_planner_config_file, "r") as f:
            self.global_planner_config = yaml.safe_load(f)
        rospy.loginfo(f"Loaded global planner config: {global_planner_config_file}")   

        # Load global planner model
        self.global_planner = GlobalPlannerOnnx(self.map_reader, self.global_planner_config, self.global_model_path, convert_to_px=False)        
        rospy.loginfo(f"Loaded global planner model: {self.global_model_path}") 

        self.current_heading = None
        self.heading_calibrated = None

        self.prev_pose = None
        self.current_pose = None
        
        self.init_gps = None
        self.current_gps = None
        
        self.goal_gps = np.array([58.382749499999996, 26.726609399999997])  # (lat, lon) of goal point
        self.goal_px_position = self.map_reader.to_px(self.goal_gps)
        
        self.current_quat = None
        self.prev_quat = None

        self.state = 0      # robot state = 0 while initializing the node
        # robot states:
        # 0: initial GPS heading calibration
        # 1: propose waypoints
        # 2: navigate to the least cost waypoint
        # 3: check goal conditioning

        self.collision_free_waypoints = None
        self.valid_scans = None

        self.obs = None
        self.yaw = None

        self.obs_D = 0.5
        self.k = 0.9
        self.alpha = 0.2

        self.selected_wp = None
        self.selected_wp_gps = None
        self.current_goal_id = 0

        self.path_to_selected_wp = None
        self.gps_path = []
        self.parent = None

        self.num_nodes = 0
        self.node_id = 0
        self.prev_node_id = 0

        self.graph = nx.Graph()
        self.sigma = PriorityQueue()
        
        self.candidate_waypoints_gps = None
        self.gps_trajectory = []

        # TF buffer
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.tf_laser_to_base_link = self.create_tf_matrix(source_frame='zed_left_camera_frame',
                                                           target_frame='base_link')
        self.tf_navsat_to_base_link = self.create_tf_matrix(source_frame='imu_link',
                                                           target_frame='base_link')
        self.tf_laser_to_navsat = self.create_tf_matrix(source_frame='zed_left_camera_frame',
                                                        target_frame='imu_link')
        

        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)        
        self.marker_pub = rospy.Publisher('/visualization_marker', MarkerArray, queue_size=1, latch=True)

        rospy.Subscriber('/imu/data', Imu, self.imu_callback, queue_size=1)
        rospy.Subscriber('/odometry/filtered', Odometry, self.odom_callback, queue_size=1)
        rospy.Subscriber('/filter/positionlla', Vector3Stamped, self.gps_callback, queue_size=1)
        rospy.Subscriber('/scan/filtered', LaserScan, self.laser_callback, queue_size=1)        
        
        rospy.Timer(rospy.Duration(1.0/self.timer_frequency), self.timer_callback)

        # Initialize ROS publishers and subscribers
        self.driving_command_publisher = rospy.Publisher('cmd_vel',
                                                          Twist,
                                                          queue_size=1)


    def create_tf_matrix(self, source_frame, target_frame):
        """
        Creates a 4X4 transformation matrix that will transform one/multiple homogeneous coordinates
        from source frame to target frame
        Input  :  source frame, target frame
        Output :  4X4 transformation matrix
        """
        transform = self.tf_buffer.lookup_transform(target_frame=target_frame, 
                                                    source_frame=source_frame, 
                                                    time=rospy.Time(), 
                                                    timeout=rospy.Duration(5.0))

        translation = np.array([transform.transform.translation.x,
                                transform.transform.translation.y,
                                transform.transform.translation.z])
        rotation = np.array([transform.transform.rotation.x,
                             transform.transform.rotation.y,
                             transform.transform.rotation.z,
                             transform.transform.rotation.w])
         
        transformation_matrix = np.dot(
            tf.transformations.translation_matrix(translation),
            tf.transformations.quaternion_matrix(rotation)
        )
        
        return transformation_matrix

    def transform_points(self, input_points, tf_matrix):
        """
        Transforms a(or a set) of points from one frame to another based on the tf matrix
        Input  :  points in source frame of the tf_matrix
        Output :  points in target frame of the tf_matrix
        """

        points_in_target_frame = np.ones(shape=(input_points.shape[0], 4))
        points_in_target_frame[:, :2] = input_points
        points_in_target_frame = np.dot(points_in_target_frame, tf_matrix.T)
        # points_in_target_frame = np.dot(tf_matrix, points_in_target_frame.T)
        # points_in_target_frame = points_in_target_frame.T

        return points_in_target_frame[:, :2]

    
    def compute_heading_north(self, init_gps, final_gps):
        
        init_gps = init_gps.reshape(-1, 2)
        final_gps = final_gps.reshape(-1, 2)

        init_gps_rad = np.deg2rad(init_gps)
        final_gps_rad = np.deg2rad(final_gps)

        # delta_lon = final_gps_rad[1] - init_gps_rad[1]
        delta_lon = final_gps_rad[:,1] - init_gps_rad[:,1]

        # Calculate the difference in distances
        # delta_x = np.cos(final_gps_rad[0]) * np.sin(delta_lon)
        # delta_y = np.cos(init_gps_rad[0]) * np.sin(final_gps_rad[0]) - np.sin(init_gps_rad[0]) * np.cos(final_gps_rad[0]) * np.cos(delta_lon)
        delta_x = np.cos(final_gps_rad[:, 0]) * np.sin(delta_lon)
        delta_y = np.cos(init_gps_rad[:, 0]) * np.sin(final_gps_rad[:, 0]) - np.sin(init_gps_rad[:, 0]) * np.cos(final_gps_rad[:, 0]) * np.cos(delta_lon)

        bearing_rad = np.arctan2(delta_x, delta_y)
        bearing_degrees = np.rad2deg(bearing_rad)

        # if bearing_degrees > 180:
        #     bearing_degrees -= 360
        # elif bearing_degrees < -180:
        #     bearing_degrees += 360

        bearing_degrees[bearing_degrees < -180] += 360
        bearing_degrees[bearing_degrees >  180] -= 360

        return np.squeeze(bearing_degrees)

    
    def heading_calibration(self):
        """
        Drive 2.0m forward and calculate the current heading w.r.t. Geographic North
        
        """
        
        linear_vel = 0.0

        if np.linalg.norm(self.current_pose - self.prev_pose) <= 2.0:
            linear_vel = 0.4
            rospy.loginfo_once(f"Initial heading being calibrated ...")
        else:
            rospy.loginfo_once("Reached 2.0m mark !!")
            final_gps = self.current_gps
            bearing_north = self.compute_heading_north(self.init_gps, final_gps)
            rospy.loginfo_once("Initial robot heading w.r.t. Geographic North calibrated")
            rospy.loginfo_once(f"Initial heading: {bearing_north} degrees (Clockwise) w.r.t. North")
            self.current_heading = bearing_north
            self.heading_calibrated = True
            
            self.node_id = 0
            self.graph.add_node(self.node_id,
                                node_id=self.node_id,
                                node_gps=self.current_gps)
            print(f"Graph nodes: {self.graph.nodes.data()}")
            
            self.state = 1            
        
        vel = Twist()
        vel.linear.x = linear_vel
        self.cmd_vel_pub.publish(vel)

    
    def compute_gps(self, current_gps, waypoints):
        """
        Computes the GPS coordinates of the relative waypoints in base_link frame,
        and current GPS coordinate
        Input  :  current gps, relative waypoints(in base_link frame)
        Output :  GPS coordinates of the relative waypoints
        """
        
        relative_distances = np.sqrt((waypoints[:, 1])**2 + (waypoints[:, 0])**2)
        # Following right hand rule
        realtive_angles_rad = np.arctan2(waypoints[:, 1], waypoints[:, 0])
        realtive_angles = np.rad2deg(realtive_angles_rad)

        current_north_heading = self.current_heading
        waypoint_angles_north = current_north_heading - realtive_angles

        # if waypoint_angles_north > 180:
        #     waypoint_angles_north -= 360
        # elif waypoint_angles_north < 180:
        #     waypoint_angles_north += 360

        waypoint_angles_north[waypoint_angles_north < -180] += 360
        waypoint_angles_north[waypoint_angles_north > 180] -= 360

        current_gps_rad = np.deg2rad(current_gps)
        waypoint_angles_north_rad = np.deg2rad(waypoint_angles_north)

        cur_lat_rad = current_gps_rad[0]
        cur_lon_rad = current_gps_rad[1]
        
        R = 6371e3  # Earth's radius in meters

        new_lat = np.degrees(np.arcsin(np.sin(cur_lat_rad)*np.cos(relative_distances/R) + np.cos(cur_lat_rad)*np.sin(relative_distances/R)*np.cos(waypoint_angles_north_rad)))
        new_lon = current_gps[1] + np.degrees(np.arctan2(np.sin(waypoint_angles_north_rad)*np.sin(relative_distances/R)*np.cos(cur_lat_rad), 
                                                  np.cos(relative_distances/R)-np.sin(cur_lat_rad)*np.sin(np.radians(new_lat))))


        new_gps = np.column_stack((new_lat, new_lon))

        return np.squeeze(new_gps)
    

    def distance_from_gps(self, gps_origin, gps_destination):
        # gps are expected to be in angles (degrees)
        # first the degrees need to be converted to radians
        # then the distance is computed 

        # phi --> lat & lambda --> lon

        gps_origin = gps_origin.reshape(-1, 2)
        gps_destination = gps_destination.reshape(-1, 2)

        phi_1 = np.deg2rad(gps_origin[:, 0])           # gps --> (lat, lon); gps[0]=lat & gps[1]=lon 
        phi_2 = np.deg2rad(gps_destination[:, 0]) 

        del_phi = np.deg2rad(gps_destination[:, 0] - gps_origin[:, 0])
        del_lambda = np.deg2rad(gps_destination[:, 1] - gps_origin[:, 1])
        
        a = (np.sin(del_phi/2))**2 + np.cos(phi_1) * np.cos(phi_2) * (np.sin(del_lambda/2))**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        R = 6371e3 # earth's radius in meters
        d = R * c

        # if d.shape[0] > 1:
        #     d = np.squeeze(d)
        
        return np.squeeze(d)
    
    def compute_wp_prob(self, wp_gps):
        pass

    def compute_cost(self, parent_node, wp):

        wp_node_id = wp.wp_node_id
        wp_node_dist = nx.shortest_path_length(G=self.graph, 
                                               source=parent_node, 
                                               target=wp_node_id,
                                               weight='distance')
        wp_gps = wp.wp_gps
        wp_prob = wp.wp_prob
        wp_to_goal_distance = self.distance_from_gps(gps_origin=wp_gps,
                                                     gps_destination=self.goal_gps)

        local_cost = wp_node_dist
        global_cost = wp_to_goal_distance/wp_prob
        global_cost = global_cost
        
        total_cost = local_cost + global_cost
        return total_cost
    

    def update_priority_queue(self, parent_node, new_wp):
        updated_queue = PriorityQueue()
        
        # Update the pre-existing sigma        
        while not self.sigma.empty():
            _, existing_node = self.sigma.get()            
            wp_px = self.map_reader.to_px(existing_node.wp_gps)                      
            wp_px = np.array(wp_px).reshape(-1, 2)
            wp_prob = self.global_planner.calculate_probs(wp_px,
                                                          self.current_px_position,
                                                          self.current_heading)
            wp_prob = np.squeeze(wp_prob)            
            existing_node.wp_prob = wp_prob
            final_cost = self.compute_cost(parent_node, existing_node)   
            print(type(final_cost))         
            updated_queue.put((final_cost, existing_node))

        # # Add new wp in sigma after computing the new cost
        # final_cost = self.compute_cost(new_wp)
        # updated_queue.put((final_cost, new_wp))
        new_wp_px = self.map_reader.to_px(new_wp.wp_gps)
        new_wp_px = np.array(new_wp_px).reshape(-1, 2)
        new_wp_prob = self.global_planner.calculate_probs(new_wp_px,
                                                          self.current_px_position,
                                                          self.current_heading)
        new_wp_prob = np.squeeze(new_wp_prob)
        new_wp.wp_prob = new_wp_prob
        final_cost = self.compute_cost(new_wp)        
        updated_queue.put((final_cost, new_wp))

        return updated_queue

    def laser_callback(self, msg):

        ranges = np.array(msg.ranges)
        angles = np.array([msg.angle_min + i * msg.angle_increment \
                           for i in range(len(msg.ranges))])

        valid_indices = ranges >= 3.5
        invalid_indices = ranges < 3.5

        valid_ranges = ranges[valid_indices]
        valid_angles = angles[valid_indices]

        invalid_ranges = ranges[invalid_indices]
        invalid_angles = angles[invalid_indices]

        valid_coords_x = valid_ranges * np.cos(valid_angles)
        valid_coords_y = valid_ranges * np.sin(valid_angles)
        valid_coords = np.column_stack((valid_coords_x, valid_coords_y))
        
        invalid_coords_x = invalid_ranges * np.cos(invalid_angles)
        invalid_coords_y = invalid_ranges * np.sin(invalid_angles)
        invalid_coords = np.column_stack((invalid_coords_x, invalid_coords_y))
        
        invalid_shapely_points = [Point(x,y) for x,y in invalid_coords]
        # invalid_multi_points = MultiPoint(invalid_shapely_points)

        collision_free_waypoints = []
        
        for valid_id, valid_angle in enumerate(valid_angles):
            x = valid_ranges[valid_id] * np.cos(valid_angle)
            y = valid_ranges[valid_id] * np.sin(valid_angle)
            
            line = LineString([Point(0, 0), Point(x, y)])
            buffer_polygon = line.buffer(BUFFER_DISTANCE)
            points_inside_buffer = [point for point in invalid_shapely_points if buffer_polygon.contains(point)]
            points_outside_buffer = [point for point in invalid_shapely_points if not buffer_polygon.contains(point)]

            if len(points_inside_buffer) == 0:
                # Waypoint 1m in that direction
                wp_x = 2.0 * np.cos(valid_angle)
                wp_y = 2.0 * np.sin(valid_angle)
                collision_free_waypoints.append([wp_x, wp_y])
        
        collision_free_waypoints = np.array(collision_free_waypoints)

        if collision_free_waypoints.shape[0] == 0:
            # rospy.logwarn("No valid waypoints")
            self.waypoints_base_link = None
            return

        waypoints_base_link = self.transform_points(input_points=collision_free_waypoints,
                                                    tf_matrix=self.tf_laser_to_base_link)
        
        self.waypoints_base_link = waypoints_base_link
        
        # getting the obstacle info for the cbf controller        
        x_obs = ranges * np.cos(angles)
        y_obs = ranges * np.sin(angles)

        obs = np.column_stack((x_obs, y_obs))
        obs_base_link = self.transform_points(input_points=obs.reshape(-1,2),
                                              tf_matrix=self.tf_laser_to_base_link)
        obs_base_link = np.squeeze(obs_base_link)
        self.obs = obs_base_link

    def draw_cropped_map(self):
        global_planner_viz = GlobalPlannerViz(self.global_planner, adjust_heading=False)
        trajectories_map_img = global_planner_viz.plot_trajectories_map(self.current_px_position,
                                                                        self.goal_px_position,
                                                                        self.current_heading)
        probability_map_img = global_planner_viz.plot_probability_map(self.current_px_position, self.goal_px_position)

        result_img = np.concatenate((trajectories_map_img, probability_map_img), axis=1)
        result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)  

        return result_img
    
    def viz_current_state(self, result_img, parent_node, candidate_wps, path_through_graph, selected_wp):
        if len(self.gps_trajectory) == 0:
            return
        
        # Visualize the parent node proposing waypoints
        parent_node_wp_gps = self.graph.nodes[parent_node]['node_gps']
        parent_node_px_coords = self.map_reader.to_px(parent_node_wp_gps)
        parent_node_crop_coords = self.map_reader.to_crop_coords(self.current_px_position, parent_node_px_coords)
        cv2.circle(result_img, parent_node_crop_coords, 2, (255, 255, 255), 2)

        # Visualize candidate waypoints
        for candidate_gps in self.candidate_waypoints_gps:
            candidate_px_coords = self.map_reader.to_px(candidate_gps)
            candidate_crop_coords = self.map_reader.to_crop_coords(self.current_px_position, candidate_px_coords)
            cv2.circle(result_img, candidate_crop_coords, 2, (255, 0, 0), 2)
        
        # visualize graph nodes
        for node in self.graph.nodes.data():
            node_gps = self.graph.nodes[node]['node_gps']
            node_px_coords = self.map_reader.to_px(node_gps)
            node_crop_coords = self.map_reader.to_crop_coords(self.current_px_position, node_px_coords)
            cv2.circle(result_img, node_crop_coords, 2, (0, 0, 255), 2)

        # Visualize path to the least cost wp
        for node in path_through_graph:
            node_gps = self.graph.nodes[node]['node_gps']
            node_px_coords = self.map_reader.to_px(node_gps)
            node_crop_coords = self.map_reader.to_crop_coords(self.current_px_position, node_px_coords)
            cv2.circle(result_img, node_crop_coords, 2, (255, 0, 0), 2)

        selected_wp_gps = selected_wp.wp_gps
        selected_wp_px_coords = self.map_reader.to_px(selected_wp_gps)
        selected_wp_crop_coords = self.map_reader.to_crop_coords(self.current_px_position, selected_wp_px_coords)
        cv2.circle(result_img, selected_wp_crop_coords, 2, (0, 255, 0), 2)

        self.show_img(result_img)
        key = cv2.waitKey(int(1/self.timer_frequency*1000))            
        if key == 27:                       

            cv2.destroyAllWindows()
            rospy.signal_shutdown("User pressed ESC")
        
    def show_img(self, map_img):        
        height, width = map_img.shape[:2]        
        map_img = cv2.resize(map_img, (2 * width, 2 * height))
        cv2.imshow('SIMULATOR', map_img)

    def follow_path(self, path_through_graph, gps_path):

        if self.current_goal_id >= len(gps_path):
            # Completed all gps points
            # update the state to 3
            self.current_goal_id = 0            
            self.node_id = path_through_graph[-1]

            self.gps_path = []
            self.path_through_graph = None 

            # current_target = ros_point()
            # self.current_target_pub.publish(current_target)

            self.state = 3

        else:
            dist_from_target_wp = self.distance_from_gps(gps_origin=self.current_gps,
                                                          gps_destination=gps_path[self.current_goal_id])
            dist_from_target_wp = np.squeeze(dist_from_target_wp)
            print(dist_from_target_wp)
            
            linear_vel = 0
            angular_vel = 0

            if dist_from_target_wp < DISTANCE_TO_TARGET_GPS:
                
                self.node_id = path_through_graph[self.current_goal_id]
                self.current_goal_id += 1
            
            else:
                
                heading_north_target_wp = self.compute_heading_north(init_gps=self.current_gps,
                                                                     final_gps=gps_path[self.current_goal_id])
                heading_local_target_wp = self.current_heading - heading_north_target_wp

                if heading_local_target_wp < -180:
                    heading_local_target_wp += 360
                elif heading_local_target_wp > 180:
                    heading_local_target_wp -= 360
                
                heading_local_target_wp_rad = np.deg2rad(heading_local_target_wp)

                if -75 < heading_local_target_wp < 75:

                    # position in navsat_link frame
                    target_x = dist_from_target_wp * np.cos(heading_local_target_wp_rad)
                    target_y = dist_from_target_wp * np.sin(heading_local_target_wp_rad)
                    
                    target_navsat = np.array([target_x, target_y])
                    # print(target_navsat)

                    tf_navsat_to_odom = self.create_tf_matrix(source_frame='navsat_link',
                                                            target_frame='odom')
                    target_odom = self.transform_points(input_points=target_navsat.reshape(-1,2),
                                                        tf_matrix=tf_navsat_to_odom)
                    target_odom = np.squeeze(target_odom)
                    # print(target_odom)

                    obs_pose = self.obs
                    tf_base_link_to_odom = self.create_tf_matrix(source_frame='base_link',
                                                            target_frame='odom')
                    obs_pose_odom = self.transform_points(input_points=obs_pose.reshape(-1,2),
                                                        tf_matrix=tf_base_link_to_odom)

                    current_pose = self.current_pose
                    # print(current_pose)

                    xi = cp.Variable(2)
                    v_x = xi[0]
                    v_y = xi[1]

                    cost = (v_x + self.k*(current_pose[0]-target_odom[0]))**2 + (v_y + self.k*(current_pose[1]-target_odom[1]))**2      
                    constraints = [cp.norm(xi, 2) <= 0.3]

                    for i in range(len(obs_pose_odom)):
                        hx = np.sqrt((current_pose[0]-obs_pose_odom[i][0])**2 + (current_pose[1]-obs_pose_odom[i][1])**2) - self.obs_D
                        grad_hx = np.vstack(((current_pose[0]-obs_pose_odom[i][0])/(hx+self.obs_D),
                                            (current_pose[1]-obs_pose_odom[i][1])/(hx+self.obs_D)))

                        constraints.append(grad_hx[0]*v_x + grad_hx[1]*v_y + self.alpha * hx >= 0)
                
                    prob = cp.Problem(cp.Minimize(cost),
                                    constraints)
                    prob.solve()

                    v_x = xi.value[0]
                    v_y = xi.value[1]
                    theta = np.arctan2(v_y, v_x)

                    v = np.sqrt((v_x**2) + (v_y**2))                
                    vel_x = v * np.cos(theta)
                    vel_y = v * np.sin(theta)

                    rot_matrix = np.array([[np.cos(self.yaw), np.sin(self.yaw)],
                                            [-np.sin(self.yaw), np.cos(self.yaw)]])
                    v = np.dot(rot_matrix, np.array([vel_x, vel_y]))

                    linear_vel = np.sqrt(v[0]**2 + v[1]**2)                
                    angular_vel = v[1]/0.45
                
                else:

                    linear_vel = 0

                    if heading_local_target_wp < 0:
                        angular_vel = -np.deg2rad(30)
                    else:
                        angular_vel = np.rad2deg(30)

            vel = Twist()
            vel.linear.x = np.clip(linear_vel, 0, 0.3)
            vel.angular.z = np.clip(angular_vel, -np.deg2rad(45), np.deg2rad(45))
            self.cmd_vel_pub.publish(vel)


    def imu_callback(self, msg):

        if self.state == 0:
            return

        self.current_quat = np.array([msg.orientation.x, 
                                      msg.orientation.y, 
                                      msg.orientation.z, 
                                      msg.orientation.w])

        if self.prev_quat is None:
            self.prev_quat = self.current_quat

        current_yaw = tf.transformations.euler_from_quaternion(self.current_quat)[2]
        prev_yaw = tf.transformations.euler_from_quaternion(self.prev_quat)[2]

        delta_yaw = current_yaw - prev_yaw
        delta_yaw = np.rad2deg(delta_yaw)

        self.current_heading -= delta_yaw
        
        if self.current_heading < -180:
            self.current_heading += 360
        elif self.current_heading > 180:
            self.current_heading -= 360

        self.prev_quat = self.current_quat

    
    def odom_callback(self, msg):
        
        if self.current_gps is None:
            return

        # Set the initial GPS coordiantes
        if self.prev_pose is None:
            self.init_gps = self.current_gps
            self.prev_pose = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y])
        
        self.current_pose = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y])
        
        orientation = msg.pose.pose.orientation
        quaternion = [orientation.x, orientation.y, orientation.z, orientation.w] 
        roll, pitch, yaw = tf.transformations.euler_from_quaternion(quaternion)

        self.yaw = yaw
    
    def gps_callback(self, msg):
        self.current_gps = np.array([msg.vector.x, msg.vector.y])
        self.current_px_position = self.map_reader.to_px(self.current_gps)

        if self.heading_calibrated:
            self.gps_trajectory.append(self.current_gps)

    
    def timer_callback(self, event=None):
        
        if self.goal_gps is None:
            return
        
        node_id = self.node_id
        prev_node_id = self.prev_node_id

        waypoints_base_link = self.waypoints_base_link
        
        dist_from_final_goal = self.distance_from_gps(gps_origin=self.current_gps,
                                                      gps_destination=self.goal_gps)
        
        if dist_from_final_goal <= DISTANCE_TO_GOAL_THRESHOLD:

            vel = Twist()
            self.cmd_vel_pub.publish()
            self.goal_gps = None

            return
        
        if self.state == 0:
            self.heading_calibration()

        elif self.state == 1:
            rospy.loginfo_once("Robot is in state 1 ..")

            parent_node = self.node_id
            
            if waypoints_base_link is None:
                angular_vel = np.rad2deg(30)

                vel = Twist()
                vel.angular.z = angular_vel

                self.cmd_vel_pub.publish(vel)
            
            else:
                waypoints_gps = self.compute_gps(current_gps=self.current_gps,
                                                 waypoints=waypoints_base_link)
                self.candidate_waypoints_gps = waypoints_gps
                
                self.global_planner.predict_probabilities(self.current_px_position, self.goal_px_position)

                for i in range(waypoints_base_link.shape[0]):
                    self.num_nodes = self.graph.number_of_nodes()
                    self.node_id = self.num_nodes

                    nodes_to_merge_with = []
                    for node in self.graph_nodes():
                        if self.distance_from_gps(gps_origin=waypoints_gps[i],
                                                  gps_destination=self.graph.nodes[node]['node_gps']) < 1.0:
                            nodes_to_merge_with.append(node)
                    
                    if len(nodes_to_merge_with) == 0:
                        # add node and edge
                        self.graph.add_node(self.node_id,
                                            node_id=self.node_id,
                                            node_gps=waypoints_gps[i])
                        print(f"Added node {self.node_id} in graph")                        

                        self.graph.add_edge(parent_node, self.node_id,
                                            distance=self.calculate_edge_distance(parent_node, self.node_id))
                        print(f"Added edge between {self.node_id} and {parent_node} in graph")    

                        

                        wp = SigmaWaypoints(wp_node_id=self.node_id,
                                            wp_gps=waypoints_gps[i],
                                            wp_prob=None)
                        
                        print(f"added node {self.node_id} in priority queue")
                        self.sigma = self.update_priority_queue(parent_node=parent_node, new_wp=wp)                    

                    else:
                        for node in nodes_to_merge_with:
                                                       
                            self.graph.add_edge(parent_node, node, distance=self.calculate_edge_distance(origin_id=parent_node,
                                                                                                         destination_id=node))
                            print(f"Updated edge between {parent_node} and {node} in graph")
                    
                    
                
                for node in self.graph.nodes():
                    for another_node in self.graph.nodes():
                        if node != another_node:
                            dist = self.distance_from_gps(gps_origin=self.graph.nodes[node]['node_gps'],
                                                          gps_destination=self.graph.nodes[another_node]['node_gps'])
                            if 1.0 < dist < 1.5:
                                if self.graph.has_edge(node, another_node):
                                    self.graph.edges[node, another_node]['distance'] = dist
                                else:
                                    self.graph.add_edge(node, another_node,
                                            distance=dist)
                    

                print("---------------------------------------------------------------")

                # print(f"Graph nodes: \n{self.graph.nodes.data()}")

                _, lowest_cost_wp = self.sigma.get()
                print(lowest_cost_wp.wp_node_id)

                path_to_lowest_cost_wp = nx.shortest_path(G=self.graph,
                                                   source=parent_node,
                                                   target=lowest_cost_wp.wp_node_id,
                                                   weight='distance')
                print(f"path to the lowest cost wp: {path_to_lowest_cost_wp}")

                self.path_through_graph = path_to_lowest_cost_wp[1:]
                self.parent = path_to_lowest_cost_wp[0]
                # self.gps_path = []
                for node_id in self.path_through_graph:
                    self.gps_path.append(self.graph.nodes[node_id]['node_gps'])
                
                self.state = 2


        elif self.state == 2:
            path_through_graph = self.path_through_graph
            gps_path = self.gps_path

            self.publish_markers(parent=self.parent, path=path_through_graph, goal_gps=self.goal_gps)                        

            result_img = self.draw_cropped_map()
            self.viz_current_state(result_img=result_img, 
                                   parent_node=self.parent, 
                                   path_through_graph=path_through_graph,
                                   selected_wp=lowest_cost_wp)
            
            self.follow_path(path_through_graph, gps_path)

        elif self.state == 3:

            dist_from_goal = self.distance_from_gps(gps_origin=self.current_gps,
                                                    gps_destination=self.goal_gps)
            
            if dist_from_goal > DISTANCE_TO_GOAL_THRESHOLD:
                self.state = 1

            else:
                rospy.loginfo_once("Reached final goal")

                vel = Twist()
                self.cmd_vel_pub.publish(vel)
                self.goal_gps = None

if __name__ == "__main__":
    rospy.init_node("Jackal A Star", log_level=rospy.INFO)
    node = JackalAstar()    
    rospy.spin()