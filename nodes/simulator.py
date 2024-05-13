import os
import yaml

import cv2
import numpy as np

from pyproj import CRS, Transformer
from queue import PriorityQueue
import networkx as nx

import rospy
from sensor_msgs.msg import NavSatFix
from geometry_msgs.msg import Twist

import time

from global_planner.data.mapping import MapReader
from global_planner.models.global_planner_onnx import GlobalPlannerOnnx
from global_planner.viz.global_planner_viz import GlobalPlannerViz
from global_planner.viz.util import calculate_angle

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

NUM_SAMPLED_WAYPOINTS = 5
DISTANCE_TO_GOAL_THRESHOLD = 10.0

class SigmaWaypoints:
    def __init__(self, 
                 wp_node_id, 
                 node_distance, 
                 wp_distance, 
                 wp_gps, 
                 wp_prob, 
                 wp_to_goal_distance):
        
        self.wp_node_id = wp_node_id                        # waypoint's parent node id in graph
        self.node_distance = node_distance                  # distance between the current node and the waypoint's parent node in the graph
        self.wp_distance = wp_distance                      # distance between the waypoint's parent node and the waypoint itself
        self.wp_gps = wp_gps                                # waypoint's gps position (lat, lon)
        self.wp_prob = wp_prob                              # waypoint's probability as predicted by global planner
        self.wp_to_goal_distance = wp_to_goal_distance      # euclidean distance between the wapypoint and the goal
        

class Visualizer:

    def __init__(self):

        # Fetch parameters
        
        self.config_dir_path = rospy.get_param('~config_dir_path')
        self.global_model_path = rospy.get_param('~global_model_path')
        self.map_path = rospy.get_param('~map_path')
        self.map_name = rospy.get_param('~map_name')
        self.initial_x = np.float32(rospy.get_param('~initial_x'))
        self.initial_y = np.float32(rospy.get_param('~initial_y'))
        self.initial_lat = np.float32(rospy.get_param('~initial_lat'))
        self.initial_lon = np.float32(rospy.get_param('~initial_lon'))                        
        self.base_link_frame = rospy.get_param('~base_link_frame', 'base_link_frame')
        self.left_camera_frame = rospy.get_param('~left_camera_frame', 'zed_left_camera_frame')
        self.left_camera_optical_frame = rospy.get_param('~left_camera_optical_frame', 'zed_left_camera_optical_frame')
        self.record_video = rospy.get_param('~record_video')        
        self.timer_frequency = rospy.get_param('~timer_frequency')
        
        
        # Load global planner configration
        # global_planner_config_file = os.path.join(self.config_dir_path, 'distance_segment.yaml')
        global_planner_config_file = os.path.join(self.config_dir_path, 'default_segment.yaml')
        with open(global_planner_config_file, "r") as f:
            self.global_planner_config = yaml.safe_load(f)
        rospy.loginfo(f"Loaded global planner config: {global_planner_config_file}")

        # Load global planner map
        map_type = self.global_planner_config["map_type"]
        map_file_path = os.path.join(self.map_path, f"{self.map_name}_{map_type}.tif")
        self.map_reader = MapReader(map_file_path, self.global_planner_config["map_size"])
        rospy.loginfo(f"Loaded global planner map: {map_file_path}")
        rospy.loginfo(f"Map resolution: {self.map_reader.map_resolution}")

        # Load global planner model
        self.global_planner = GlobalPlannerOnnx(self.map_reader, self.global_planner_config, self.global_model_path, convert_to_px=False)        
        rospy.loginfo(f"Loaded global planner model: {self.global_model_path}")        

        self.video_resolution = (int(4.5*self.map_reader.map_size), int(3*self.map_reader.map_size))
        # Initialize video writer
        if self.record_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            # self.video = cv2.VideoWriter(self.record_video, fourcc, int(self.timer_frequency), (640, 360))
            self.video = cv2.VideoWriter(self.record_video, fourcc, int(self.timer_frequency), self.video_resolution)
            rospy.loginfo(f"Recording video to: {self.record_video}")


        # Create an empty graph
        self.graph = nx.Graph()
        # Create an empty open set Sigma
        self.sigma = PriorityQueue()
        self.wp_gps = set()

        self.node_id = 0
        self.prev_node_id = None

        # Initialize internal variables
        self.goal_gps = None
        self.current_gps = None
        self.current_heading = None
        
        self.crs_wgs84 = CRS.from_epsg(4326)
        self.crs_utm = CRS.from_epsg(32635)
        self.transformer = Transformer.from_crs(self.crs_wgs84, self.crs_utm)
        
        self.current_gps = None
        self.init_lat = None
        self.init_lon = None
            
        self.angle = 0
        print(self.angle)
        print(type(self.angle))   
        self.angle = np.int32(self.angle)     

        self.current_heading = 0
        
        self.prev_position = None
        
        self.counter = 0
        self.num_timesteps = 0

        rospy.Timer(rospy.Duration(1.0/self.timer_frequency), self.timer_callback)

        # Initialize ROS publishers and subscribers
        self.driving_command_publisher = rospy.Publisher('cmd_vel',
                                                          Twist,
                                                          queue_size=1)

        # rospy.Subscriber('/goal_gps',
        #                 NavSatFix,
        #                 self.goal_gps_callback)

        # self.goal_gps = np.array([58.384611, 26.725902]) # Delta
        # self.goal_gps = np.array([58.383218, 26.727666]) # Rossi 1
        self.goal_gps = np.array([self.initial_lat, self.initial_lon])
        self.goal_position = self.map_reader.to_px(self.goal_gps)
        self.goal_utm_position = self.transform(self.goal_gps[0], self.goal_gps[1])

        init_x_pos, init_y_pos = np.array([self.initial_x, self.initial_y]) + np.array([self.goal_utm_position[0], self.goal_utm_position[1]])

        self.init_lat, self.init_lon = self.transformer.transform(init_x_pos, init_y_pos, direction="INVERSE")
        self.current_gps = np.array([self.init_lat, self.init_lon])
        
        self.current_position = self.map_reader.to_px(np.array(self.current_gps))
        self.current_utm = np.array([init_x_pos, init_y_pos])

        self.prev_gps = None
        

    def transform(self, lat, lon):

        coords = self.transformer.transform(lat, lon)

        easting = coords[0]
        northing = coords[1]

        return easting, northing

    def draw_cropped_map(self):
        """
        Function to visualize the cropped map based on current position.
        Current position is always in the center.        
        On top of this map image, the candidate waypoints or the graph is visualized by the
        'viz_current_state' funtion. 

        candidate waypoints: pixel coordinates of the relative candidate waypoints
        """
        global_planner_viz = GlobalPlannerViz(self.global_planner, adjust_heading=False)        
        # trajectories_map_img = global_planner_viz.plot_trajectories_map(self.current_position,
        #                                                                 self.goal_position,
        #                                                                 self.north_heading,
        #                                                                 [candidate_waypoints],
        #                                                                 TRAJ_COLORS)
        trajectories_map_img = global_planner_viz.plot_trajectories_map(self.current_position,
                                                                        self.goal_position,
                                                                        self.north_heading)
        probability_map_img = global_planner_viz.plot_probability_map(self.current_position, self.goal_position)
        result_img = np.concatenate((trajectories_map_img, probability_map_img), axis=1) 
        # result_img --> cropped map
        # result_img = cv2.cvtColor(trajectories_map_img, cv2.COLOR_BGR2RGB)    
        result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)    

        return result_img
    
    def viz_gps_point(self, result_img, gps, color_current_pos, color_selected_point, radius_selected_point=2):        

        selected_node_px_coords = self.map_reader.to_px(gps)
        selected_node_crop_coords = self.map_reader.to_crop_coordinates(self.current_position, selected_node_px_coords)
        cv2.circle(result_img, selected_node_crop_coords, radius_selected_point, color_selected_point, 2)

        self.show_img(result_img)
        
        
    def viz_current_state(self, result_img, candidate_waypoints, path, selected_wp):
        """
        Depending on the path from current node in graph to the best wp in the priority queue,
        this function visualizes either the candidate waypoints, or the graph
        backtracking on top of the cropped map.

        result_img: cropped map
        candidate_waypoints: pixel coordinates of the relative candidate waypoints
        path: path from the current node in graph to the node annotated with the best global planner selected waypoint
        selected_wp: best wp that the global planner selects 
        graph: networkX graph that is being built ON-THE-GO
        """
        
        for node in self.graph.nodes:
            node_gps = self.graph.nodes.data()[node]['node_gps']
            node_px_coords = self.map_reader.to_px(node_gps)
            node_crop_coords = self.map_reader.to_crop_coordinates(self.current_position, node_px_coords)
            cv2.circle(result_img, node_crop_coords, 2, (255, 0, 0), 2)

        if len(path) != 1:
            
            path_gps = []

            for node_id in path:
                node_gps = self.graph.nodes.data()[node_id]['node_gps']
                path_gps.append(node_gps)
            
            best_wp_gps = selected_wp.wp_gps
            best_wp_px_coords = self.map_reader.to_px(best_wp_gps)
            best_wp_crop_coordinates = self.map_reader.to_crop_coordinates(self.current_position, best_wp_px_coords)

            # Visualizes the sub-graph that is to be back traversed (in white)
            for gps in path_gps:
                self.viz_gps_point(result_img=result_img,
                                   gps=gps,
                                   color_current_pos=(255, 0, 0),
                                   color_selected_point=(255, 255, 255))

            # Visualizes the robot hopping over graph nodes to reach the best waypoint in priority queue (in blue)
            for gps in path_gps:
                self.viz_gps_point(result_img=result_img,
                                   gps=gps,
                                   color_current_pos=(255, 0, 0),
                                   color_selected_point=(255, 0, 0))

                key = cv2.waitKey(int(1/self.timer_frequency*1000))
                if key == 27:            
                    if self.record_video:
                        self.video.release()
                    
                    print("-------------------------------------------------------------------------------")
                    print("-------------------------------------------------------------------------------")

                    cv2.destroyAllWindows()
                    rospy.signal_shutdown("User pressed ESC")

            # Visualizes the best waypoint from the entire priority queue (in green)
            # Once the robot reaches the final node after back-tracking
            cv2.circle(result_img, best_wp_crop_coordinates, 2, (0, 255, 0), 2)            
            self.show_img(result_img)
            
            key = cv2.waitKey(int(1/self.timer_frequency*1000))            
            if key == 27:            
                if self.record_video:
                    self.video.release()
                
                print("-------------------------------------------------------------------------------")
                print("-------------------------------------------------------------------------------")

                cv2.destroyAllWindows()
                rospy.signal_shutdown("User pressed ESC")

            # Update new gps position of the robot after backtracking the graph
            # previous position --> parent node of the best wp in sigma
            # new position --> best wp in sigma
            self.prev_gps = path_gps[-1]
            self.prev_position = self.map_reader.to_px(self.prev_gps)

            self.current_gps = selected_wp.wp_gps
            self.current_position = self.map_reader.to_px(self.current_gps)

            # Calculate current heading of the robot (from the parent node of the best wp to the wp itself)
            self.current_heading = calculate_angle(self.prev_position, self.current_position)            

            # Update node id
            # previous node id --> node id of the parent node in the graph
            # new node id --> latest node id + 1
            self.prev_node_id = selected_wp.wp_node_id
            self.node_id += 1

            self.num_timesteps += 1

            

            return            

        # Visualize the candidate wps 
        for i in range(NUM_SAMPLED_WAYPOINTS):
                wp_crop_coord = self.map_reader.to_crop_coordinates(self.current_position, candidate_waypoints[i])
                cv2.circle(result_img, wp_crop_coord, 2, (0, 0, 255), 2)
        
        self.viz_gps_point(result_img=result_img,
                                   gps=selected_wp.wp_gps,
                                   color_current_pos=(255, 0, 0),
                                   color_selected_point=(255, 255, 255),
                                   radius_selected_point=4)
        
        self.show_img(result_img)

        # Update the new gps position for the robot
        self.prev_gps = self.current_gps
        self.prev_position = self.current_position

        self.current_gps = selected_wp.wp_gps        
        self.current_position = self.map_reader.to_px(self.current_gps)

        # Calculate current heading of the robot
        self.current_heading = calculate_angle(self.prev_position, self.current_position)

        # Update node id in the graph
        self.prev_node_id = self.node_id
        self.node_id += 1
        self.num_timesteps += 1
    
    
    def show_img(self, map_img):        
        height, width = map_img.shape[:2]        
        map_img = cv2.resize(map_img, (2 * width, 2 * height))
        cv2.imshow('SIMULATOR', map_img)
        
        if self.record_video:
                self.video.write(cv2.resize(map_img, self.video_resolution))


    def sample_waypoints(self, num_samples, alpha_range=150, min_delta=7, max_delta=12):
        """
        Proposes waypoints in meters relative to robot's current position
        """
        alpha = np.random.randint(0, alpha_range, size=num_samples)        
        alpha = (alpha - alpha_range//2)

        # num_wps = num_samples
        # alpha = [i for i in range(0, alpha_range, alpha_range//num_wps)]
        # alpha = np.array(alpha)

        alpha_rad = np.deg2rad(alpha)
        
        delta_x = np.random.randint(min_delta, max_delta, size=num_samples) * np.cos(alpha_rad)
        delta_y = np.random.randint(min_delta, max_delta, size=num_samples) * np.sin(alpha_rad)
        
        deltas = np.column_stack((delta_x, delta_y))        

        return deltas  


    def compute_gps(self, current_gps, relative_distance, theta):
        """
        Computes the gps position of a point at a relative distance (w.r.t. current position) 
        which has a bearing of 'theta' w.r.t. North in clockwise direction
        relative_distance: np array or a float(distance)
        theta: np array or a float(distance)
        """
        lat = current_gps[0]
        lon = current_gps[1]

        cur_lat_rad = np.radians(lat)
        cur_lon_rad = np.radians(lon)

        R = 6371e3 # earth's radius in meters

        theta_rad = np.radians(theta)

        new_lat = np.degrees(np.arcsin(np.sin(cur_lat_rad)*np.cos(relative_distance/R) + np.cos(cur_lat_rad)*np.sin(relative_distance/R)*np.cos(theta_rad)))
        new_lon = lon + np.degrees(np.arctan2(np.sin(theta_rad)*np.sin(relative_distance/R)*np.cos(cur_lat_rad), 
                                                  np.cos(relative_distance/R)-np.sin(cur_lat_rad)*np.sin(np.radians(new_lat)))) 
        
        new_gps = np.column_stack((new_lat, new_lon))
        return new_gps
    
    
    def goal_gps_callback(self, gps_msg):
        
        self.goal_gps = np.array([gps_msg.latitude, gps_msg.longitude])
        self.goal_position = self.map_reader.to_px(self.goal_gps)
        self.goal_utm_position = self.transform(self.goal_gps[0], self.goal_gps[1])

        init_x_pos, init_y_pos = np.array([self.initial_x, self.initial_y]) + np.array([self.goal_utm_position[0], self.goal_utm_position[1]])

        self.init_lat, self.init_lon = self.transformer.transform(init_x_pos, init_y_pos, direction="INVERSE")
        self.current_gps = np.array([self.init_lat, self.init_lon])
        # self.current_gps = (58.38483793171251, 26.726995343394613)
        self.current_position = self.map_reader.to_px(np.array(self.current_gps))
        self.current_utm = np.array([init_x_pos, init_y_pos])

        # heading w.r.t east in Counter-clockwise direction
        self.current_heading = 0

        self.prev_gps = None
        self.prev_position = None

        
    def distance_from_gps(self, gps_origin, gps_destination):
        # gps are expected to be in angles (degrees)
        # first the degrees need to be converted to radians
        # then the distance is computed 

        # phi --> lat & lambda --> lon

        phi_1 = np.deg2rad(gps_origin[0])           # gps --> (lat, lon); gps[0]=lat & gps[1]=lon 
        phi_2 = np.deg2rad(gps_destination[0]) 

        del_phi = np.deg2rad(gps_destination[0] - gps_origin[0])
        del_lambda = np.deg2rad(gps_destination[1] - gps_origin[1])
        
        a = (np.sin(del_phi/2))**2 + np.cos(phi_1) * np.cos(phi_2) * (np.sin(del_lambda/2))**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        R = 6371e3 # earth's radius in meters
        d = R * c

        return d
    

    def calculate_edge_distance(self, origin_id, destination_id):
        
        gps_origin = self.graph.nodes.data()[origin_id]['node_gps']
        gps_destination = self.graph.nodes.data()[destination_id]['node_gps']

        dist = self.distance_from_gps(gps_origin, gps_destination)
        return dist
    
    
    def compute_cost(self, wp):

        wp_node_id = wp.wp_node_id
        wp_node_dist = nx.shortest_path_length(G=self.graph, 
                                               source=self.node_id, 
                                               target=wp_node_id,
                                               weight='distance')
        wp_dist = wp.wp_distance
        wp_gps = wp.wp_gps
        wp_prob = wp.wp_prob
        wp_to_goal_distance = wp.wp_to_goal_distance

        local_cost = wp_node_dist + wp_dist
        global_cost = wp_to_goal_distance/wp_prob
        # global_cost = wp_to_goal_distance*(1-wp_prob)
        
        
        total_cost = local_cost + global_cost
        
        return total_cost
    

    def update_priority_queue(self, new_wp):
        updated_queue = PriorityQueue()
        
        # Update the pre-existing sigma        
        while not self.sigma.empty():
            _,_, existing_node = self.sigma.get()            
            wp_px = self.map_reader.to_px(existing_node.wp_gps)                      
            wp_px = np.array(wp_px).reshape(-1, 2)            

            wp_prob = self.global_planner.calculate_probs(wp_px,
                                                          self.current_position,
                                                          self.north_heading)
            wp_prob = np.squeeze(wp_prob)            
            existing_node.wp_prob = wp_prob
            final_cost = self.compute_cost(existing_node)
            self.counter += 1     
            updated_queue.put((final_cost, self.counter, existing_node))

        # # Add new wp in sigma after computing the new cost
        # final_cost = self.compute_cost(new_wp)
        # updated_queue.put((final_cost, new_wp))
        new_wp_px = self.map_reader.to_px(new_wp.wp_gps)
        new_wp_px = np.array(new_wp_px).reshape(-1, 2)

        new_wp_prob = self.global_planner.calculate_probs(new_wp_px,
                                                          self.current_position,
                                                          self.north_heading)
        new_wp_prob = np.squeeze(new_wp_prob)
        
        new_wp.wp_prob = new_wp_prob
        final_cost = self.compute_cost(new_wp)
        self.counter += 1        
        updated_queue.put((final_cost, self.counter, new_wp))

        return updated_queue

    
    def timer_callback(self, event=None):
        
        print(f"Total Robot Timesteps: {self.num_timesteps}")

        if self.current_gps is not None:
            
            print(f"Current GPS: {self.current_gps}")
            print(f"Current position: {self.current_position}")
            print(f"Current heading: {self.current_heading}")

            print(f"Goal GPS: {self.goal_gps}")           
            print(f"Goal position: {self.goal_position}")

            current_crop_coords = self.map_reader.to_crop_coordinates(self.current_position, self.current_position)
            print(f"Current crop coords: {current_crop_coords}")            

            goal_crop_coords = self.map_reader.to_crop_coordinates(self.current_position, self.goal_position)
            print(f"Goal crop coords: {goal_crop_coords}")            

            if self.prev_gps is not None:

                distance_to_goal = self.distance_from_gps(self.current_gps, self.goal_gps)
                print(f"distance to goal: {distance_to_goal}")

                if distance_to_goal < DISTANCE_TO_GOAL_THRESHOLD:            
                    print("Final Goal reached !!")                    
                    if self.record_video:
                        self.video.release()
                    print("-------------------------------------------------------------------------------")
                    print("-------------------------------------------------------------------------------")

                    cv2.destroyAllWindows()
                    rospy.signal_shutdown("User pressed ESC")
                            
                key = cv2.waitKey(1)
                # Exit on 'ESC' press or when the robot is within the proximity threshold near the goal
                if key == 27:
                    print("User enabled STOP. Exiting the node..")            
                    if self.record_video:
                        self.video.release()                

            self.graph.add_node(self.node_id, 
                                node_id=self.node_id, 
                                node_gps=self.current_gps)
            if self.node_id > 0:
                self.graph.add_edge(self.node_id-1, self.node_id, 
                                    distance=self.calculate_edge_distance(self.prev_node_id, self.node_id)
                                   )

            self.north_heading = self.current_heading + 90            

            # Candidate waypoints relative to the robot's current position in meters                        
            candidate_waypoints = self.sample_waypoints(NUM_SAMPLED_WAYPOINTS)

            # relative distances of the candidate waypoints
            candidate_waypoints_distances = np.linalg.norm(candidate_waypoints, axis=1)

            # angles w.r.t. to robot's current heading as per Right-Hand-Rule direction convention
            # i.e., positive --> Counter-clockwise || negative --> Clockwise
            candidate_waypoints_angles_rad = np.arctan2(candidate_waypoints[:, 1], candidate_waypoints[:, 0])
            candidate_waypoints_angles_deg = np.degrees(candidate_waypoints_angles_rad)

            # wp angles w.r.t. north in clockwise direction
            waypoints_angles = self.north_heading - candidate_waypoints_angles_deg

            # Computing gps position of the waypoints
            candidate_gps = self.compute_gps(self.current_gps, candidate_waypoints_distances, waypoints_angles)
            
            # Convert the GPS psoition of waypoints to px coordinates
            wp_px = self.map_reader.lat_lon_to_pixel(candidate_gps[:, 0], candidate_gps[:, 1])

            # run the Global planner inference and compute probabilites for the waypoints
            self.global_planner.predict_probabilities(self.current_position, self.goal_position)
            
            # start_time = time.time()
            # waypoint_probs = self.global_planner.calculate_probs(wp_px,
            #                                                      self.current_position,
            #                                                                  self.north_heading)
            # print(f"time taken to compute probabilities: {time.time()-start_time} seconds")
            
            for i in range(NUM_SAMPLED_WAYPOINTS):                
                
                if tuple(candidate_gps[i]) not in self.wp_gps:                
                    self.wp_gps.add(tuple(candidate_gps[i]))
                    
                    # w = SigmaWaypoints(wp_node_id=self.node_id,
                    #                    node_distance=0,
                    #                    wp_distance=candidate_waypoints_distances[i],
                    #                    wp_gps=candidate_gps[i],
                    #                    wp_prob=waypoint_probs[i],
                    #                    wp_to_goal_distance=self.distance_from_gps(candidate_gps[i], self.goal_gps)
                    #                   )
                    
                    w = SigmaWaypoints(wp_node_id=self.node_id,
                                       node_distance=0,
                                       wp_distance=candidate_waypoints_distances[i],
                                       wp_gps=candidate_gps[i],
                                       wp_prob=None,
                                       wp_to_goal_distance=self.distance_from_gps(candidate_gps[i], self.goal_gps)
                                      )
                    
                    self.sigma = self.update_priority_queue(new_wp=w)
            
            # get the best wp from sigma
            _,_, selected_wp = self.sigma.get()
            self.wp_gps.discard(tuple(selected_wp.wp_gps))

            # get the path to the best wp from the graph
            path_to_selected_wp = nx.shortest_path(G=self.graph,
                                                   source=self.node_id,
                                                   target=selected_wp.wp_node_id,
                                                   weight='distance')
            print(f"Path to the selected wp in graph:\n{path_to_selected_wp}")

            cropped_map_img = self.draw_cropped_map()
            self.viz_current_state(cropped_map_img, wp_px, path_to_selected_wp, selected_wp)
            
            print("-------------------------------------------------------------------------------")
            

if __name__ == "__main__":
    rospy.init_node("Graph Search Navigation Simulator", log_level=rospy.INFO)
    node = Visualizer()    
    rospy.spin()