import numpy as np

import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Imu, NavSatFix, LaserScan
from nav_msgs.msg import Odometry
import tf.transformations
from visualization_msgs.msg import Marker, MarkerArray
import tf2_ros
import tf

from shapely.geometry import Point, LineString
from queue import PriorityQueue
import networkx as nx
import cvxpy as cp

DISTANCE_TO_GOAL_THRESHOLD = 0.85
DISTANCE_TO_TARGET_GPS = 0.5
BUFFER_DISTANCE = 0.4 # 0.275

class SigmaWaypoints:
    def __init__(self,
                 wp_node_id,                  
                 wp_gps, 
                 wp_odom_pos, 
                 wp_to_goal_distance):
        
        self.wp_node_id = wp_node_id                                # waypoint's node id in the graph
        self.wp_gps = wp_gps                                        # waypoint's gps position (lat, lon)
        self.wp_odom_pos = wp_odom_pos                              # waypoint's position in odom frame
        self.wp_to_goal_distance = wp_to_goal_distance              # euclidean distance between the wapypoint and the goal


class JackalSim:

    def __init__(self):
        
        self.current_heading = None
        self.heading_calibrated = False

        self.prev_pose = None
        self.current_pose = None

        self.init_gps = None
        self.current_gps = None
        
        self.current_quat = None
        self.prev_quat = None

        # robot states for state machine logic
        # states:
        # 0 --> calibrate initial heading
        # 1 --> propose waypoints
        # 2 --> follow best waypoint
        # 3 --> reached best wp
        self.state = 0

        self.collision_free_waypoints = None
        self.valid_scans = None

        self.obs = None
        self.yaw = None

        self.obs_D = 0.5
        self.k = 0.8
        self.alpha = 0.35

        self.best_wp = None
        self.best_wp_gps = None
        self.current_goal_id = 0

        self.path_through_graph = None
        self.parent = None
        self.gps_path = []
        self.odom_pose_path = []
        self.path_to_selected_wp = None
        
        self.rotate = False
        self.init_rotate_heading = None

        self.linear_vel = 0
        self.angualr_vel = 0

        # self.goal_gps = np.array([49.90004467081109, 8.900069470995431])
        # self.goal_gps = np.array([49.90004566787815, 8.899944535761064])
        # self.goal_gps = np.array([49.900036217862485, 8.89990357085164])

        # self.goal_gps = np.array([49.89999999659677, 8.900080587590345])
        self.goal_gps = np.array([49.90006135821165, 8.900067923531012])
        self.goal_pos = None

        self.num_nodes = 0
        self.node_id = 0
        self.prev_node_id = 0

        self.selected_wp = None
        self.all_wps = []

        # Create an empty graph
        self.graph = nx.Graph()
        # Create an empty open set Sigma
        self.sigma = PriorityQueue()

        # TF buffer
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        self.tf_laser_to_base_link = self.create_tf_matrix(source_frame='front_laser',
                                                           target_frame='base_link')
        self.tf_navsat_to_odom = self.create_tf_matrix(source_frame='navsat_link',
                                                       target_frame='odom')
        self.tf_base_link_to_odom = None

        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.marker_pub = rospy.Publisher('/visualization_marker', MarkerArray, queue_size=1, latch=True)

        rospy.Subscriber('/imu/data', Imu, self.imu_callback, queue_size=1)
        rospy.Subscriber('/odometry/filtered', Odometry, self.odom_callback, queue_size=1)
        rospy.Subscriber('/navsat/fix', NavSatFix, self.gps_callback, queue_size=1)
        rospy.Subscriber('/scan/filtered', LaserScan, self.laser_callback, queue_size=1)
        rospy.Subscriber('/front/scan', LaserScan, self.raw_laser_callback, queue_size=1)

        rospy.Timer(rospy.Duration(1.0/20.0), self.timer_callback)
    
    
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
        Drive 1.0m forward and calculate the current heading w.r.t. Geographic North
        
        """
        
        linear_vel = 0.0

        if np.linalg.norm(self.current_pose - self.prev_pose) <= 1.0:
            linear_vel = 0.5
            rospy.loginfo_once(f"Initial heading being calibrated ...")
        else:
            rospy.loginfo_once("Reached 1.0m mark !!")
            final_gps = self.current_gps
            bearing_north = self.compute_heading_north(self.init_gps, final_gps)
            rospy.loginfo_once("Initial robot heading w.r.t. Geographic North calibrated")
            rospy.loginfo_once(f"Initial heading: {bearing_north} degrees (Clockwise) w.r.t. North")
            self.current_heading = bearing_north
            # self.heading_calibrated = True

            self.node_id = 0
            self.graph.add_node(self.node_id,
                                node_id=self.node_id,
                                node_gps=self.current_gps,
                                node_odom_pos=self.current_pose)
            print(f"Graph nodes: {self.graph.nodes.data()}")
            
            self.state = 1
            rospy.loginfo_once("Robot is in state 1 ..")
        
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

        return new_gps
    

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

        if d.shape[0] > 1:
            d = np.squeeze(d)
        return d
    

    def compute_cost(self, parent_node, wp):

        # wp_parent_node_id = wp.wp_parent_node_id
        wp_node_id = wp.wp_node_id
        local_cost = nx.shortest_path_length(G=self.graph, 
                                               source=parent_node, 
                                               target=wp_node_id,
                                               weight='distance')
        # wp_distance = wp.wp_distance
        wp_gps = wp.wp_gps
        wp_to_goal_distance = wp.wp_to_goal_distance
        
        # local_cost = wp_parent_node_distance + wp_distance
        # node_gps = self.graph.nodes[wp_node_id]['node_gps']
        
        # wp_distance = self.distance_from_gps(gps_origin=node_gps, gps_destination=wp_gps)
        # local_cost = wp_parent_node_distance + wp_distance
        global_cost = wp_to_goal_distance
        
        total_cost = local_cost + global_cost

        # total_cost = global_cost
        return total_cost
    

    def update_priority_queue(self, parent_node, new_wp):
        updated_queue = PriorityQueue()
        
        # Update the pre-existing sigma        
        while not self.sigma.empty():
            _, existing_node = self.sigma.get()
            final_cost = self.compute_cost(parent_node, existing_node)
            updated_queue.put((final_cost, existing_node))

        # Add new wp in sigma after computing the new cost
        final_cost = self.compute_cost(parent_node, new_wp)
        updated_queue.put((final_cost, new_wp))

        return updated_queue
    

    def calculate_edge_distance(self, origin_id, destination_id):

        origin = self.graph.nodes[origin_id]['node_odom_pos']
        destination = self.graph.nodes[destination_id]['node_odom_pos']
        

        dist = np.sqrt((destination[0]-origin[0])**2 + (destination[1]-origin[1])**2)
        return dist
    

    # def publish_markers(self, target_wp, goal_gps):
    def publish_markers(self, parent, path, goal_gps):
        
        # target_gps = target_wp.wp_gps
        markers=[]

        marker_id = 0
        for node in self.graph.nodes():

            x = self.graph.nodes[node]['node_odom_pos'][0]
            y = self.graph.nodes[node]['node_odom_pos'][1]

            node_marker = Marker()
            node_marker.header.frame_id = "odom"
            node_marker.type = Marker.SPHERE
            node_marker.action = Marker.ADD
            node_marker.scale.x = 0.1
            node_marker.scale.y = 0.1
            node_marker.scale.z = 0.1
            node_marker.color.a = 1.0
            node_marker.color.r = 0.0
            node_marker.color.g = 0.0
            node_marker.color.b = 1.0
            node_marker.pose.orientation.w = 1.0
            node_marker.pose.position = Point(x, y, 0)
            node_marker.id = marker_id
            markers.append(node_marker)
            marker_id += 1

        # edge_id = node_id
        for u, v in self.graph.edges():

            x_u = self.graph.nodes[u]['node_odom_pos'][0]
            y_u = self.graph.nodes[u]['node_odom_pos'][1]

            x_v = self.graph.nodes[v]['node_odom_pos'][0]
            y_v = self.graph.nodes[v]['node_odom_pos'][1]

            edge_marker = Marker()
            edge_marker.header.frame_id = "odom"
            edge_marker.type = Marker.LINE_LIST
            edge_marker.action = Marker.ADD
            edge_marker.scale.x = 0.03  # Line width
            edge_marker.color.a = 1.0
            edge_marker.color.r = 1.0
            edge_marker.color.g = 1.0
            edge_marker.color.b = 0.0
            
            edge_marker.id = marker_id
            edge_marker.pose.orientation.w = 1

            #start point
            start_point = Point(x_u, y_u, 0)
            edge_marker.points.append(start_point)

            # end point
            end_point = Point(x_v, y_v, 0)
            edge_marker.points.append(end_point)

            markers.append(edge_marker)
            marker_id +=1
    
        for node in path:

            x_node = self.graph.nodes[node]['node_odom_pos'][0]
            y_node = self.graph.nodes[node]['node_odom_pos'][1]

            node_marker = Marker()
            node_marker.header.frame_id = "odom"
            node_marker.type = Marker.SPHERE
            node_marker.action = Marker.ADD
            node_marker.scale.x = 0.2
            node_marker.scale.y = 0.2
            node_marker.scale.z = 0.2
            node_marker.color.a = 1.0
            node_marker.color.r = 1.0
            node_marker.color.g = 0.0
            node_marker.color.b = 0.0
            node_marker.pose.orientation.w = 1.0
            node_marker.pose.position = Point(x_node, y_node, 0)
            node_marker.id = marker_id
            markers.append(node_marker)
            marker_id += 1
        
        x_parent_node = self.graph.nodes[parent]['node_odom_pos'][0]
        y_parent_node = self.graph.nodes[parent]['node_odom_pos'][1]

        parent_node_marker = Marker()
        parent_node_marker.header.frame_id = "odom"
        parent_node_marker.type = Marker.SPHERE
        parent_node_marker.action = Marker.ADD
        parent_node_marker.scale.x = 0.2
        parent_node_marker.scale.y = 0.2
        parent_node_marker.scale.z = 0.2
        parent_node_marker.color.a = 1.0
        parent_node_marker.color.r = 1.0
        parent_node_marker.color.g = 1.0
        parent_node_marker.color.b = 1.0
        parent_node_marker.pose.orientation.w = 1.0
        parent_node_marker.pose.position = Point(x_parent_node, y_parent_node, 0)
        parent_node_marker.id = marker_id
        markers.append(parent_node_marker)
        marker_id += 1  

        d_goal = self.distance_from_gps(gps_origin=self.current_gps,
                                        gps_destination=goal_gps)
        d_goal = np.squeeze(d_goal)
        
        heading_goal = self.compute_heading_north(init_gps=self.current_gps,
                                                  final_gps=goal_gps)
        
        heading_goal_local = self.current_heading - heading_goal

        if heading_goal_local < -180:
            heading_goal_local += 360
        elif heading_goal_local > 180:
            heading_goal_local -= 360
        
        heading_goal_local_rad = np.deg2rad(heading_goal_local)

        x_goal = d_goal * np.cos(heading_goal_local_rad)
        y_goal = d_goal * np.sin(heading_goal_local_rad)

        marker_goal = Marker()
        marker_goal.header.frame_id = "navsat_link"
        marker_goal.type = Marker.SPHERE
        marker_goal.action = Marker.ADD
        marker_goal.id = marker_id
        marker_goal.scale.x = 0.2
        marker_goal.scale.y = 0.2
        marker_goal.scale.z = 0.2
        marker_goal.color.a = 1.0
        marker_goal.color.r = 1.0
        marker_goal.color.g = 1.0
        marker_goal.color.b = 0.0
        marker_goal.pose.orientation.w = 1.0
        marker_goal.pose.position = Point(x_goal, y_goal, 0.0)

        markers.append(marker_goal)

        marker_array = MarkerArray()
        marker_array.markers = markers

        self.marker_pub.publish(marker_array)


    # def follow_path(self, path_through_graph, gps_path):
    def follow_path(self, path_through_graph, odom_pose_path):

        if self.current_goal_id >= len(odom_pose_path):
            # Completed all gps points
            # update the state to 3
            self.current_goal_id = 0            
            self.node_id = path_through_graph[-1]

            self.gps_path = []
            self.odom_pose_path = []
            self.path_through_graph = None 

            self.state = 3

        else:

            odom_pose_path = np.array(odom_pose_path)
            # odom_pose_path = np.squeeze(odom_pose_path)
            odom_pose_path = odom_pose_path.reshape(-1, 2)
            
            tf_odom_to_base_link = self.create_tf_matrix(source_frame='odom',
                                                             target_frame='base_link')
            goals_in_base_link = self.transform_points(input_points=odom_pose_path,
                                                       tf_matrix=tf_odom_to_base_link)
            current_goal_in_base_link = goals_in_base_link[self.current_goal_id]

            # print("---------------------------------------")
            # print(odom_pose_path)
            # print(f"Current goal: {odom_pose_path[self.current_goal_id]}")

            # print(current_goal_in_base_link)
            # print("----------------------------------------")
            
            dist_from_target_wp = np.linalg.norm(current_goal_in_base_link)
            
            linear_vel = 0
            angular_vel = 0

            # if dist_from_target_gps < DISTANCE_TO_TARGET_GPS:
            if dist_from_target_wp < DISTANCE_TO_TARGET_GPS:
                self.node_id = path_through_graph[self.current_goal_id]
                self.current_goal_id += 1
            
            else:
                
                # linear_vel = dist_from_target_wp
                # angular_vel = np.arctan2(current_goal_in_base_link[1], current_goal_in_base_link[0])

                heading_target_wp = np.arctan2(current_goal_in_base_link[1], current_goal_in_base_link[0])
                heading_target_wp_deg = np.rad2deg(heading_target_wp)

                if -75 < heading_target_wp_deg < 75:
                    target_x = current_goal_in_base_link[0]
                    target_y = current_goal_in_base_link[1]

                    tf_base_link_to_odom = self.create_tf_matrix(source_frame='base_link',
                                                                 target_frame='odom')
                    target_odom = self.transform_points(input_points=current_goal_in_base_link.reshape(-1, 2),
                                                        tf_matrix=tf_base_link_to_odom)
                    target_odom = np.squeeze(target_odom)
                    
                    obs_pose_base_link = self.obs
                    obs_pose_odom = self.transform_points(input_points=obs_pose_base_link.reshape(-1, 2),
                                                     tf_matrix=tf_base_link_to_odom)
                    obs_pose_odom = np.squeeze(obs_pose_odom)
                    
                    current_pose = self.current_pose
                    yaw = self.yaw

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

                    if heading_target_wp_deg < 0:
                        angular_vel = -np.deg2rad(30)
                    else:
                        angular_vel = np.rad2deg(30)
                

            vel = Twist()
            vel.linear.x = np.clip(linear_vel, 0, 0.3)
            # vel.angular.z = np.clip(angular_vel, -np.deg2rad(45), np.deg2rad(45))
            vel.angular.z = angular_vel
            self.cmd_vel_pub.publish(vel)
            

    def waypoint_cost(self, waypoints_xy):
        waypoints_gps = self.compute_gps(current_gps=self.current_gps,
                                             waypoints=waypoints_xy)
        distance_waypoints_to_goal = self.distance_from_gps(gps_origin=waypoints_gps,
                                                                gps_destination=self.goal_gps)

        waypoints_global_cost = distance_waypoints_to_goal

        return waypoints_gps, waypoints_global_cost
        

    def gps_callback(self, msg):
        self.current_gps = np.array([msg.latitude, msg.longitude])


    def raw_laser_callback(self, msg):

        valid_xy = []
        for idx, range in enumerate(msg.ranges):

            if np.isinf(range):
                range = msg.range_max

            if range > 1.0:

                angle = msg.angle_min + idx * msg.angle_increment
                x = 0.5 * np.cos(angle)
                y = 0.5 * np.sin(angle)
                valid_xy.append([x, y])
        
        valid_xy = np.array(valid_xy)

        valid_xy = self.transform_points(input_points=valid_xy,
                                         tf_matrix=self.tf_laser_to_base_link)
        
        self.valid_scans = valid_xy 

    
    def laser_callback(self, msg):

        ranges = np.array(msg.ranges)
        angles = np.array([msg.angle_min + i * msg.angle_increment \
                           for i in range(len(msg.ranges)) ])

        valid_indices = ranges >= 3.0
        invalid_indices = ranges < 3.0

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
                wp_x = 1.75 * np.cos(valid_angle)
                wp_y = 1.75 * np.sin(valid_angle)
                collision_free_waypoints.append([wp_x, wp_y])
        
        collision_free_waypoints = np.array(collision_free_waypoints)
        if collision_free_waypoints.shape[0] == 0:
            # rospy.logwarn("No valid waypoints")
            self.waypoints_base_link = None
            return

        # print(collision_free_waypoints)
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


    def timer_callback(self, event=None):
        
        # linear_vel = self.linear_vel
        # angular_vel = self.angualr_vel

        if self.goal_gps is None:
            return

        node_id = self.node_id
        prev_node_id = self.prev_node_id

        waypoints_base_link = self.waypoints_base_link

        dist_from_final_goal = self.distance_from_gps(gps_origin=self.current_gps,
                                                      gps_destination=self.goal_gps)            

        if dist_from_final_goal <= DISTANCE_TO_GOAL_THRESHOLD:
            rospy.loginfo_once("Reached final goal")

            vel = Twist()
            self.cmd_vel_pub.publish(vel)
            self.goal_gps = None
            
            return
            

        if self.state == 0:
            self.heading_calibration()

        elif self.state == 1:
           # propose waypoints
           # For each waypoint, see if there are any nodes in the graph close to the current wp
           # If there is not, then add a node and an edge to the wp from current node
           # Add the new node to priority queue
           # Else replace the current wp with the closest node and add an edge from current node to that pre-existing node
           # get the path from current node to the best node from priority queue along the graph
           # set the gps path for the desired path
            parent_node = self.node_id
            if waypoints_base_link is None:
                    # rotate clockwise
                    vel = Twist()
                    linear_vel = 0
                    angular_vel = np.deg2rad(30)

                    vel.linear.x = linear_vel
                    vel.angular.z = angular_vel

                    self.cmd_vel_pub.publish(vel)

            else:

                self.tf_base_link_to_odom = self.create_tf_matrix(source_frame='base_link',
                                                                  target_frame='odom')

                waypoints_in_odom_frame = self.transform_points(input_points=self.waypoints_base_link,
                                                                tf_matrix=self.tf_base_link_to_odom)

                waypoints_gps, waypoints_global_cost = self.waypoint_cost(waypoints_xy=waypoints_base_link)

                print(waypoints_base_link.shape[0])
                print(len(waypoints_gps))
                print(self.node_id)

                for i in range(waypoints_base_link.shape[0]):
                    # self.prev_node_id = self.node_id
                    self.num_nodes = self.graph.number_of_nodes()
                    self.node_id = self.num_nodes

                    # print(f"parent node: {parent_node}")
                    # print(f"current node: {self.node_id}")
                    # print(f"Total nodes: {self.num_nodes}")

                    nodes_to_merge_with = []
                    for node in self.graph.nodes():
                        
                        node_coords = self.graph.nodes[node]['node_odom_pos']
                        waypoint_coords = waypoints_in_odom_frame[i]
                        
                        dist = np.sqrt((waypoint_coords[0]-node_coords[0])**2 + (waypoint_coords[1]-node_coords[1])**2)
                        # if self.distance_from_gps(gps_origin=waypoints_gps[i],
                        #                           gps_destination=self.graph.nodes[node]['node_gps']) < 0.75:
                        if dist < 1.0:
                            nodes_to_merge_with.append(node)

                    print(f"nodes to merge: {nodes_to_merge_with}")
                    

                    if len(nodes_to_merge_with) == 0:
                        # add node and edge
                        self.graph.add_node(self.node_id,
                                            node_id=self.node_id,
                                            node_gps=waypoints_gps[i],
                                            node_odom_pos=waypoints_in_odom_frame[i])
                        print(f"Added node {self.node_id} in graph")                        

                        self.graph.add_edge(parent_node, self.node_id,
                                            distance=self.calculate_edge_distance(parent_node, self.node_id))
                        print(f"Added edge between {self.node_id} and {parent_node} in graph")    

                        wp = SigmaWaypoints(wp_node_id=self.node_id,
                                            wp_gps=waypoints_gps[i],
                                            wp_odom_pos=waypoints_in_odom_frame[i],
                                            wp_to_goal_distance=waypoints_global_cost[i])
                        
                        print(f"added node {self.node_id} in priority queue")
                        self.sigma = self.update_priority_queue(parent_node=parent_node, new_wp=wp)                    

                    else:
                        for node in nodes_to_merge_with:
                            # if self.graph.has_edge(self.node_id, node):
                            #     self.graph.edges[parent_node, node]['distance'] = self.calculate_edge_distance(origin_id=parent_node,
                            #                                                                                     destination_id=node)
                            # else:
                            #     self.graph.add_edge(parent_node, node, distance=self.calculate_edge_distance(origin_id=parent_node,
                            #                                                                                 destination_id=node))                            
                            self.graph.add_edge(parent_node, node, distance=self.calculate_edge_distance(origin_id=parent_node,
                                                                                                         destination_id=node))
                            print(f"Updated edge between {parent_node} and {node} in graph")
                    
                    # print(f"Graph nodes: \n{self.graph.nodes.data()}")
                    # print(f"Graph edges: \n{self.graph.edges.data()}")
                
                for node in self.graph.nodes():
                    for another_node in self.graph.nodes():
                        if node != another_node:
                            # dist = self.distance_from_gps(gps_origin=self.graph.nodes[node]['node_gps'],
                            #                               gps_destination=self.graph.nodes[another_node]['node_gps'])
                            dist = self.calculate_edge_distance(origin_id=node, destination_id=another_node)
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
                    self.odom_pose_path.append(self.graph.nodes[node_id]['node_odom_pos'])
                    self.gps_path.append(self.graph.nodes[node_id]['node_gps'])
                
                # print(self.odom_pose_path)

                tf_odom_to_base_link = self.create_tf_matrix(source_frame='odom',
                                                           target_frame='base_link')

                odom_pose_path = np.array(self.odom_pose_path)
                odom_pose_path = np.squeeze(odom_pose_path)
                odom_pose_path = odom_pose_path.reshape(-1, 2)

                print(odom_pose_path)

                # tf_odom_to_base_link = np.invert(self.tf_base_link_to_odom)
                base_link_path = self.transform_points(input_points=odom_pose_path, tf_matrix=tf_odom_to_base_link)

                print(base_link_path)

                self.state = 2
                

        
        elif self.state == 2:
            # get the gps path along the graph to traverse
            # follow the path
            path_through_graph = self.path_through_graph
            gps_path = self.gps_path
            odom_pose_path = self.odom_pose_path

            self.publish_markers(parent=self.parent, path=path_through_graph, goal_gps=self.goal_gps)
            # self.follow_path(path_through_graph, gps_path)
            self.follow_path(path_through_graph, odom_pose_path)

            # dist_to_goal_gps = self.distance_from_gps(gps_origin=self.current_gps,
            #                                           gps_destination=self.goal_gps)
            # if dist_to_goal_gps < DISTANCE_TO_GOAL_THRESHOLD:
            #     self.state = 3
            # else:
            #     self.state = 1


        elif self.state == 3:
            # check if the robot reached final goal
            # if the robot reached final goal, then stop everything
            # else set the state to state==1 to propose the waypoints from current node
            
            dist_from_goal = self.distance_from_gps(gps_origin=self.current_gps,
                                                    gps_destination=self.goal_gps)
            if dist_from_goal > DISTANCE_TO_GOAL_THRESHOLD:
                self.state = 1

            else:
                rospy.loginfo_once("Reached final goal")

                vel = Twist()
                self.cmd_vel_pub.publish(vel)
                self.goal_gps = None
        

    def run(self):
        rospy.spin()

if __name__ == "__main__":

    rospy.init_node("jackal_sim", anonymous=True)
    node = JackalSim()
    node.run()