# Physical A*

This repo contains the code to run the physical A* algorithm as demonstrated by the [`ViKiNG`](https://arxiv.org/abs/2202.11271) paper.   

To run the code, a working Robot Operating System (ROS) environment is required. A local installation of ROS is recommended which can be done by following this link: [`ROS desktop installation`](http://wiki.ros.org/noetic/Installation/Ubuntu)   

### Building
To build the ros workspace and getting started, the first thing would be to create a catkin workspace and source all source code.   

```
mkdir -p robot_ws/src
cd robot_ws/src
git clone https://github.com/AnishShr/physical_astar.git
catkin build
```

### Getting necessary files
In order to run one of the simulations (raster map simulation), it needs the model file which is not included in the repo beacuse of it's large size. This model can be downloaded from: [`model`](https://www.dropbox.com/scl/fi/0l1j6y96w8elcey2vfb30/distance_segment.onnx?rlkey=7sdzxfwtgd51h889a7voumsbe&st=fpz9h1py&dl=0)   

Once downloaded, copy the file to: `robot_ws/src/physical_astar/data/onnx_models`. You might need to create the `onnx_models` directory inside the `data` folder.

### Environment
As the algorithm implementation uses some very specific packages, it is recommended to create a dedicated conda environment.   
To create this conda environment, please go through the following steps:   
```
conda install mamba -c conda-forge
conda create -n physical-astar python=3.9.16
conda activate physical-astar
mamba install pytorch-lightning tensorboard pandas jupyter matplotlib pytransform3d transformations h5py utm onnx opencv moviepy -c conda-forge
mamba install pyyaml rasterio pyproj
pip install onnxruntime-gpu==1.16.0
pip install cvxpy
```

After creating the environment, please source the catkin workspace:
```
cd robot_ws
source devel/setup.bash
```

### Installing required ros packages and dependencies
From the base of the catkin_ws (or in this case robot_ws), run the following commands:   
```
rosdep install --from-paths src --ignore-src -r -y
```

### Running the nodes

One of the simulation environment in which the algorithm is tested in Gazebo which uses [`Jackal Simulation`](https://docs.clearpathrobotics.com/docs/ros1noetic/robots/outdoor_robots/jackal/tutorials_jackal#simulating-jackal). In order to run these simulation nodes, please make sure you have all required packages. You can install these packages by going though the [`Jackal installation`](https://docs.clearpathrobotics.com/docs/ros1noetic/robots/outdoor_robots/jackal/tutorials_jackal#jackal-software-setup) guidelines
   
Once the Jackal software setup has been complete, you can move forward with the Gazebo simulation with Jackal.   
   
Open a new terminal window and run the Jackal simulation nodes. P.S.: These nodes have some dependency issues while running with the conda environment, so it is highly recommended to run the following commands in a local environment (**not with the conda environment**):
```
roslaunch physical_astar jackal_sim.launch
```

This should launch a gazebo world with Jackal launched in it, and a RViz window of the same simulation environment.

![jackal_sim_gazebo](https://github.com/AnishShr/physical_astar/assets/62991158/9aeb5637-0168-458c-b510-a31acece7f1d)

![jackal_sim_rviz](https://github.com/AnishShr/physical_astar/assets/62991158/baab8223-77b8-44db-aa40-f1434347389f)


In another terminal (**where you have activated the conda environment**), run the following command:
```
rosrun physical_astar jackal_sim_gps_cbf.py
```

You should see the Jackal initially driving in a straight path as it is calibrating it's initial GPS heading, and then start to create a graph and implement the physcial A* algorithm.

If you are able to run this, then you are all set up. This is just a *getting started* instruction. Follow along to run the simulation nodes for both: gazebo simulation and raster map simualiton in various scenarios.    

#### Gazebo simulation in various scenarios

The launch file which launches the gazebo simulaiton environment and RViz visualization is inside the launh directory `.../robot_ws/src/physical_astar/launch/jackalsim.launch`.   
In this launch file, the initial posiiton where the jackal will be spawned is configurable. Depending on various scenarios, these initial posiiton can be configured.   

![jackal_sim](https://github.com/AnishShr/physical_astar/assets/62991158/16fe9864-029a-413c-b3fb-6dab8bb3b4d3)


##### Easy Scenario

To choose one of the easy scnearios, uncomment the block of code corresponding to that scenario (and comment others) and run the following commands (**from the local terminal- not conda environment**)   
```
roslaunch physical_astar jackal_sim.launch nav_type_easy:=true
```

##### Difficult Scenario
To choose the scneario, run the following commands (**from the local terminal- not conda environment**)   
```
roslaunch physical_astar jackal_sim.launch
```

Once the gazebo and rviz window appear, run the following command from the **terminal where conda environment is activated**:
```
rosrun physical_astar jackal_sim_gps_cbf.py
```

##### Naive algorithm
In order to run the naive algorithm, run the following in place of the physical a* node:
```
rosrun physical_astar jackal_sim_no_graph.py
```

For either case, i.e., naive or physical a* algorithm, the terminal window in which the node is running, the number of robot time steps is always logged as the robot published velocity after initial GPS heading calibration.


##### Odometry based physcial A*
To run the physical A* with odometry coordinates, rather than GPS coordinates, run the following command:   
```
rosrun physical_astar jackal_sim_odom_cbf.py
```

*Regardless of which approach is run, please make sure that the terminal window with local environment (non-conda environment) should launch gazebo and rviz.*


#### Raster map simulation in various scenarios

The launch file for lauching different scenarios for raster map simulation is inside the launch directory `.../robot_Ws/src/physical_astar/launch/simulator.launch`.   

The initial UTM offsets for each scenario is set. Please uncomment the block of code for the corresponding scenario (and comment others) to run the raster map simulation for that scenario.   

![raster_sim](https://github.com/AnishShr/physical_astar/assets/62991158/546ff578-b1cb-4777-8018-b84955835ab6)
   

To run the raster map simualtion, from the terminal window with conda environment activated, please run the following command:   
```
roslaunch physical_astar simualator.launch
```