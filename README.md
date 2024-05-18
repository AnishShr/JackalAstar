# Physical A*

This repo contains the code to run the physical A* algorithm as demonstrated by the [`ViKiNG`](https://arxiv.org/abs/2202.11271) paper.   

To run the code, a working Robot Operating System (ROS) environment is required. A local installation of ROS is recommended which can be done by following this link: [`ROS desktop installation`](http://wiki.ros.org/noetic/Installation/Ubuntu)   

### Building
To build the ros workspace and getting started, the first thing would be to create a catkin workspace and source all source code.   

```
mkdir -p robot_ws/src
git clone https://github.com/AnishShr/physical_astar.git
catkin build
```

### Environment
As the algorithm implementation uses some very specific packages, it is recommended to create a dedicated conda environment.   
To create this conda environment, please go through the following steps:   
```
conda install mamba -c conda-forge
conda create -n physical-astar python=3.9.16
mamba install pytorch-lightning tensorboard pandas jupyter matplotlib pytransform3d transformations h5py utm onnx opencv moviepy -c conda-forge
mamba install pyyaml rasterio pyproj
pip install onnxruntime-gpu==1.16.0
pip install cvxpy
```

After creating the environment, activate the environment and source the catkin workspace:   
```
conda activate physical-astar
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

[Images of Gazebo world and RViz]

In another terminal (**where you have activated the conda environment**), run the following command:
```
rosrun physical_astar jackal_sim_gps_cbf.py
```

You should see the Jackal initially driving in a straight path as it is calibrating it's initial GPS heading, and then start to create a graph and implement the physcial A* algorithm.


#### Gazebo simulation in various scenarios

#### Raster map simulation in various scenarios
