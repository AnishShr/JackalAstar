from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

setup_args = generate_distutils_setup(
    scripts=['nodes/deadman_switch.py',
             'nodes/simulator.py',
             'nodes/simulator_naive.py',
             'nodes/jackal_diagnosis_node.py',
             'nodes/laserscan_downsample.py',
             'nodes/laserscan_manipulate.py',
             'nodes/laserscan_sample.py',
             'nodes/jackal_sim_no_graph.py',
             'nodes/jackal_sim_gps_cbf.py',
             'nodes/jackal_sim_odom_cbf.py',
             'nodes/jackal_a*.py'
             ],
    packages=['global_planner'],
    package_dir={'': 'src'},
    requires= ['rospy']
)

setup(**setup_args)
