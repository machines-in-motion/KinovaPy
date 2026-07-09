import numpy as np
import os
import sys
import time
import mujoco
import pinocchio as pin

from locompc.utils import load_yaml_file
from KinovaPy.interface import KinovaHardwareInterface
from KinovaPy import plot
from KinovaPy import MESHES_PATH, URDF_PATH, SCENE_PATH
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "")


# Read YAML config file
config_dir = os.path.join(CONFIG_PATH, 'kinova_move2goal.yml')
config = load_yaml_file(config_dir)


### MPC ###
# Load Pinocchio model
urdf_name = 'GEN3-7DOF-VISION_ARM_URDF_V12.urdf'
urdf_path = os.path.join(URDF_PATH, urdf_name)
meshes_dir = MESHES_PATH
pin_robot = pin.RobotWrapper.BuildFromURDF(urdf_path, meshes_dir, root_joint=None)
rmodel = pin_robot.model
cmodel = pin_robot.collision_model
vmodel = pin_robot.visual_model
rdata = rmodel.createData()
nq = rmodel.nq
nv = rmodel.nv
nu = rmodel.nv

# PD control gains
kp = np.asarray(config['kp_scale'])*np.asarray(config['kp_ref'])
kd = np.asarray(config['kd_scale'])*np.asarray(config['kd_ref'])\

# Extract robot limits
velocity_limit = np.asarray(rmodel.velocityLimit)
torque_limit = np.asarray(rmodel.effortLimit)
# scale down the limits
velocity_limit = 0.3*velocity_limit
torque_limit = 0.5*torque_limit
# print
print(f"velocity_limit = {velocity_limit} rad/s")
print(f"torque_limit = {torque_limit} Nm")


### Start ###
run_time = config['sim_time']
start = input("\nPress [ENTER] to start...")
print("\n---------------------------- Experiment running ----------------------------")

### Move to goal ###
from KinovaPy import utilities
args = utilities.parseConnectionArguments()
tcp_connection = utilities.DeviceConnection.createTcpConnection(args)
udp_connection = utilities.DeviceConnection.createUdpConnection(args)
router = tcp_connection.__enter__()
router_real_time = udp_connection.__enter__()
robot = KinovaHardwareInterface(router, router_real_time, torque_limits=torque_limit)
robot.stop_command_stream()
time.sleep(1.0)
robot.move_to_home(q0=np.asarray(config['q0_real']))
time.sleep(1.0)

# Initialization
state = robot.get_robot_states()
q0 = np.asarray(state['position'], dtype=float)
v0 = np.asarray(state['velocity'], dtype=float)
u0 = np.asarray(state['torque'], dtype=float)
print(f'q0 = {q0}')
print(f'v0 = {v0}')
print(f'u0 = {u0}')


# Stop the robot
robot.stop_command_stream()
udp_connection.__exit__(None, None, None)
tcp_connection.__exit__(None, None, None)


# Info
print("\n--------------------------- Experiment finished ---------------------------")
