import numpy as np
import os
import sys
import time
import mujoco
import pinocchio as pin
import cv2
import pyspacemouse
from pathlib import Path


from locompc.plan.manipulation import ReachGoal
from locompc.utils import load_yaml_file
from locompc.sim.mujoco import MjSim, MjSimCmd
from locompc.utils import load_yaml_file
from locompc.utils import CustomLogger, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT
logger = CustomLogger(__name__, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT).logger

from KinovaPy.controller import KinovaMPC
from KinovaPy.interface import KinovaHardwareInterface
from KinovaPy.model import PinocchioModel
from KinovaPy import plot
from KinovaPy import SCENE_PATH

from KinovaPy.utils.joy import SpaceMouseExpert
from KinovaPy.gim4305 import Gim4305

spacemouse0 = SpaceMouseExpert()
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "")
config_dir = os.path.join(CONFIG_PATH, 'kinova_float.yml')
config = load_yaml_file(config_dir)

QUIT = "q" in sys.argv
STAY = "stay" in sys.argv
REAL = "real" in sys.argv
WITHPLOT = "plot" in sys.argv 
SAVEDATA = "savedata" in sys.argv


# Read YAML config file
config_dir = os.path.join(CONFIG_PATH, 'kinova_float.yml')
config = load_yaml_file(config_dir)


### MPC ###
# Load Pinocchio model
pin_model = PinocchioModel(frame_names=config['endEffectorFrameName'])
rmodel = pin_model.rmodel
rdata = pin_model.rdata
nq = pin_model.nq
nv = pin_model.nv
nu = pin_model.nu

# Load controller
if not REAL:
  mode = 'sim'
else:
  mode = 'real'
if WITHPLOT or SAVEDATA:
  record = True
else:
  record = False
controller = KinovaMPC(rmodel, rdata, config, planner=None, record=record, mode=mode)

# Low-level control & safety
kp = np.asarray(config['kp_scale'])*np.asarray(config['kp_ref'])
kd = np.asarray(config['kd_scale'])*np.asarray(config['kd_ref'])
umin = -0.99*np.asarray(controller.mpc.oc.ctrlLimit)
umax = 0.99*np.asarray(controller.mpc.oc.ctrlLimit)
qmin = 0.99*np.asarray(controller.mpc.oc.stateLowerLimit)
qmax = 0.99*np.asarray(controller.mpc.oc.stateUpperLimit)
dqmin = -0.99*np.asarray(controller.mpc.oc.velocityLimit)
dqmax = 0.99*np.asarray(controller.mpc.oc.velocityLimit)


### Simulation or Real ###
if not REAL:
  # Load Mujoco model
  xml_name = 'scene_kinova.xml'
  xml_path = os.path.join(SCENE_PATH, xml_name)
  model = mujoco.MjModel.from_xml_path(xml_path)
  data = mujoco.MjData(model)
  ctrl_sim_ratio = round(config['dt_mpc']/config['dt_sim'])
  robot = MjSim(model, config, u0=config['u0'], floatingbase=False)
else:
  from KinovaPy import utilities
  args = utilities.parseConnectionArguments()
  tcp_connection = utilities.DeviceConnection.createTcpConnection(args)
  udp_connection = utilities.DeviceConnection.createUdpConnection(args)
  router = tcp_connection.__enter__()
  router_real_time = udp_connection.__enter__()
  robot = KinovaHardwareInterface(router, router_real_time, torque_limits=umax)
  robot.stop_command_stream()
  time.sleep(1.0)
  robot.move_to_home(q0=np.asarray(config['q0_real']))
time.sleep(1.0)

# Initialization
if not REAL:
  q0 = np.asarray(config['q0_mj'])
  v0 = np.asarray(config['dq0'])
  u0 = np.asarray(config['u0'])
else:
  q0, v0, u0 = controller.get_states(robot)
x0 = np.concatenate([q0, v0])
logger.debug(f'q0 = {q0}')
logger.debug(f'u0 = {u0}')


### Manipulation plan ###
# Starting pose
pin_model.update(q0, v0)
info0 = pin_model.get_info()
frameIds = list(pin_model.frame_ids.values())
for name in pin_model.frame_names:
  p0 = np.concatenate([info0['frame_poses'][name]['position'], info0['frame_poses'][name]['rpy']])
  logger.debug(f"Initial end-effector pose: {p0}")
pose0 = pin.SE3(info0['frame_poses'][pin_model.ee_frame_name]['rotation'],
                info0['frame_poses'][pin_model.ee_frame_name]['position'])



### Warmstart ###
controller.warmstart(robot, None, None)

### Start ###
run_time = config['sim_time']
start = input("\nPress [ENTER] to start...")
print("\n---------------------------- Experiment running ----------------------------")

start_time = time.perf_counter()
# os.makedirs("data/camera_data/run_11", exist_ok=True) #*************CHANGE NUMBER EVERY RUN

# Send initial command
q_des = q0.copy()
dq_des = np.zeros(nq)
tau_des = pin.computeGeneralizedGravity(rmodel, rdata, q_des)
if not REAL:
  cmd = MjSimCmd(tau_des, q_des, dq_des, kp, kd)
  robot.start()
  robot.set_cmd(cmd)
else:
  robot.start_command_stream(control_mode="TORQUE")
  controller.send_command(robot, tau_des, q_des, dq_des, kp=kp, kd=kd)



pose = pose0
dt = config['dt']

#Main loop
frame_count = 0
spacemouse0 = SpaceMouseExpert()
motor = Gim4305()
print("Enabling motor...")
motor.enable()
time.sleep(0.1)
motor.set_zero()
device = pyspacemouse.open()

while time.perf_counter()-start_time < run_time:

  state = device.read()


  tic = time.perf_counter()
  if config['sync'] and not REAL:
    step_number = robot.step_counter

  # Define task
  
  space_action, space_buttons = spacemouse0.get_action()
  goals = [pose]
  delta_trans = space_action[:3]*config['dt']
  digital_cmd = [space_buttons[0], space_buttons[1]]
  pose.translation += delta_trans

  jk = 0
  jl = 0.005

  direct = digital_cmd[0] #left button when wire faces away from you
  direct2 = digital_cmd[1] #right button when wire faces away from you

  if direct == 1:
    motor.command(place=0, zoom=6, jk=jk, jl=jl, umph=0.05)
  elif direct2 == 1:
    motor.command(place=0, zoom=-6, jk=jk, jl=jl, umph=-0.05)
  else:
    motor.command(place=0.0, zoom=0.0,jk=jk, jl=jl, umph=0.0)
  

  
    q_now, dq_now, _ = controller.get_states(robot)
    pin_model.update(q_now, dq_now)
    pose_current = pin_model.get_info()['frame_poses'][pin_model.ee_frame_name]['position']
    print(pose_current)
    print(pose.translation)
    print(state.z)
    manipPlan = [ReachGoal(frameIds, goals)]*(config['N_h']+1)
      
  # MPC
  u_des, x_des = controller.update(robot, manipPlan=manipPlan)
  tau_des = np.clip(u_des, umin, umax)
  q_des = np.clip(x_des[:len(q0)], qmin, qmax)
  dq_des = np.clip(x_des[len(q0):], dqmin, dqmax)



  # Send joint torques, joint positions and velocities to robot
  if not REAL:
    cmd = MjSimCmd(tau_des, q_des, dq_des, kp, kd)
    robot.set_cmd(cmd)
  else:
    controller.send_command(robot, tau_des, q_des, dq_des, kp=kp, kd=kd)

  # Logger
  if config['verbose']:
    logger.debug(f'Solve time = {controller.mpc.solve_time:.4f}s')

  while time.perf_counter() - tic < controller.dt_mpc:
    time.sleep(0.0001)
  freq = 1./(time.perf_counter() - tic)
  print(freq)


if not REAL:
  robot.close()


# Trim data
if record:
  xs = controller.xs[:controller.i]
  us = controller.us[:controller.i]
  x_des = controller.x_des[:controller.i]
  u_des = controller.u_des[:controller.i]
  x_all = controller.x_all[:controller.i]
  u_all = controller.u_all[:controller.i]
  sol_stats = controller.sol_stats[:controller.i]


# Final state
q_final, v_final, u_final = controller.get_states(robot)
# Forward kinematics to get x,y,z,roll,pitch,yaw
pin_model.update(q_final, v_final)
frameId = pin_model.ee_frame_id  # end-effector frame id, used below for plotting
ee_pose_final = pin_model.get_info()['frame_poses'][pin_model.ee_frame_name]
p_final = np.concatenate([ee_pose_final['position'], ee_pose_final['rpy']])
print(f"Final configuration: {q_final}")
print(f"Final end-effector pose: {p_final}")


reach_pose = pin.SE3(ee_pose_final['rotation'], ee_pose_final['position'])


# Stop robot
if REAL:
  robot.stop_command_stream()
  udp_connection.__exit__(None, None, None)
  tcp_connection.__exit__(None, None, None)
  cv2.destroyAllWindows()
  motor.close()


# Info
print("\n--------------------------- Experiment finished ---------------------------")
logger.info(f"Experiment finished after {(time.perf_counter()-start_time):.3f}s.")
if record:
  logger.info(f'[MPC] max sol time: {np.max(sol_stats[:,0]):.4f}s')
  logger.info(f'[MPC] mean sol time: {np.mean(sol_stats[:,0]):.4f}s')


# Save data
if SAVEDATA:
  import pandas as pd
  data = np.concatenate([xs, us, x_des, u_des, sol_stats], axis=1)
  df = pd.DataFrame(data)
  df.to_csv('data/training_data/run_011.csv')


# Plot the MPC solution
if WITHPLOT:
  plot.plotFrameTrajectory(rmodel, xs, frameId, None, label='End-effector')
  plot.plotJointTrajectory(xs, us, controller.dt_mpc, figTitle='Joint position & torque trajectory')
  plot.plotSolutionVsActual(x_des, u_des, xs, us, controller.dt_mpc, joint_id=5)
