import numpy as np
import os
import sys
import time
import mujoco
import pinocchio as pin

from locompc.plan.manipulation import ReachGoal
from locompc.sim.mujoco import MjSim, MjSimCmd
from locompc.utils import load_yaml_file
from locompc.utils import CustomLogger, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT
logger = CustomLogger(__name__, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT).logger

from KinovaPy.controller import KinovaMPC
from KinovaPy.interface import KinovaHardwareInterface
from KinovaPy import plot
from KinovaPy import MESHES_PATH, URDF_PATH, SCENE_PATH
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "")

STAY = "stay" in sys.argv
REAL = "real" in sys.argv
WITHPLOT = "plot" in sys.argv 
SAVEDATA = "savedata" in sys.argv


# Read YAML config file
config_dir = os.path.join(CONFIG_PATH, 'kinova_reach.yml')
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
pin.forwardKinematics(rmodel, rdata, q0)
pin.updateFramePlacements(rmodel, rdata)
frameNames = config['endEffectorFrameName']
frameIds = []
for name in frameNames:
  frameId = rmodel.getFrameId(name)  # end-effector frame id
  frameIds += [frameId]
  pose0 = rdata.oMf[frameId].copy()
  pos0 = rdata.oMf[frameId].translation.copy()
  rot0 = rdata.oMf[frameId].rotation.copy()
  rpy0 = pin.rpy.matrixToRpy(rot0)
  p0 = np.concatenate([pos0, rpy0])
  logger.debug(f"Initial end-effector pose: {p0}")


# Define reach pose for the end-effector
reach_pos = np.array([0.3, 0.3, 0.2])
reach_rot = pin.rpy.rpyToMatrix(np.array([np.pi, 0., np.pi/2.]))
reach_pose = pin.SE3(reach_rot, reach_pos)


### Warmstart ###
# Initial goal poses
goals = [reach_pose]
manipPlan = [ReachGoal(frameIds, goals)]*(config['N_h']+1)
controller.warmstart(robot, manipPlan, None)


### Start ###
run_time = config['sim_time']
start = input("\nPress [ENTER] to start...")
print("\n---------------------------- Experiment running ----------------------------")

start_time = time.perf_counter()

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

# Main loop
while time.perf_counter()-start_time < run_time:
  tic = time.perf_counter()
  if config['sync'] and not REAL:
    step_number = robot.step_counter

  # Define task
  goals = [reach_pose]
  manipPlan = [ReachGoal(frameIds, goals)]*(config['N_h']+1)

  # MPC
  u_des, x_des = controller.update(robot, manipPlan=manipPlan)
  tau_des = np.clip(u_des, umin, umax)
  q_des = np.clip(x_des[:len(q0)], qmin, qmax)
  dq_des = np.clip(x_des[len(q0):], dqmin, dqmax)
  # q_des += np.array([0., 0., 0., 0., 0., 0., 0.])
  # dq_des = np.zeros(nq)
  # tau_des = pin.computeGeneralizedGravity(rmodel, rdata, q_des)

  # Send joint torques, joint positions and velocities to robot
  if not REAL:
    cmd = MjSimCmd(tau_des, q_des, dq_des, kp, kd)
    robot.set_cmd(cmd)
  else:
    controller.send_command(robot, tau_des, q_des, dq_des, kp=kp, kd=kd)

  # Logger
  if config['verbose']:
    logger.debug(f'Solve time = {controller.mpc.solve_time:.4f}s')

  # wait until next control step
  if config['sync'] and not REAL:
    while robot.step_counter < (step_number//ctrl_sim_ratio+1)*ctrl_sim_ratio:
      time.sleep(0.0001)
  else:
    while time.perf_counter() - tic < controller.dt_mpc:
      time.sleep(0.00001)
if not REAL:
  robot.close()
else:
  robot.stop_command_stream()
  udp_connection.__exit__(None, None, None)
  tcp_connection.__exit__(None, None, None)


# Trim data
if record:
  xs = controller.xs[:controller.i]
  us = controller.us[:controller.i]
  x_des = controller.x_des[:controller.i]
  u_des = controller.u_des[:controller.i]
  x_all = controller.x_all[:controller.i]
  u_all = controller.u_all[:controller.i]
  sol_stats = controller.sol_stats[:controller.i]


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
  df.to_csv('data/kinova_xs_us_hs_xdes_udes_sol.csv')


# Plot the MPC solution
if WITHPLOT:
  plot.plotFrameTrajectory(rmodel, xs, frameId, None, label='End-effector')
  plot.plotJointTrajectory(xs, us, controller.dt_mpc, figTitle='Joint position & torque trajectory')
  plot.plotSolutionVsActual(x_des, u_des, xs, us, controller.dt_mpc, joint_id=5)
