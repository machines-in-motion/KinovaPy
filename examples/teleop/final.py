import numpy as np
import os
import sys
import time
import mujoco
import pinocchio as pin
import cv2

from worker import Worker
from locompc.plan.manipulation import ReachGoal
from locompc.utils import load_yaml_file
from locompc.sim.mujoco import MjSim, MjSimCmd
from locompc.utils import CustomLogger, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT
logger = CustomLogger(__name__, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT).logger

from KinovaPy.controller import KinovaMPC
from KinovaPy.interface import KinovaHardwareInterface
from KinovaPy.model import PinocchioModel
from KinovaPy import plot
from KinovaPy import SCENE_PATH
from KinovaPy.utils.joy import SpaceMouseExpert
from KinovaPy.gim4305 import Gim4305

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "")
config_dir = os.path.join(CONFIG_PATH, 'kinova_config.yml')
config = load_yaml_file(config_dir)

QUIT = "q" in sys.argv
STAY = "stay" in sys.argv
REAL = "real" in sys.argv
WITHPLOT = "plot" in sys.argv 
SAVEDATA = "savedata" in sys.argv


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


### Warmstart ###
pose = pose0
goals = [pose]
manipPlan = [ReachGoal(frameIds, goals)]*(config['N_h']+1)
controller.warmstart(robot, manipPlan, nb=100)


### Start ###
run_time = config['sim_time']
start = input("\nPress [ENTER] to start...")
print("\n---------------------------- Experiment running ----------------------------")

# Joystick
spacemouse = SpaceMouseExpert()

# Gripper
# motor = Gim4305()
# print("Enabling motor...")
# motor.enable()
# time.sleep(0.1)
# motor.set_zero

# Camera thread
w = Worker()
w.start()

# Send initial command to robot
q0, v0, u0 = controller.get_states(robot)
q_des = q0.copy()
dq_des = np.zeros(nq)
tau_des = u0
if not REAL:
  cmd = MjSimCmd(tau_des, q_des, dq_des, kp, kd)
  robot.start()
  robot.set_cmd(cmd)
else:
  robot.start_command_stream(control_mode="TORQUE")
  controller.send_command(robot, tau_des, q_des, dq_des, kp=kp, kd=kd)

start_time = time.perf_counter()
try:
  while time.perf_counter()-start_time < run_time:
    tic = time.perf_counter()
    if config['sync'] and not REAL:
      step_number = robot.step_counter

    # Define task
    # reads joystick imput, converts to x,y,z
    space_action, space_buttons = spacemouse.get_action()
    delta_trans = space_action[:3]*config['dt']*0.2
    digital_cmd = [space_buttons[0], space_buttons[1]]
    pose.translation += delta_trans

    # Safety boundaries
    if pose.translation[2] >= 0.60: #top
      pose.translation[2] = 0.60

    if pose.translation[2] <= 0.17: #bottom
        pose.translation[2] = 0.17

    if pose.translation[1] >= 0.40: #left
        pose.translation[1] = 0.40

    if pose.translation[1] <= -0.40: #right
        pose.translation[1] = -0.40

    if pose.translation[0] >= 0.58: #front
        pose.translation[0] = 0.58

    if pose.translation[0] <= 0.25: #back
        pose.translation[0] = 0.25


    ### MPC  (DO NOT CHANGE)
    goals = [pose]
    manipPlan = [ReachGoal(frameIds, goals)]*(config['N_h']+1)
    u_des, x_des = controller.update(robot, manipPlan=manipPlan)
    tau_des = np.clip(u_des, umin, umax)
    q_des = np.clip(x_des[:len(q0)], qmin, qmax)
    dq_des = np.clip(x_des[len(q0):], dqmin, dqmax)
    if not REAL:
      cmd = MjSimCmd(tau_des, q_des, dq_des, kp, kd)
      robot.set_cmd(cmd)
    else:
      controller.send_command(robot, tau_des, q_des, dq_des, kp=kp, kd=kd)


    ### Gripper
    jk = 0
    jl = 0.03

    direct = digital_cmd[0] #left button when wire faces away from you
    direct2 = digital_cmd[1] #right button when wire faces away from you

    # if direct == 1:
    #   motor.command(place=0, zoom=6, jk=jk, jl=jl, umph=0.05)
    # elif direct2 == 1:
    #   motor.command(place=0, zoom=-6, jk=jk, jl=jl, umph=-0.05)
    # else:
    #   motor.command(place=0.0, zoom=0.0,jk=jk, jl=jl, umph=0.0)


    # Logger
    if config['verbose']:
      logger.debug(f'Solve time = {controller.mpc.solve_time:.4f}s')

    # Display the latest camera frame (must happen on the main thread)
    w.show()

    # wait until next control step
    if config['sync'] and not REAL:
      while robot.step_counter < (step_number//ctrl_sim_ratio+1)*ctrl_sim_ratio:
        time.sleep(0.0001)
    else:
      while time.perf_counter() - tic < controller.dt_mpc:
        time.sleep(0.0001)

    if not w.thread.is_alive():
      break
except KeyboardInterrupt:
    pass
finally:
  print("Stopping all services")
  # Stop camera
  w.stop()
  # Stop robot & gripper
  if REAL:
    robot.stop_command_stream()
    udp_connection.__exit__(None, None, None)
    tcp_connection.__exit__(None, None, None)
    cv2.destroyAllWindows()
    # motor.close()
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
  w.thread.join()


# Plot
if WITHPLOT:
  plot.plotFrameTrajectory(rmodel, xs, frameIds[0], None, label='End-effector')
  plot.plotJointTrajectory(xs, us, controller.dt_mpc, figTitle='Joint position & torque trajectory')
  plot.plotSolutionVsActual(x_des, u_des, xs, us, controller.dt_mpc, joint_id=5)
