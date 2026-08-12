import cv2  # needed for camera resize/color convert below
import numpy as np
import os
import sys
import time
import mujoco
import pinocchio as pin
from pathlib import Path

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
from button import Button
from lerobot_recorder import Recorder, RecorderError

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "")
config_dir = os.path.join(CONFIG_PATH, 'kinova_config.yml')
config = load_yaml_file(config_dir)

QUIT = "q" in sys.argv
STAY = "stay" in sys.argv
REAL = "real" in sys.argv
WITHPLOT = "plot" in sys.argv 
SAVEDATA = "savedata" in sys.argv

if SAVEDATA:
  from lerobot_recorder import Recorder
  # CAMERA_HEIGHT = 96 #lerobot camera dimensions
  # CAMERA_WIDTH = 128 #lerobot camera dimensions
  CAMERA_HEIGHT = 720 #lerobot camera dimensions
  CAMERA_WIDTH = 1280 #lerobot camera dimensions


def delete_episode(rec, index, frames):
  """Delete the episode being recorded, because the take was a fail.

  Everything logged since start_episode() is dropped and nothing reaches the
  dataset - a bad demonstration is worse than no demonstration, since the policy
  will faithfully learn to copy the mistake.

  Only the take currently in progress can go. Once stop_episode() has written an
  episode out, the recorder has no way to remove it again; the fix there is to
  record into a fresh dataset name. So decide a take is a fail *before* you press
  SPACE to stop it.
  """
  rec.cancel_episode()
  print(f"[rec] episode {index} deleted as a fail - {frames} frame(s) dropped")

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
# KinovaMPC's history is a fixed-size buffer sized from config['sim_time'], and
# controller.update() indexes into it unconditionally - so it overflows once the
# run outlasts sim_time. Since this script now runs until you quit, only enable
# it for 'plot' (still bounded, and guarded in the main loop). 'savedata' gets
# its per-step data from the LeRobot dataset instead, which grows as needed.
record = WITHPLOT
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

# Initialize Joystick
# NOTE: SpaceMouseExpert spawns background processes via multiprocessing, and
# fork() inherits every open file descriptor. Anything that opens hardware must
# come *after* this, otherwise the forked child keeps a handle on that device
# and it stays busy even after this script exits.
spacemouse = SpaceMouseExpert()

# Initialize Gripper
motor = Gim4305()

# Initialize Camera (after the forks above - see note)
w = Worker()
print("Enabling motor...")
motor.enable()
time.sleep(0.1)
motor.set_zero()


# Initialize LeRobot dataset recorder
if SAVEDATA:
  rec = Recorder(
      "teleop_demo",
      fps=10,
      task="teleop pick and place",
      root=Path(__file__).resolve().parent.parent / "datasets",
  )
  rec.add_camera("front", height=CAMERA_HEIGHT, width=CAMERA_WIDTH)
  rec.add_state("joints", nq)
  rec.add_state("gripper", 1)
  rec.add_action("joints_target", nq)
  rec.add_action("gripper_target", 1)
  rec.add_action("control", 6)
  rec.add_state("epose_target", 6)
  rec.add_state("time", 1)
  print(rec.describe())
  print()
  # No start_episode() here - episodes are opened and closed from the main loop
  # with the SPACE key, so one run of this script can record many of them.

# Start robot
if not REAL:
  kd *= 0.1
  robot.start()
else:
  # Switch robot to torque control mode
  robot.start_command_stream(control_mode="TORQUE")
  # Stay at the initial pose for 2 seconds before recording
  for _ in range(200):
    u_des, x_des = controller.update(robot, manipPlan=manipPlan)
    tau_des = np.clip(u_des, umin, umax)
    q_des = np.clip(x_des[:len(q0)], qmin, qmax)
    dq_des = np.clip(x_des[len(q0):], dqmin, dqmax)
    controller.send_command(robot, tau_des, q_des, dq_des, kp=kp, kd=kd)
    time.sleep(controller.dt_mpc)
# Reset the robot data recording
controller.i = 0
logger.debug("Start recording")

  # Start Camera thread
w.start()


### Main loop ###
start_time = time.perf_counter()
recording = False       # is a LeRobot episode currently open?
episode_start = start_time
episode_count = 0

if SAVEDATA:
  print("\n  [SPACE] start/stop recording an episode      "
        "[x] delete the current take (fail)      [q] quit\n")
else:
  print("\n  [q] quit\n")

keys = Button().start()  # restored in the finally below
old_t = time.time()
step_counter = 0
last_sample_timestamp = time.time()
try:
  while True:
    tic = time.perf_counter()
    if config['sync'] and not REAL:
      step_number = robot.step_counter

    ### Keyboard: episode toggle + quit
    # A keypress is inherently one event, so no edge detection or debounce is
    # needed here - unlike a held button, which would toggle every iteration.
    pressed = keys.get_keys()
    if " " in pressed:
      if not SAVEDATA:
        print("[rec] nothing to record: re-run with the 'savedata' argument")
      elif not recording:
        rec.start_episode()
        recording = True
        episode_start = time.perf_counter()
        episode_frames = 0
        print(f"[rec] episode {episode_count} recording... (SPACE to stop)")
      else:
        rec.stop_episode()
        recording = False
        print(f"[rec] episode {episode_count} saved - {episode_frames} frames, "
              f"{time.perf_counter()-episode_start:.1f}s")
        episode_count += 1
    if "x" in pressed:
      if not SAVEDATA:
        print("[rec] nothing to delete: re-run with the 'savedata' argument")
      elif not recording:
        # Saved episodes are final - see delete_episode()'s docstring.
        print("[rec] no take in progress; a saved episode cannot be deleted")
      else:
        delete_episode(rec, episode_count, episode_frames)
        recording = False
        # episode_count is deliberately not bumped: the take never became an
        # episode, so the next one takes this number.
    if "q" in pressed:
      print("[keys] quit")
      break

    # MPC history is a fixed-size buffer; stop before update() runs off the end.
    if record and controller.i >= len(controller.xs) - 1:
      print(f"[mpc] history buffer full after {run_time}s (config sim_time) - stopping")
      break

    

    #robot_state = robot.get_robot_states() # The AI should also feel how the robot moves


    # Define task
    # reads joystick imput, converts to x,y,z
    space_action, space_buttons = spacemouse.get_action() # The action sent to the AI to learn
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

    # (gripper_reply) instead of discarding it, so it can be logged
    if direct == 1:
      gripper_reply = motor.command(place=0, zoom=4, jk=jk, jl=jl, umph=0.05)
    elif direct2 == 1:
      gripper_reply = motor.command(place=0, zoom=-4, jk=jk, jl=jl, umph=-0.05)
    else:
      gripper_reply = motor.command(place=0.0, zoom=0.0,jk=jk, jl=jl, umph=0.0)

    
    # Record to LeRobot dataset (only while an episode is open - see SPACE above)
    if SAVEDATA and recording and (time.time() - last_sample_timestamp >= 0.1):  # Record every 0.1 seconds

      current_camera_img = w.get_frame() # The task view (newest RealSense frame, BGR)
      current_camera_stamp = w.latest_frame_stamp
      if current_camera_img is not None:
        last_sample_timestamp = time.time()
        now = time.time()
        print(1./(now - old_t))
        old_t = now 
        q_obs, v_obs, _ = controller.get_states(robot)
        # RealSense color stream is bgr8 (see worker.py), LeRobot stores RGB.
        cam = cv2.resize(current_camera_img, (CAMERA_WIDTH, CAMERA_HEIGHT),
                        interpolation=cv2.INTER_AREA)
        cam = cv2.cvtColor(cam, cv2.COLOR_BGR2RGB)

        rec.log("front", cam)
        rec.log("joints", q_obs)
        rec.log("gripper", gripper_reply["position"] if gripper_reply else 0.0)
        rec.log("joints_target", q_des)
        rec.log("gripper_target", float(direct - direct2))
        rec.log("control", space_action[:6])
        # pose is a pin.SE3 (a 4x4 matrix); flatten it to [x,y,z,roll,pitch,yaw]
        rec.log("epose_target", np.concatenate(
            [pose.translation, pin.rpy.matrixToRpy(pose.rotation)]))
        # Seconds since *this episode* started, so every episode's clock begins at
        # 0. The absolute perf_counter() value is far too large to survive the
        # float32 cast (0.01s steps collapse to 0).
        rec.log("time", time.perf_counter() - episode_start)
        rec.commit()
        episode_frames += 1

    step_counter += 1
    # Logger
    if config['verbose']:
      logger.debug(f'Solve time = {controller.mpc.solve_time:.4f}s')

    # No need to display the live view (it is heavy)
    # w.show()c

    # Wait until next control step
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
  # First thing: give the terminal back, so anything printed below is readable
  # and the shell still works if this block itself raises.
  keys.restore()
  print("Stopping all services")
  if REAL:
      robot.stop_command_stream()
      udp_connection.__exit__(None, None, None)
      tcp_connection.__exit__(None, None, None)
  if not REAL:
      robot.close()
  # Finish the LeRobot episode and write the dataset out
  if SAVEDATA:
    # Quitting mid-episode still saves it rather than throwing the take away.
    if recording:
      rec.stop_episode()
      print(f"[rec] episode {episode_count} saved on exit - {episode_frames} frames")
      episode_count += 1
    rec.close()
    print(f"[rec] {episode_count} episode(s) recorded this run")
  # Stop camera
  w.stop()
  # Stop joystick
  spacemouse.close()
  # Stop gripper
  motor.close()
  # Stop robot
 


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
# Gated on `record`, not SAVEDATA: these arrays only exist when the MPC history
# was captured (i.e. 'plot'). With 'savedata' the per-step data lives in the
# LeRobot dataset, split per episode, which is the point of this script.
if record:
  import pandas as pd
  data = np.concatenate([xs, us, x_des, u_des, sol_stats], axis=1)
  df = pd.DataFrame(data)
  os.makedirs('data/pos_data', exist_ok=True)
  df.to_csv('data/pos_data/run_1.csv')
  # No join here: w.stop() above already drained the image writer queue. An
  # unbounded join on the capture thread would hang the process (and hold the
  # camera) whenever that thread is wedged on a dead device.


# Plot
if WITHPLOT:
  plot.plotJointTrajectory(xs, us, controller.dt_mpc, figTitle='Joint position & torque trajectory')
  plot.plotFrameTrajectory(rmodel, xs, frameIds[0], None, label='End-effector')
  plot.plotSolutionVsActual(x_des, u_des, xs, us, controller.dt_mpc, joint_id=5)
