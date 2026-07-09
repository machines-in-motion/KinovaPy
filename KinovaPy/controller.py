import numpy as np
from locompc.control.mpc import MaNMPC
from locompc.utils import CustomLogger, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT
logger = CustomLogger(__name__, GLOBAL_LOG_LEVEL, GLOBAL_LOG_FORMAT).logger

from kortex_api.Exceptions.KServerException import KServerException

from KinovaPy.interface import KinovaHardwareInterface


##### Sim/Real controller #####
class KinovaMPC:
  def __init__(self, rmodel, rdata, config, planner=None, record=True, mode='sim'):
    self.rmodel = rmodel
    self.rdata = rdata
    self.nq = self.rmodel.nq
    self.mode = mode
    self.record = record

    # Initialization
    self.q0 = np.asarray(config['q0'])
    self.v0 = np.asarray(config['dq0'])
    self.x0 = np.concatenate([self.q0, self.v0])
    self.u0 = np.asarray(config['u0'])

    # Define controller
    self.dt_mpc = config['dt_mpc']
    self.dt = config['dt']
    self.mpc_ocp_ratio = round(self.dt_mpc/self.dt) if self.dt_mpc>self.dt else 1
    self.mpc = MaNMPC(rmodel, rdata, config, planner)

    # Warmstart
    self.mpc.warm_start(self.x0, self.u0, manipPlan=None, terminalModel=None)
    self.mpc.solve()
    self.init = False

    if self.record:
      run_length = round(config['sim_time']/self.dt_mpc)+50  # extra steps for safety
      self.xs = np.empty((run_length, len(self.x0)))
      self.us = np.empty((run_length, len(self.u0)))
      self.x_des = np.empty((run_length, len(self.x0)))
      self.u_des = np.empty((run_length, len(self.u0)))
      self.x_all = np.empty((run_length, config['N_h']+1, len(self.x0)))
      self.u_all = np.empty((run_length, config['N_h'], len(self.u0)))
      self.sol_times = np.empty((run_length, 1))
      self.sol_stats = np.empty((run_length, 4))
      self.i = 0

  def get_states(self, robot):
    # Read current states
    if self.mode=='sim':
      q, v = robot.get_states()
      u = robot.get_torques()
    else: # real robot hardware interface
      if isinstance(robot, KinovaHardwareInterface):
        state = robot.get_robot_states()
        if state is None:
          return False
        q = np.asarray(state['position'], dtype=float)
        v = np.asarray(state['velocity'], dtype=float)
        u = np.asarray(state['torque'], dtype=float)

    # wrap joint angles to [-pi, pi] for controller-level consistency
    q = (q + np.pi) % (2.0 * np.pi) - np.pi

    return q, v, u


  def warmstart(self, robot, manipPlan=None, terminalModel=None, nb=1):
    for i in range(nb):
      q, v, u = self.get_states(robot)
      x = np.concatenate([q, np.zeros(len(v))])
      self.mpc.warm_start(x, u, manipPlan=manipPlan, terminalModel=terminalModel)
      self.mpc.solve()

  def send_command(self, robot, tau_des, q_des, dq_des, kp=None, kd=None):
    if isinstance(robot, KinovaHardwareInterface):
      return robot.set_command(
        desired_torque=tau_des,
        desired_joint_position=q_des,
        desired_velocity=dq_des,
        kp=kp,
        kd=kd,
      )
    return False


  def update(self, robot, manipPlan=None, terminalModel=None):
    # Read current states
    q, v, u = self.get_states(robot)

    ### MPC
    x = np.concatenate([q, v])
    if not self.init:
      self.mpc.warm_start(x, u, manipPlan=manipPlan, terminalModel=terminalModel)
      self.init = True
    else:
      self.mpc.update(x, u, manipPlan=manipPlan, terminalModel=terminalModel)
    self.mpc.solve()
    u_sol, x_sol = self.mpc.get_solution_trajectory()
    u_des = u_sol[0]
    x_des = x_sol[self.mpc_ocp_ratio]

    if self.record:
      # Record real trajectory
      self.xs[self.i] = x.copy()
      self.us[self.i] = u.copy()
      # Record solution trajectory
      self.x_des[self.i] = x_des.copy()
      self.u_des[self.i] = u_des
      self.x_all[self.i] = x_sol.copy()
      self.u_all[self.i] = u_sol.copy()
      # Record compute time
      self.sol_times[self.i] = self.mpc.solve_time
      # Record compute stats
      self.sol_stats[self.i] = [self.mpc.solve_time, self.mpc.solver.iter, 
                                self.mpc.solver.KKT, self.mpc.solver.constraint_norm]
      self.i += 1
    return u_des, x_des
