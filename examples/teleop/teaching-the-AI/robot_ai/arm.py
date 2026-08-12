"""The arm block: a Kinova Gen3 that moves the way your joystick used to.

There is a lot of machinery under this file - a whole-body optimiser solving a
control problem 100 times a second, and a torque loop underneath that running at
1000 times a second - but you do not have to touch any of it. From the outside
the arm can do exactly four things:

    arm.go_home()                       stand up in the starting pose
    arm.move_like_joystick([x, y, z])   nudge the hand, exactly like the SpaceMouse did
    arm.hand_position()                 where is the hand right now
    arm.stop()                          let go, safely

`move_like_joystick` takes the SAME three numbers the SpaceMouse produced while
you were recording - each between -1 and +1 - which is precisely what your policy
was trained to predict. That is not a coincidence. It is the whole design: the
policy's job is to replace the human's hand on the joystick, so it must speak the
joystick's language.

WHAT THE NUMBERS MEAN
    1 unit for 1 tick (10 ms)  =  2 mm of hand movement
    so a full 1.0 held for a whole second  =  20 cm

SAFETY
    The hand is not allowed outside this box, no matter what the policy says:
        x  0.25 .. 0.58 m   (away from the base)
        y -0.40 .. 0.40 m   (left / right)
        z  0.17 .. 0.60 m   (up / down)
    This is the same box the teleop script used, so your recordings never went
    outside it either. Do not widen it to "see what happens".

Every heavy import happens inside the methods on purpose, so that the training
half of this project still works on a laptop with no robot libraries installed.
"""

import argparse
import time
from pathlib import Path

import numpy as np

# Same numbers as the teleop recording script. Change them and your policy is
# suddenly driving a robot that behaves differently from the one it learned on.
JOYSTICK_SCALE = 0.2  # joystick unit -> metres per second-ish, before dt
CONTROL_DT = 0.01  # the arm thinks 100 times a second
HOME_JOINTS = [0.0, 0.0, 0.0, 1.57, 0.0, 1.57, -1.57]
WORKSPACE = {"x": (0.25, 0.58), "y": (-0.40, 0.40), "z": (0.17, 0.60)}

# Where the hand ends up when the arm is at HOME_JOINTS (worked out from the
# robot's own model). Only used by dry runs; on the real robot it is measured.
HOME_HAND_XYZ = np.array([0.3146, -0.0249, 0.5384])

CONFIG_PATH = Path(__file__).with_name("kinova_config.yml")


def clamp_to_workspace(xyz):
    """Push a hand position back inside the safe box. Returns (position, was_clamped)."""
    xyz = np.asarray(xyz, dtype=float).copy()
    limits = [WORKSPACE["x"], WORKSPACE["y"], WORKSPACE["z"]]
    clamped = False
    for i, (low, high) in enumerate(limits):
        if xyz[i] < low:
            xyz[i], clamped = low, True
        elif xyz[i] > high:
            xyz[i], clamped = high, True
    return xyz, clamped


class Arm:
    """The Kinova arm. Always use it with `with`, so it lets go if anything breaks."""

    def __init__(self, dry_run=False, ip="192.168.1.10"):
        self.dry_run = bool(dry_run)
        self.ip = ip
        self.connected = False
        self.times_clamped = 0

        self._config = None
        self._controller = None
        self._robot = None
        self._tcp = None
        self._udp = None
        self._frame_ids = None
        self._target = None  # a full pose (position + orientation)
        self._limits = {}

    # --------------------------------------------------------------- lifetime

    def connect(self):
        """Wake the arm up, stand it in the home pose, and take control of it."""
        if self.dry_run:
            print("[arm] dry run: pretending. Nothing will move.")
            self._target_xyz_dry = HOME_HAND_XYZ.copy()
            self.connected = True
            return self

        import pinocchio as pin
        from locompc.plan.manipulation import ReachGoal
        from locompc.utils import load_yaml_file

        from KinovaPy import utilities
        from KinovaPy.controller import KinovaMPC
        from KinovaPy.interface import KinovaHardwareInterface
        from KinovaPy.model import PinocchioModel

        self._pin = pin
        self._ReachGoal = ReachGoal

        self._config = load_yaml_file(str(CONFIG_PATH))
        model = PinocchioModel(frame_names=self._config["endEffectorFrameName"])
        self._controller = KinovaMPC(
            model.rmodel, model.rdata, self._config, planner=None, record=False, mode="real"
        )

        oc = self._controller.mpc.oc
        self._limits = {
            "u": 0.99 * np.asarray(oc.ctrlLimit),
            "qmin": 0.99 * np.asarray(oc.stateLowerLimit),
            "qmax": 0.99 * np.asarray(oc.stateUpperLimit),
            "dq": 0.99 * np.asarray(oc.velocityLimit),
            "kp": np.asarray(self._config["kp_scale"]) * np.asarray(self._config["kp_ref"]),
            "kd": np.asarray(self._config["kd_scale"]) * np.asarray(self._config["kd_ref"]),
        }

        print(f"[arm] connecting to {self.ip} ...")
        # KinovaPy normally reads these off the command line; we hand them over
        # directly so that your own script's arguments are left alone.
        args = argparse.Namespace(ip=self.ip, username="admin", password="admin")
        self._tcp = utilities.DeviceConnection.createTcpConnection(args)
        self._udp = utilities.DeviceConnection.createUdpConnection(args)
        router = self._tcp.__enter__()
        router_rt = self._udp.__enter__()
        self._robot = KinovaHardwareInterface(router, router_rt, torque_limits=self._limits["u"])
        self._robot.stop_command_stream()
        time.sleep(1.0)

        self.go_home()
        self.connected = True
        return self

    def go_home(self):
        """Drive to the starting pose and hold there. Call this before every run."""
        if self.dry_run:
            self._target_xyz_dry = HOME_HAND_XYZ.copy()
            print(f"[arm] (dry) home, hand at {HOME_HAND_XYZ.round(3)}")
            return

        pin = self._pin
        print("[arm] moving to the home pose ...")
        self._robot.move_to_home(q0=np.asarray(HOME_JOINTS))
        time.sleep(1.0)

        q0, _, _ = self._controller.get_states(self._robot)
        rmodel = self._controller.rmodel
        rdata = self._controller.rdata
        pin.forwardKinematics(rmodel, rdata, q0)
        pin.updateFramePlacements(rmodel, rdata)
        self._frame_ids = [rmodel.getFrameId(n) for n in self._config["endEffectorFrameName"]]
        self._target = rdata.oMf[self._frame_ids[0]].copy()

        # Let the optimiser think about standing still for a moment before it
        # is allowed to command any torque.
        plan = self._plan()
        self._controller.warmstart(self._robot, plan, nb=100)
        self._robot.start_command_stream(control_mode="TORQUE")
        for _ in range(200):
            self._one_tick()
        print(f"[arm] ready, hand at {self.hand_position().round(3)}")

    def stop(self):
        """Hand the arm back to its own safety controller and hang up."""
        if self.dry_run:
            self.connected = False
            return
        if self._robot is not None:
            try:
                self._robot.stop_command_stream()
            except Exception as exc:
                print(f"[arm] while stopping: {exc}")
        for conn in (self._udp, self._tcp):
            if conn is not None:
                try:
                    conn.__exit__(None, None, None)
                except Exception:
                    pass
        self._robot, self._tcp, self._udp = None, None, None
        self.connected = False
        print("[arm] stopped")

    def __enter__(self):
        return self.connect()

    def __exit__(self, *exc):
        self.stop()
        return False

    # ---------------------------------------------------------------- moving

    def move_like_joystick(self, xyz, seconds=0.1):
        """Nudge the hand for `seconds`, as if a human held the joystick at `xyz`.

        xyz     three numbers between -1 and +1 (forward, sideways, up)
        seconds how long to hold it. 0.1 s = one step of a 10 Hz policy.
        """
        if not self.connected:
            raise RuntimeError("call arm.connect() (or use `with Arm() as arm:`) first")

        xyz = np.clip(np.asarray(xyz, dtype=float).reshape(3), -1.0, 1.0)
        ticks = max(1, int(round(seconds / CONTROL_DT)))
        step = xyz * CONTROL_DT * JOYSTICK_SCALE

        for _ in range(ticks):
            tic = time.perf_counter()
            if self.dry_run:
                moved, hit_wall = clamp_to_workspace(self._target_xyz_dry + step)
                self._target_xyz_dry = moved
            else:
                moved, hit_wall = clamp_to_workspace(self._target.translation + step)
                self._target.translation[:] = moved
                self._one_tick()
            if hit_wall:
                self.times_clamped += 1
            while time.perf_counter() - tic < CONTROL_DT:
                time.sleep(0.0001)

    def hand_position(self):
        """Where the hand is being asked to be, in metres (x, y, z)."""
        if self.dry_run:
            return self._target_xyz_dry.copy()
        return np.asarray(self._target.translation).copy()

    def joint_positions(self):
        """Where the seven joints actually are, in radians.

        These are the same numbers the recording stored as `joints`, read the
        same way (wrapped identically), so a policy that was trained on them
        sees exactly what it expects.
        """
        if not self.connected:
            raise RuntimeError("call arm.connect() first")
        if self.dry_run:
            return np.asarray(HOME_JOINTS, dtype=float)
        q, _, _ = self._controller.get_states(self._robot)
        return np.asarray(q, dtype=float)

    # --------------------------------------------------------------- internals

    def _plan(self):
        goal = self._ReachGoal(self._frame_ids, [self._target])
        return [goal] * (self._config["N_h"] + 1)

    def _one_tick(self):
        """One turn of the 100 Hz loop: think, then command."""
        lim = self._limits
        u_des, x_des = self._controller.update(self._robot, manipPlan=self._plan())
        tau = np.clip(u_des, -lim["u"], lim["u"])
        q = np.clip(x_des[: len(HOME_JOINTS)], lim["qmin"], lim["qmax"])
        dq = np.clip(x_des[len(HOME_JOINTS) :], -lim["dq"], lim["dq"])
        self._controller.send_command(self._robot, tau, q, dq, kp=lim["kp"], kd=lim["kd"])
