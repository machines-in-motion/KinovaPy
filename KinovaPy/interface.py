#! /usr/bin/env python3

import os
import sys
import threading
import time
import numpy as np

from kortex_api.autogen.client_stubs.ActuatorConfigClientRpc import ActuatorConfigClient
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient, BaseFunctionUid
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.client_stubs.DeviceManagerClientRpc import DeviceManagerClient
from kortex_api.autogen.messages import ActuatorConfig_pb2, Base_pb2, BaseCyclic_pb2, Common_pb2
from kortex_api.RouterClient import RouterClientSendOptions


class KinovaHardwareInterface:
    """Basic hardware interface for the Kinova Gen3 robot."""

    @staticmethod
    def _normalize_router(router):
        if router is None:
            return None
        if hasattr(router, "registerNotificationCallback"):
            return router
        if hasattr(router, "router") and hasattr(router.router, "registerNotificationCallback"):
            return router.router
        raise TypeError("Expected a Kortex RouterClient or a DeviceConnection wrapper")

    def __init__(self, router, router_real_time=None, torque_limits=None):
        self.ACTION_TIMEOUT_DURATION = 20

        router = self._normalize_router(router)
        router_real_time = self._normalize_router(router_real_time or router)

        self.actuator_config = ActuatorConfigClient(router)
        self.base = BaseClient(router)
        self.base_cyclic = BaseCyclicClient(router_real_time or router)

        self.base_command = BaseCyclic_pb2.Command()
        self.base_feedback = BaseCyclic_pb2.Feedback()

        self.device_manager = DeviceManagerClient(router)
        device_handles = self.device_manager.ReadAllDevices()
        self.actuator_count = self.base.GetActuatorCount().count

        for handle in device_handles.device_handle:
            if handle.device_type in (Common_pb2.BIG_ACTUATOR, Common_pb2.SMALL_ACTUATOR):
                self.base_command.actuators.add()
                self.base_feedback.actuators.add()

        self.send_option = RouterClientSendOptions()
        self.send_option.andForget = False
        self.send_option.delay_ms = 0
        self.send_option.timeout_ms = 3

        self._command_lock = threading.Lock()
        self._latest_command = {
            "desired_torque": None,
            "desired_joint_position": None,
            "desired_velocity": None,
            "kp": None,
            "kd": None,
        }
        self._cyclic_thread = None
        self._stop_event = threading.Event()
        self._streaming = False
        self._stream_period_s = 0.001
        self._user_joint_torque_limits = None
        if torque_limits is not None:
            self.set_joint_torque_limits(torque_limits)
        self._current_control_mode = None

    def _set_servoing_mode(self, mode):
        servo_mode = Base_pb2.ServoingModeInformation()
        servo_mode.servoing_mode = mode
        self.base.SetServoingMode(servo_mode)

    def _set_actuator_control_mode(self, mode_name):
        mode_name = mode_name.upper()
        if not hasattr(ActuatorConfig_pb2.ControlMode, 'Value'):
            return False

        try:
            mode_value = ActuatorConfig_pb2.ControlMode.Value(mode_name)
        except ValueError:
            return False

        set_mode = getattr(self.actuator_config, "SetControlMode", None)
        if set_mode is None:
            return False

        for actuator_idx in range(self.actuator_count):
            try:
                control_mode_msg = ActuatorConfig_pb2.ControlModeInformation()
                control_mode_msg.control_mode = mode_value
                self.SendCallWithRetry(set_mode, 3, control_mode_msg, actuator_idx + 1)
            except Exception as exc:
                print(f"Failed to set actuator {actuator_idx + 1} control mode to {mode_name}: {exc}")
                return False

        return True

    def get_available_actuator_control_modes(self):
        if not hasattr(ActuatorConfig_pb2.ControlMode, 'keys'):
            return []
        return [name for name in ActuatorConfig_pb2.ControlMode.keys() if name != 'NONE']

    def set_current_control_mode(self, mode_name):
        """Record the current high-level control mode used by the cyclic publisher.

        mode_name is stored as an upper-case string (e.g. 'TORQUE', 'POSITION').
        """
        if mode_name is None:
            self._current_control_mode = None
        else:
            self._current_control_mode = str(mode_name).upper()

    def get_current_control_mode(self):
        return self._current_control_mode

    def set_joint_torque_limits(self, torque_limits):
        torque_limits = self._as_array(torque_limits, self.actuator_count)
        self._user_joint_torque_limits = np.abs(torque_limits).astype(float)

    def clear_joint_torque_limits(self):
        self._user_joint_torque_limits = None

    def get_joint_torque_limits(self):
        if self._user_joint_torque_limits is not None:
            return self._user_joint_torque_limits

        limits = np.full(self.actuator_count, np.inf, dtype=float)
        try:
            future = self.base.router.send(
                None,
                1,
                BaseFunctionUid.uidGetAllJointsTorqueHardLimitation,
                0,
                self.send_option,
            )
            result = future.result(self.send_option.getTimeoutInSecond())
            resp = Base_pb2.JointsLimitationsList()
            resp.ParseFromString(result.payload)
        except Exception as exc:
            print(f"Failed to read joint torque limits: {exc}")
            return limits

        for entry in resp.joints_limitations:
            if entry.type == Base_pb2.LimitationType.TORQUE_LIMITATION:
                idx = int(entry.joint_identifier) - 1
                if 0 <= idx < self.actuator_count:
                    limits[idx] = float(entry.value)
        return limits

    def _refresh_feedback(self):
        return self.SendCallWithRetry(self.base_cyclic.RefreshFeedback, 3)

    def _as_array(self, value, size, default=None):
        if value is None:
            if default is None:
                return np.zeros(size, dtype=float)
            return np.array(default, dtype=float)
        if np.isscalar(value):
            return np.full(size, float(value), dtype=float)
        arr = np.asarray(value, dtype=float).reshape(-1)
        if arr.size == 1:
            return np.full(size, arr[0], dtype=float)
        if arr.size != size:
            raise ValueError(f"Expected {size} values, received {arr.size}")
        return arr

    def _wrap_angles(self, angles):
        angles = np.asarray(angles, dtype=float)
        return (angles + np.pi) % (2.0 * np.pi) - np.pi

    def check_for_end_or_abort(self, event):
        def check(notification, event=event):
            print("EVENT : " + Base_pb2.ActionEvent.Name(notification.action_event))
            if notification.action_event in (Base_pb2.ACTION_END, Base_pb2.ACTION_ABORT):
                event.set()
        return check

    def get_robot_states(self):
        """Return joint position, velocity and torque arrays in SI units."""
        feedback = self._refresh_feedback()
        if feedback is None:
            return None

        self.base_feedback = feedback
        positions_deg = np.array([act.position for act in feedback.actuators[:self.actuator_count]], dtype=float)
        velocities_deg_s = np.array([act.velocity for act in feedback.actuators[:self.actuator_count]], dtype=float)
        torques_nm = np.array([act.torque for act in feedback.actuators[:self.actuator_count]], dtype=float)

        positions = np.deg2rad(positions_deg)
        velocities = np.deg2rad(velocities_deg_s)
        torques = torques_nm

        return {
            "position": positions,
            "velocity": velocities,
            "torque": torques,
        }

    def move_to_home(self, q0=None):
        """Move the arm to a user-defined joint configuration q0 or the predefined Home pose.

        If `q0` is None the stored robot action named "Home" is executed.
        If `q0` is provided (radians), a bang-bang position controller will
        drive the joints from the current position to `q0` using position-mode commands.
        """
        # Ensure any background streaming is stopped
        self.stop_command_stream()

        if q0 is None:
            # Make sure the arm is in Single Level Servoing mode for stored action
            base_servo_mode = Base_pb2.ServoingModeInformation()
            base_servo_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
            self.base.SetServoingMode(base_servo_mode)

            # Move arm to ready position (stored Home action)
            print("Moving the arm to a safe position (stored Home action)")
            action_type = Base_pb2.RequestedActionType()
            action_type.action_type = Base_pb2.REACH_JOINT_ANGLES
            action_list = self.base.ReadAllActions(action_type)
            action_handle = None
            for action in action_list.action_list:
                if action.name == "Home":
                    action_handle = action.handle

            if action_handle is None:
                print("Can't reach safe position. Exiting")
                return False

            e = threading.Event()
            notification_handle = self.base.OnNotificationActionTopic(
                self.check_for_end_or_abort(e),
                Base_pb2.NotificationOptions()
            )

            self.base.ExecuteActionFromReference(action_handle)

            print("Waiting for movement to finish ...")
            finished = e.wait(self.ACTION_TIMEOUT_DURATION)
            self.base.Unsubscribe(notification_handle)

            if finished:
                print("Cartesian movement completed")
            else:
                print("Timeout on action notification wait")
            return finished

        # Otherwise perform bang-bang position control to q0
        q0_array = self._as_array(q0, self.actuator_count)

        # Switch to position control mode
        if not self._set_actuator_control_mode('POSITION'):
            print('Failed to switch actuators to POSITION mode')
            return False

        # Use low-level servoing for direct commands
        self._set_servoing_mode(Base_pb2.LOW_LEVEL_SERVOING)

        # Bang-bang parameters
        vmax = 0.5  # rad/s commanded velocity magnitude
        dt = 0.002   # control interval (s)
        tol = 1e-3  # rad tolerance
        max_time = self.ACTION_TIMEOUT_DURATION

        start_time = time.time()

        # read initial state
        state = self.get_robot_states()
        if state is None:
            return False
        current = np.array(state['position'], dtype=float)

        # command loop
        while True:
            error = self._wrap_angles(q0_array - current)
            if np.all(np.abs(error) <= tol):
                break

            direction = np.sign(error)
            step = direction * vmax * dt
            # avoid overshoot per-joint
            step = np.where(np.abs(step) > np.abs(error), error, step)

            q_next = current + step

            # populate command message
            self.base_command.frame_id = (self.base_command.frame_id + 1) % 65536
            for actuator_idx in range(self.actuator_count):
                self.base_command.actuators[actuator_idx].flags = 1  # position
                self.base_command.actuators[actuator_idx].position = float(np.rad2deg(q_next[actuator_idx]))
                self.base_command.actuators[actuator_idx].velocity = 0.0
                self.base_command.actuators[actuator_idx].torque_joint = 0.0
                self.base_command.actuators[actuator_idx].command_id = self.base_command.frame_id

            try:
                fb = self.base_cyclic.Refresh(self.base_command, 0, self.send_option)
                if fb is not None:
                    current = np.deg2rad(np.array([a.position for a in fb.actuators[:self.actuator_count]], dtype=float))
            except Exception as exc:
                print(f'Failed to send bang-bang command: {exc}')

            if time.time() - start_time > max_time:
                print('Timed out while moving to home (bang-bang)')
                return False

            time.sleep(dt)

        # restore servoing to single-level after motion
        self._set_servoing_mode(Base_pb2.SINGLE_LEVEL_SERVOING)
        return True


    def _publish_command(self, command=None):

        state = self.get_robot_states()
        if state is None:
            return False

        position = state["position"]
        velocity = state["velocity"]
        torque = state["torque"]

        if command is None:
            with self._command_lock:
                command = self._latest_command.copy()

        desired_torque = command.get("desired_torque", None)
        desired_joint_position = command.get("desired_joint_position", None)
        desired_velocity = command.get("desired_velocity", None)
        kp = command.get("kp", None)
        kd = command.get("kd", None)

        # Default to the measured state unless the caller explicitly supplies a target.
        q_ref_rad = self._as_array(desired_joint_position, self.actuator_count, default=position)
        dq_ref_rad_s = self._as_array(desired_velocity, self.actuator_count, default=velocity)
        tau_ff = self._as_array(desired_torque, self.actuator_count, default=np.zeros(self.actuator_count))
        kp_arr = self._as_array(kp, self.actuator_count, default=np.zeros(self.actuator_count))
        kd_arr = self._as_array(kd, self.actuator_count, default=np.zeros(self.actuator_count))

        mode = (self._current_control_mode or "TORQUE").upper()

        if mode == "TORQUE":
            q_error = self._wrap_angles(q_ref_rad - position)
            dq_error = dq_ref_rad_s - velocity
            tau_cmd = tau_ff + kp_arr * q_error + kd_arr * dq_error

            torque_limits = self.get_joint_torque_limits()
            tau_cmd = np.clip(tau_cmd, -torque_limits, torque_limits)

            q_des_deg = np.rad2deg(position)   # set to measured position to avoid following error to trigger
            dq_des_deg_s = np.rad2deg(dq_ref_rad_s)
        elif mode == "POSITION":
            # position-control: send desired joint positions/velocities, zero torque
            q_des_deg = np.rad2deg(q_ref_rad)
            dq_des_deg_s = np.rad2deg(dq_ref_rad_s)
            tau_cmd = np.zeros(self.actuator_count, dtype=float)
        elif mode == "VELOCITY":
            # velocity-control: keep current positions, send velocity command, zero torque
            q_des_deg = np.rad2deg(position)
            dq_des_deg_s = np.rad2deg(dq_ref_rad_s)
            tau_cmd = np.zeros(self.actuator_count, dtype=float)
        else:
            # fallback: do not command torques
            q_des_deg = np.rad2deg(q_ref_rad)
            dq_des_deg_s = np.rad2deg(dq_ref_rad_s)
            tau_cmd = np.zeros(self.actuator_count, dtype=float)

        self.base_command.frame_id += 1
        if self.base_command.frame_id > 65535:
            self.base_command.frame_id = 0

        for actuator_idx in range(self.actuator_count):
            # flags is a bitmask: enable position(1), velocity(2), torque(4) as needed
            if mode == "TORQUE":
                flags = 1 | 2 | 4
            elif mode == "POSITION":
                flags = 1
            elif mode == "VELOCITY":
                flags = 2
            else:
                flags = 0

            self.base_command.actuators[actuator_idx].flags = int(flags)
            self.base_command.actuators[actuator_idx].position = float(q_des_deg[actuator_idx])
            self.base_command.actuators[actuator_idx].velocity = float(dq_des_deg_s[actuator_idx])
            self.base_command.actuators[actuator_idx].torque_joint = float(tau_cmd[actuator_idx])
            self.base_command.actuators[actuator_idx].command_id = self.base_command.frame_id

        try:
            self.base_feedback = self.base_cyclic.Refresh(self.base_command, 0, self.send_option)
            return True
        except Exception as exc:
            print(f"Failed to send command: {exc}")
            return False

    def set_command(self, desired_torque=None, desired_joint_position=None, desired_velocity=None,
                    kp=None, kd=None):
        """Update the latest target command and send it immediately."""
        with self._command_lock:
            if desired_torque is not None:
                self._latest_command["desired_torque"] = desired_torque
            if desired_joint_position is not None:
                self._latest_command["desired_joint_position"] = desired_joint_position
            if desired_velocity is not None:
                self._latest_command["desired_velocity"] = desired_velocity
            if kp is not None:
                self._latest_command["kp"] = kp
            if kd is not None:
                self._latest_command["kd"] = kd
            command = self._latest_command.copy()

        self._publish_command(command)
        return True

    def start_command_stream(self, period_s=0.001, control_mode="TORQUE"):
        """Start a background thread that publishes commands at approximately 1 kHz.

        control_mode: one of the actuator control mode names supported by the
        Kortex API (e.g. 'TORQUE', 'POSITION', 'VELOCITY').
        """
        if self._streaming:
            return True

        mode_name = control_mode.upper()
        if mode_name not in self.get_available_actuator_control_modes():
            raise ValueError(
                f"Unsupported control mode '{control_mode}'. "
                f"Available modes: {self.get_available_actuator_control_modes()}"
            )

        if not self._set_actuator_control_mode(mode_name):
            raise RuntimeError(f"Failed to switch actuators to {mode_name} mode")

        # record the mode so the cyclic publisher can choose which fields to write
        self.set_current_control_mode(mode_name)

        self._set_servoing_mode(Base_pb2.LOW_LEVEL_SERVOING)

        # Send an initial hold command so the arm does not fall under gravity
        try:
            state = self.get_robot_states()
            if state is None:
                state = {
                    'position': np.zeros(self.actuator_count, dtype=float),
                    'velocity': np.zeros(self.actuator_count, dtype=float),
                    'torque': np.zeros(self.actuator_count, dtype=float),
                }

            if mode_name == 'TORQUE':
                # use measured torques as a safe initial torque hold
                self.set_command(desired_torque=state['torque'],
                                 desired_joint_position=state['position'],
                                 desired_velocity=np.zeros(self.actuator_count))
            elif mode_name in ('POSITION', 'VELOCITY'):
                # command current positions and zero velocities to hold
                self.set_command(desired_joint_position=state['position'],
                                 desired_velocity=np.zeros(self.actuator_count))
        except Exception as exc:
            print(f"Failed to send initial hold command: {exc}")

        self._stream_period_s = period_s
        self._stop_event = threading.Event()
        self._streaming = True
        self._cyclic_thread = threading.Thread(target=self._command_loop, daemon=True)
        self._cyclic_thread.start()
        return True

    def _command_loop(self):
        while not self._stop_event.is_set():
            self._publish_command()
            time.sleep(self._stream_period_s)
        # self._streaming = False

    def stop_command_stream(self):
        """Stop the background command stream."""
        if not self._streaming:
            return
        self._stop_event.set()
        if self._cyclic_thread is not None:
            self._cyclic_thread.join(timeout=1.0)
        self._set_actuator_control_mode("position")
        self._set_servoing_mode(Base_pb2.SINGLE_LEVEL_SERVOING)
        self.set_current_control_mode(None)
        self._streaming = False

    @staticmethod
    def SendCallWithRetry(call, retry, *args):
        for _ in range(retry):
            try:
                return call(*args)
            except Exception:
                continue
        print("Failed to communicate")
        return None


TorqueExample = KinovaHardwareInterface


def main():
    import argparse

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    import utilities

    parser = argparse.ArgumentParser()
    # parser.add_argument("--ip", default="192.168.1.10")
    args = utilities.parseConnectionArguments(parser)

    with utilities.DeviceConnection.createTcpConnection(args) as router:
        with utilities.DeviceConnection.createUdpConnection(args) as router_real_time:
            interface = KinovaHardwareInterface(router, router_real_time)
            interface.move_to_home()
            state = interface.get_robot_states()
            if state is not None:
                print("Initial states:", state["position"])


if __name__ == "__main__":
    raise SystemExit(main())