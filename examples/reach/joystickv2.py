import pyspacemouse
import time
import numpy as np
import os
import sys

device = pyspacemouse.open()

while device:
    state =device.read()
    
    from kortex_api.autogen.messages import Base_pb2

    def spacemouse_to_twist(state, linear_scale=1.0, angular_scale=1.0, deadzone=0.05):
        def dz(v):
            return 0.0 if abs(v) < deadzone else v

        command = Base_pb2.TwistCommand()
        command.reference_frame = Base_pb2.CARTESIAN_REFERENCE_FRAME_TOOL  # or _BASE
        twist = command.twist
        twist.linear_x  = dz(state.x)     * linear_scale
        twist.linear_y  = dz(state.y)     * linear_scale
        twist.linear_z  = dz(state.z)     * linear_scale
        twist.angular_x = dz(state.roll)  * angular_scale
        twist.angular_y = dz(state.pitch) * angular_scale
        twist.angular_z = dz(state.yaw)   * angular_scale

        print(
             f"x={twist.linear_x} y={twist.linear_y} z={twist.linear_z}"
             f"roll{twist.angular_x} pitch={twist.angular_y} yaw={twist.angular_z}"
             f"buttons={state.buttons}")
    
