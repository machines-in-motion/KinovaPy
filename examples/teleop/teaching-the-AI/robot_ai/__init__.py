"""robot_ai - the lego blocks for teaching a robot arm by showing it what to do.

There are eight blocks and that is the whole toolbox:

    LEARNING                                        WHERE
    RobotData      your recorded demonstrations     robot_ai/data.py
    split_episodes hold some takes back for testing robot_ai/data.py
    Policy         the network: picture -> actions  robot_ai/model.py
    vision         picture -> numbers               robot_ai/vision.py

    DRIVING
    Camera         the robot's eye                  robot_ai/camera.py
    Arm            the robot's arm                  robot_ai/arm.py
    Gripper        the robot's hand                 robot_ai/gripper.py
    Keys           your keyboard                    robot_ai/keyboard.py

You will write two scripts of your own using these: one that trains a policy,
and one that lets the policy drive. See tutorials/.

The robot blocks are imported lazily (the heavy robot libraries only load when
you actually build an Arm or a Camera), so `import robot_ai` works fine on a
machine with no robot attached.
"""

from . import vision  # noqa: F401
from .data import RobotData, split_episodes  # noqa: F401
from .model import Policy, pick_device  # noqa: F401

__all__ = [
    "vision",
    "RobotData",
    "split_episodes",
    "Policy",
    "pick_device",
    "Camera",
    "Arm",
    "Gripper",
    "Keys",
]


def __getattr__(name):
    """Load the robot blocks only when someone asks for them."""
    if name == "Camera":
        from .camera import Camera

        return Camera
    if name == "Arm":
        from .arm import Arm

        return Arm
    if name == "Gripper":
        from .gripper import Gripper

        return Gripper
    if name == "Keys":
        from .keyboard import Keys

        return Keys
    raise AttributeError(f"module 'robot_ai' has no attribute {name!r}")
