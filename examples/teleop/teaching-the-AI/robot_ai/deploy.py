from robot_ai import Policy
from robot_ai import pick_device
from robot_ai import Camera
from robot_ai import Arm
from robot_ai import Gripper
from robot_ai import Keys
import sys

DRY = "--dry-run" in sys.argv

policy = Policy.load("my_policy.pt", device=pick_device())
print(f"chunk: {policy.chunk}, action_size: {policy.action_size}, state_size: {policy.state_size}")

with Camera(dry_run=DRY) as camera, Arm(dry_run=DRY) as arm, \
    Gripper(dry_run=DRY) as gripper, Keys() as keys:
        print("[space] start / stop the policy \n [q]     quit")
        while True:
            if keys.pressed() == "q":
                break

            if keys.pressed() == " ":
                print("policy running...")
                while True:
                    image = camera.read()
                    state = arm.joint_position() + [gripper.position()]
                    actions = policy(image, state)
                    arm.move(actions[0])
                    gripper.move(actions[1])
                    if keys.pressed() == " ":
                        print("policy stopped")
                        break