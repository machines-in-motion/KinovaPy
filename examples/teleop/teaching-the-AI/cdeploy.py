#!/usr/bin/env python
"""Let the policy drive.

    python cdeploy.py --dry-run     # nothing moves; do this first, ten times
    python cdeploy.py               # the real arm moves

STAND CLEAR OF THE ARM AND KEEP THE E-STOP IN YOUR HAND.
"""

import argparse
import time

import numpy as np

from robot_ai import Arm, Camera, Gripper, Keys, Policy, pick_device

# ---------------------------------------------------------------- your choices

# How many rows of the chunk to execute before taking a new picture.
#   1            = option A: reacts fastest, can shudder
#   policy.chunk = option B: smooth, but blind for a whole chunk
#   3            = option C: in between
# The keyboard is now read between rows, so this no longer changes how late the
# stop button is - only how long the robot goes without looking.
ROWS_TO_USE = 3

# The policy predicts a decimal like 0.07; gripper.set() wants exactly -1/0/+1.
# Anything with |value| below this becomes 0. Print the raw numbers and look at
# their actual range before trusting this.
GRIPPER_CUTOFF = 0.5


def to_gripper_command(value):
    """One decimal from the policy -> exactly -1, 0 or +1."""
    if value > GRIPPER_CUTOFF:
        return 1
    if value < -GRIPPER_CUTOFF:
        return -1
    return 0


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dry-run", action="store_true", help="pretend; nothing moves")
    parser.add_argument("--policy", default="my_policy.pt", help="the checkpoint to drive with")
    args = parser.parse_args()
    dry = args.dry_run

    policy = Policy.load(args.policy, device=pick_device())
    print(f"chunk: {policy.chunk}, action_size: {policy.action_size}, state_size: {policy.state_size}")

    rows = max(1, min(ROWS_TO_USE, policy.chunk))
    print(f"executing {rows} of every {policy.chunk} predicted rows")

    if not dry:
        print(__doc__)
        if input("Is the area clear and is the e-stop in your hand? Type yes: ").strip().lower() != "yes":
            print("stopping, nothing was moved")
            return

    with Camera(dry_run=dry) as camera, Arm(dry_run=dry) as arm, \
            Gripper(dry_run=dry) as gripper, Keys() as keys:

        print("\n[space] start / stop the policy\n[q]     quit\n")
        running = False
        quitting = False
        steps = 0
        run_time = 0.0
        predict_time = 0.0
        stalls = 0

        def may_move():
            """Read the keyboard once and say whether the robot is allowed to move.

            pressed() drains the buffer and returns EVERY key since the last
            call, so ask once per turn and test with `in`, not `==`. This gets
            called between rows as well as between chunks, so pressing space is
            never more than one row (0.1 s) late however big ROWS_TO_USE is.
            """
            nonlocal running, quitting
            pressed = keys.pressed()
            if "q" in pressed:
                quitting = True
            if " " in pressed:
                running = not running
                print("policy running..." if running else "policy stopped")
            return running and not quitting

        def hold_still():
            """Stop the gripper turning. Never just stop sending commands.

            set() is a velocity+torque command that persists until it is
            replaced, so a loop that skips a turn without this leaves the hand
            squeezing on whatever it was holding.
            """
            gripper.set(0)

        try:
            while not quitting:
                if not may_move():
                    hold_still()
                    time.sleep(0.02)
                    continue

                turn_started = time.perf_counter()

                image = camera.get_image()
                if image is None:
                    hold_still()  # no frame: stop, do not coast
                    time.sleep(0.01)
                    continue

                # Measured, not commanded - the only honest signal about where
                # the robot actually is.
                joints = arm.joint_positions()

                # The eight numbers, in the order the recording stored them:
                # joints.0 ... joints.6, then gripper.0
                state = None
                if policy.state_size:
                    state = np.concatenate([joints, [gripper.position()]])

                predict_started = time.perf_counter()
                actions = policy.predict(image, state)  # (chunk, 7), REAL units
                predict_time += time.perf_counter() - predict_started

                # The workspace box cannot catch this: every comparison against
                # NaN is False, so clamp_to_workspace() passes it straight
                # through into a torque-controlled arm. Check it ourselves.
                if not np.isfinite(actions).all():
                    hold_still()
                    running = False
                    print("!! policy produced NaN/inf - stopped. Check the checkpoint.")
                    continue

                for row in actions[:rows]:
                    if not may_move():
                        hold_still()
                        break
                    # Gripper first: move_like_joystick blocks for the whole
                    # 0.1 s, so setting it afterwards would apply each gripper
                    # action one row later than the hand position it was
                    # predicted for. In your recording they were simultaneous.
                    gripper.set(to_gripper_command(row[-1]))
                    arm.move_like_joystick(row[0:3], seconds=0.1)
                    steps += 1

                run_time += time.perf_counter() - turn_started

                moved = float(np.abs(arm.joint_positions() - joints).max())
                first = actions[0]
                asked_for = float(np.abs(first[0:3]).max())
                print(
                    f"step {steps:5d}  xyz {np.round(first[0:3], 2)}  "
                    f"grip {first[-1]:+.2f}->{to_gripper_command(first[-1]):+d}  "
                    f"target {np.round(arm.hand_position(), 3)}  moved {moved:.3f} rad"
                )
                # hand_position() is where the hand is being ASKED to be. If the
                # arm is jammed it keeps reading healthy, so watch the joints.
                if not dry and asked_for > 0.1 and moved < 1e-3:
                    stalls += 1
                    print("   !! commanded but the joints did not move - jammed?")

        except KeyboardInterrupt:
            print("\ninterrupted")

        if steps:
            print(f"\n{steps} actions in {run_time:.1f} s of running = "
                  f"{steps / run_time:.1f} Hz (aiming for 10)")
            print(f"prediction took {predict_time / steps * 1000:.0f} ms per action on average")
        if arm.times_clamped:
            print(f"the safety box stopped the hand {arm.times_clamped} times")
        if stalls:
            print(f"the arm failed to move on {stalls} turns - it was pushing at something")


if __name__ == "__main__":
    main()
