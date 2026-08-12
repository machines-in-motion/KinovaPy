#!/usr/bin/env python
"""Find out which way the gripper opens. Do this before you ever run a policy.

    python tools/test_gripper.py

The motor turns for half a second one way, then the other, and asks you what you
saw. Nothing else in the project knows which direction is "open" - only you can
see that, so write the answer in your notes.

BEFORE YOU RUN IT: nothing in the gripper, nobody's fingers near it.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from robot_ai import Gripper  # noqa: E402


def main():
    print(__doc__)
    if input("Is the gripper empty and clear? Type yes to continue: ").strip().lower() != "yes":
        print("stopping, nothing was moved")
        return

    with Gripper() as gripper:
        for direction in (+1, -1):
            print(f"\nsending {direction:+d} for 0.5 s ...")
            end = time.time() + 0.5
            while time.time() < end:
                gripper.set(direction)
                time.sleep(0.05)
            gripper.set(0)
            answer = input(f"  what did {direction:+d} do? (open / close / nothing): ")
            print(f"  noted: {direction:+d} = {answer.strip() or 'no answer'}")

    print("\nWrite that down. Your deploy script has to turn the policy's decimal")
    print("prediction into exactly -1, 0 or +1, and you need to know which is which.\n")


if __name__ == "__main__":
    main()
