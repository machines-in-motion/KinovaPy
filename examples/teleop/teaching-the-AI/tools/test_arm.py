#!/usr/bin/env python
"""Check the arm moves before you blame your policy.

    python tools/test_arm.py --dry-run     # no robot needed, just prints
    python tools/test_arm.py               # the real arm moves ~5 cm and comes back

It homes the arm, nudges the hand forward, then back, then reports how far it
actually went. If this does not work, nothing built on top of it will.

STAND CLEAR OF THE ARM AND KEEP THE E-STOP IN YOUR HAND.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from robot_ai import Arm  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="pretend; nothing moves")
    args = parser.parse_args()

    if not args.dry_run:
        print(__doc__)
        if input("Is the area clear and is the e-stop in your hand? Type yes: ").strip().lower() != "yes":
            print("stopping, nothing was moved")
            return

    with Arm(dry_run=args.dry_run) as arm:
        start = arm.hand_position()
        print(f"hand starts at {start.round(3)}")

        print("nudging forward for half a second at half speed ...")
        arm.move_like_joystick([0.5, 0.0, 0.0], seconds=0.5)
        forward = arm.hand_position()
        print(f"hand now at    {forward.round(3)}   (moved {np.linalg.norm(forward-start)*100:.1f} cm)")

        print("and back ...")
        arm.move_like_joystick([-0.5, 0.0, 0.0], seconds=0.5)
        back = arm.hand_position()
        print(f"hand back at   {back.round(3)}   ({np.linalg.norm(back-start)*100:.1f} cm from the start)")

        if arm.times_clamped:
            print(f"\nthe safety box stopped the hand {arm.times_clamped} times - it was pushing at a wall")

    expected = 0.5 * 0.5 * 0.2  # joystick * seconds * scale = metres
    print(f"\nexpected about {expected*100:.0f} cm each way. Did you get that?\n")


if __name__ == "__main__":
    main()
