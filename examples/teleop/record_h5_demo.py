"""Record a few episodes as .h5 files, then convert them to a LeRobot dataset.

    python examples/record_h5_demo.py

Compare this with record_demo.py: the only difference is the import line.

    from lerobot_recorder import Recorder   ->  writes a LeRobot dataset
    from h5_recorder import Recorder        ->  writes one .h5 per episode

Everything after that is identical, which is the point.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fake_robot import CAMERA_HEIGHT, CAMERA_WIDTH, FakeRobot, demo_motion  # noqa: E402

from h5_recorder import Recorder, episode_files  # noqa: E402

FPS = 20
STEPS_PER_EPISODE = 40
EPISODES = 3

PROJECT = Path(__file__).resolve().parent.parent


def main() -> None:
    rec = Recorder(
        "cube_picking_h5",
        fps=FPS,
        task="pick up the red cube",
        root=PROJECT / "datasets",
    )

    rec.add_camera("front", height=CAMERA_HEIGHT, width=CAMERA_WIDTH)
    rec.add_state("joints", 2)
    rec.add_state("gripper", 1)
    rec.add_action("joints_target", 2)
    rec.add_action("gripper_target", 1)

    print(rec.describe(), "\n")

    for episode in range(EPISODES):
        robot = FakeRobot(seed=episode + rec.episode_count)
        motion = demo_motion(robot, STEPS_PER_EPISODE)

        with rec.episode():
            for target_joints, target_gripper in motion:
                loop_started = time.perf_counter()

                rec.log("front", robot.camera())
                rec.log("joints", robot.joints())
                rec.log("gripper", robot.gripper())

                robot.move_to(target_joints, target_gripper)
                rec.log("joints_target", target_joints)
                rec.log("gripper_target", target_gripper)

                rec.commit()

                time.sleep(max(0.0, 1 / FPS - (time.perf_counter() - loop_started)))

    rec.close()

    print(f"\nEpisode files in {rec.path}:")
    for path in episode_files(rec.path):
        print(f"  {path.name}  ({path.stat().st_size / 1e6:.1f} MB)")

    print(
        "\nTurn these into a LeRobot dataset with:\n"
        f"  python h5_to_lerobot.py {rec.path}"
    )


if __name__ == "__main__":
    main()
