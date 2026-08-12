#!/usr/bin/env python
"""Is this computer ready? Run this first, and any time something stops working.

    python check_setup.py

Nothing here changes anything - it only looks and reports.
"""

import importlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
OK, WARN, BAD = "  ok ", " !!  ", "FAIL "
problems = []
warnings = []


def check(label, fn, needed_for="", hint=""):
    try:
        detail = fn()
    except Exception as exc:
        detail = f"{type(exc).__name__}: {str(exc)[:90]}"
        if needed_for == "later":
            warnings.append((label, detail, hint))
            print(f"{WARN}{label:22s} {detail}")
        else:
            problems.append((label, detail, hint))
            print(f"{BAD}{label:22s} {detail}")
        return False
    print(f"{OK}{label:22s} {detail}")
    return True


def version_of(module):
    def go():
        mod = importlib.import_module(module)
        return getattr(mod, "__version__", "installed")

    return go


print("\n=== the basics ===")
check("python", lambda: f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
check("conda environment", lambda: os.environ.get("CONDA_DEFAULT_ENV") or "(none - did you activate?)")
check("numpy", version_of("numpy"))
check("opencv", version_of("cv2"))
check("torch", version_of("torch"))
check("torchvision", version_of("torchvision"))


def gpu():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("no GPU visible - training will work but be slow")
    return torch.cuda.get_device_name(0)


check("gpu", gpu, needed_for="later", hint="training on the CPU takes hours instead of minutes")

print("\n=== for preparing the data ===")
check("lerobot", version_of("lerobot"), hint="only needed by prepare_data.py")

print("\n=== for driving the robot ===")
check(
    "pinocchio",
    version_of("pinocchio"),
    needed_for="later",
    hint="almost always the LD_LIBRARY_PATH problem - use `source activate_env.sh`",
)
check("locompc", version_of("locompc"), needed_for="later")
check("KinovaPy", version_of("KinovaPy"), needed_for="later")
check("pyrealsense2", version_of("pyrealsense2"), needed_for="later")
check("python-can", version_of("can"), needed_for="later")

print("\n=== hardware, right now ===")


def camera():
    import pyrealsense2 as rs

    names = [d.get_info(rs.camera_info.name) for d in rs.context().query_devices()]
    if not names:
        raise RuntimeError("no RealSense camera plugged in")
    return ", ".join(names)


def can_bus():
    out = subprocess.run(["ip", "-br", "link", "show"], capture_output=True, text=True).stdout
    for line in out.splitlines():
        if line.startswith("can0"):
            return line.split()[1]
    raise RuntimeError("can0 is not there - the gripper adapter is unplugged or not set up")


def arm_reachable():
    ip = "192.168.1.10"
    if shutil.which("ping") is None:
        raise RuntimeError("no ping command")
    done = subprocess.run(["ping", "-c1", "-W2", ip], capture_output=True)
    if done.returncode != 0:
        raise RuntimeError(f"cannot reach the arm at {ip} - is it on, and is the cable in?")
    return f"{ip} answers"


check("camera", camera, needed_for="later")
check("gripper bus (can0)", can_bus, needed_for="later")
check("arm (192.168.1.10)", arm_reachable, needed_for="later")

print("\n=== your files ===")


def prepared_data():
    import json

    candidates = sorted((HERE / "data").glob("*/meta.json")) if (HERE / "data").is_dir() else []
    if not candidates:
        raise RuntimeError("no prepared data yet - run prepare_data.py")
    meta = json.loads(candidates[0].read_text())
    return f"{candidates[0].parent.name}: {meta['num_frames']} moments, {meta['num_episodes']} takes"


def blocks():
    sys.path.insert(0, str(HERE))
    import robot_ai

    return "all blocks import cleanly"


check("robot_ai blocks", blocks)
check("prepared data", prepared_data, needed_for="later", hint="tutorial 00 tells you how")

print("\n" + "=" * 62)
if problems:
    print(f"{len(problems)} thing(s) must be fixed before you can do anything:")
    for label, detail, hint in problems:
        print(f"  - {label}: {detail}")
        if hint:
            print(f"      hint: {hint}")
elif warnings:
    print("Good enough to start training. Still missing, needed later:")
    for label, detail, hint in warnings:
        print(f"  - {label}: {detail}")
        if hint:
            print(f"      hint: {hint}")
    print("\nIf the robot, camera or gripper are simply switched off, that is fine for now.")
else:
    print("Everything is ready. Open tutorials/00_setup.md")
print("=" * 62 + "\n")
