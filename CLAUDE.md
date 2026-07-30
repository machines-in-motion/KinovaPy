# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

KinovaPy is a Python package for controlling a Kinova Gen3 7-DOF robot arm, both in simulation (MuJoCo) and on real hardware (via Kinova's Kortex API). It wraps a separate, external MPC framework (`locompc`) with a Kinova-specific hardware interface, controller adapter, and a set of example scripts (reaching, floating/gravity-compensation, teleop via joystick/SpaceMouse, camera-based perception).

This package is not self-contained: several core imports (`locompc`, `kortex_api`, `pinocchio`, `mujoco`) come from sibling packages/SDKs that are expected to already be installed in the environment, not from this repo.

## Install

```
pip install -e .
```

There is no lint/test/build tooling configured in this repo (no pytest, no linter config, no CI). Verifying changes generally means running the relevant example script against MuJoCo sim (default) or the real arm (`real` arg, requires hardware on the network).

## Running examples

Example scripts are plain Python files (not argparse-based) that read custom flags out of `sys.argv` by membership check, e.g.:

```
python examples/reach/kinova_reach.py            # simulation
python examples/reach/kinova_reach.py real        # real hardware
python examples/reach/kinova_reach.py plot        # record + plot trajectories at the end
python examples/reach/kinova_reach.py savedata    # record + dump CSV to data/
python examples/reach/kinova_reach.py real plot savedata
```

Each example script is meant to be run with its directory as the CWD (it loads a same-named `.yml` config via `os.path.dirname(__file__)`, but writes output to a relative `data/` folder, so `cd` into the example's folder first). Scripts block on `input("Press [ENTER] to start...")` before actually commanding the robot — this is a deliberate safety gate, don't remove it.

Real-robot connection defaults to IP `192.168.1.10`, user/pass `admin`/`admin` (see `KinovaPy/utilities.py:parseConnectionArguments`); override with `--ip`/`-u`/`-p`.

## Architecture

**`KinovaPy/interface.py` — `KinovaHardwareInterface`** (aliased as `TorqueExample` for back-compat)
Low-level wrapper around the Kortex API (`kortex_api.autogen.client_stubs.*`). Owns the TCP (command) and UDP (real-time, 1kHz) routers, actuator config client, and cyclic base client. Responsibilities:
- Reads joint state (`get_robot_states`) in SI units (rad, rad/s, Nm), converting from the API's degrees.
- Publishes commands (`_publish_command`) at whatever `current_control_mode` is set (`TORQUE` / `POSITION` / `VELOCITY`), computing PD + feedforward torque itself in `TORQUE` mode.
- Runs a background thread (`start_command_stream`/`_command_loop`) that republishes the latest command at ~1kHz; the actual command is updated via `set_command` from the main control loop without blocking it.
- `move_to_home` either triggers Kinova's stored "Home" action (single-level servoing) or bang-bang position-controls to an arbitrary `q0` (low-level servoing) if one is given.
- Always call `stop_command_stream()` before switching modes or exiting — it restores POSITION mode and single-level servoing so the arm is left in a safe state.

**`KinovaPy/controller.py` — `KinovaMPC`**
Thin adapter between `locompc`'s `MaNMPC` solver and either a MuJoCo sim robot or a `KinovaHardwareInterface`. `mode='sim'|'real'` selects which state-reading path `get_states` takes. If `record=True`, every `update()` call appends state/solution/timing history to preallocated numpy arrays (`xs`, `us`, `x_des`, `u_des`, `x_all`, `u_all`, `sol_times`, `sol_stats`) — examples trim these to `controller.i` entries at the end and optionally plot (`KinovaPy/plot.py`) or CSV-dump them.

**`KinovaPy/utilities.py`**
`DeviceConnection` — context manager wrapping `RouterClient`/`SessionManager` setup for TCP (port 10000, command) or UDP (port 10001, real-time) connections to the arm. Examples open one of each and pass the underlying routers into `KinovaHardwareInterface`.

**`KinovaPy/utils/`**
Teleop input devices, vendored/adapted third-party code:
- `spacemouse.py` — HID-level driver (`easyhid`) for 3Dconnexion SpaceMouse devices, with per-device axis/button mappings.
- `joy.py` (`SpaceMouseExpert`) — wraps `pyspacemouse` in a background `multiprocessing.Process` and exposes `get_action()`/`close()` for polling 6-DOF deltas + buttons without blocking the control loop.

**`KinovaPy/assets/`**
Robot description resources: URDF + STL meshes (for Pinocchio kinematics/dynamics), a MuJoCo scene XML (`scene_kinova.xml`) + robot XML (for sim), packaged via `package_data` in `setup.py`. Exposed as path constants from `KinovaPy/__init__.py` (`MESHES_PATH`, `URDF_PATH`, `XML_PATH`, `SCENE_PATH`).

**`examples/`**
Each subfolder (`reach/`, `float/`, `move_tp_goal/`) pairs a driver script with a same-named `.yml` config (MPC weights, joint limits, initial poses, constraint/cost selection — see the `WHICH_CONSTRAINTS`/`WHICH_COSTS` lists in the YAML for what's actually active). The common script shape is: load YAML → build Pinocchio model from URDF → build `KinovaMPC` → branch on sim (MuJoCo `MjSim`) vs real (`KinovaHardwareInterface`) → warmstart the MPC → loop calling `controller.update()` and sending the clipped torque/position/velocity command → on exit, restore safe state and optionally plot/save.
- `reach/` drives the end effector to a single fixed target pose.
- `float/` holds/gravity-compensates around the current pose (`stay` arg) or continuously re-tracks the live pose each iteration (zero-impedance "floating" behavior).
- `move_tp_goal/` is a simpler PD-only (non-MPC) move-to-goal script.
- Loose top-level scripts (`obbec_camera.py`, `claw.py`, `hand.py`, `test_spacemouse.py`) are standalone perception/teleop utilities, not part of the MPC pipeline.

## Conventions specific to this codebase

- Joint angles are radians and wrapped to `[-pi, pi]` internally (`_wrap_angles` in `interface.py`); the Kortex API itself uses degrees, so conversion happens at the interface boundary — don't push degree values past `interface.py`.
- `sim`/`real` branches (`if not REAL: ... else: ...`) are the standard way examples share one control loop across MuJoCo and hardware; keep new examples consistent with this pattern rather than introducing a different abstraction.
- Control mode names (`'TORQUE'`, `'POSITION'`, `'VELOCITY'`) are matched case-insensitively against `ActuatorConfig_pb2.ControlMode` and always upper-cased before comparison/storage.
