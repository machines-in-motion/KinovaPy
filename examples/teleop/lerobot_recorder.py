# """A simple LeRobot dataset recorder.

# Record robot demonstrations into a standard LeRobot dataset without having to
# learn the LeRobot dataset API.

#     from lerobot_recorder import Recorder

#     rec = Recorder("cube_picking", fps=30, task="pick up the red cube")
#     rec.add_camera("wrist", height=480, width=640)
#     rec.add_state("arm", 6)
#     rec.add_action("arm_target", 6)

#     rec.start_episode()
#     while robot.is_running():
#         rec.log("wrist", camera.read())
#         rec.log("arm", robot.joint_positions())
#         rec.log("arm_target", controller.target())
#         rec.commit()            # <- stores everything logged as ONE sample
#     rec.stop_episode()

#     rec.close()

# The three rules:

# 1. ``log(name, value)`` remembers a value.  Logging the same name twice in one
#    loop keeps only the last value.
# 2. ``commit()`` turns everything logged since the last commit into one sample.
#    Every field you declared must have been logged, or you get an error.
# 3. ``close()`` finishes the dataset.  Without it the dataset is incomplete.

# Values may be Python lists, numbers, numpy arrays, torch tensors or PIL images.
# They are converted for you.
# """

from __future__ import annotations

import atexit
import contextlib
from pathlib import Path

import numpy as np

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
except ImportError as exc:  # pragma: no cover - depends on user's environment
    raise ImportError(
        "lerobot with dataset support is required.\n"
        "Install it with:  pip install 'lerobot[dataset]'"
    ) from exc


__all__ = ["Recorder", "RecorderError"]


# Columns LeRobot fills in by itself.  Students must not use these as field names.
_RESERVED = {"timestamp", "frame_index", "episode_index", "index", "task_index", "task"}

_STATE_COLUMN = "observation.state"
_ACTION_COLUMN = "action"
_IMAGE_PREFIX = "observation.images."


class RecorderError(ValueError):
    """Raised when the recorder is used incorrectly.

    Subclasses ``ValueError``, so ``except ValueError`` also catches it.
    """


# --------------------------------------------------------------------------- #
# Value conversion helpers
# --------------------------------------------------------------------------- #

def _to_numpy(value, field: str):
    """Convert lists / numbers / torch tensors / PIL images to a numpy array."""
    # torch tensor -- detected without importing torch, so torch stays optional.
    if hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "numpy"):
        value = value.detach().cpu().numpy()

    # PIL image
    elif value.__class__.__module__.startswith("PIL."):
        value = np.asarray(value)

    try:
        array = np.asarray(value)
    except Exception as exc:
        raise RecorderError(
            f"Could not read the value logged for '{field}'. "
            f"Expected a number, list, numpy array, torch tensor or PIL image, "
            f"but got {type(value).__name__}."
        ) from exc

    if array.dtype == object:
        raise RecorderError(
            f"The value logged for '{field}' is not a plain array of numbers. "
            f"If you are logging a list, make sure it is not ragged "
            f"(every row must have the same length)."
        )
    return array


def _as_vector(value, field: str, dim: int) -> np.ndarray:
    """Convert a logged value into a flat float32 vector of length ``dim``."""
    array = _to_numpy(value, field)

    if array.ndim > 1:
        # Tolerate the common (1, N) / (N, 1) shapes from robot SDKs.
        if array.size != dim:
            raise RecorderError(
                f"'{field}' expects {dim} number(s), but the value logged has "
                f"shape {tuple(array.shape)} ({array.size} numbers)."
            )
        array = array.reshape(-1)

    array = np.atleast_1d(array).astype(np.float32, copy=False).reshape(-1)

    if array.size != dim:
        raise RecorderError(
            f"'{field}' expects {dim} number(s), but {array.size} were logged."
        )
    if not np.isfinite(array).all():
        raise RecorderError(
            f"'{field}' was logged with NaN or infinity. "
            f"Check the sensor or controller producing this value."
        )
    return array


def _as_image(value, field: str, height: int, width: int, channels: int) -> np.ndarray:
    # """Convert a logged value into an (H, W, C) uint8 image."""
    array = _to_numpy(value, field)

    if array.ndim != 3:
        raise RecorderError(
            f"'{field}' expects an image of shape ({height}, {width}, {channels}), "
            f"but the value logged has shape {tuple(array.shape)}."
        )

    # Accept channel-first (C, H, W) as produced by torchvision and many models.
    if array.shape[0] in (1, 3, 4) and array.shape != (height, width, channels):
        if array.shape[1:] == (height, width):
            array = np.transpose(array, (1, 2, 0))

    # Drop an alpha channel if the camera produced RGBA.
    if channels == 3 and array.shape[-1] == 4:
        array = array[..., :3]

    # Floats are assumed to be 0..1 if they stay in that range, else 0..255.
    if np.issubdtype(array.dtype, np.floating):
        scale = 255.0 if float(np.nanmax(array, initial=0.0)) <= 1.0 else 1.0
        array = np.clip(array * scale, 0, 255)
    elif array.dtype != np.uint8:
        array = np.clip(array, 0, 255)

    array = np.ascontiguousarray(array, dtype=np.uint8)

    if array.shape != (height, width, channels):
        raise RecorderError(
            f"'{field}' expects an image of shape ({height}, {width}, {channels}), "
            f"but got {tuple(array.shape)}. "
            f"Check the camera resolution you passed to add_camera()."
        )
    return array


# --------------------------------------------------------------------------- #
# Field bookkeeping
# --------------------------------------------------------------------------- #

def _split_names(names, total: int) -> dict[str, int]:
    # """Recover ``{part_name: dim}`` from a LeRobot feature ``names`` list.

    # ``["arm.0", ..., "arm.5", "gripper.0"]`` becomes ``{"arm": 6, "gripper": 1}``.
    # Datasets recorded by other tools may use other conventions; anything we
    # cannot split is treated as one part per entry, which is still usable.
    # """
    if not names:
        return {}
    parts: dict[str, int] = {}
    for entry in names:
        prefix = entry.rsplit(".", 1)[0] if "." in entry else entry
        parts[prefix] = parts.get(prefix, 0) + 1
    return parts if sum(parts.values()) == total else {}


class _Part:
    # """One named piece of the state or action vector."""

    __slots__ = ("name", "dim", "start")

    def __init__(self, name: str, dim: int, start: int):
        self.name = name
        self.dim = dim
        self.start = start


class _Camera:
    __slots__ = ("name", "column", "height", "width", "channels")

    def __init__(self, name: str, column: str, height: int, width: int, channels: int):
        self.name = name
        self.column = column
        self.height = height
        self.width = width
        self.channels = channels


# --------------------------------------------------------------------------- #
# The recorder
# --------------------------------------------------------------------------- #

class Recorder:
    """Records episodes into a LeRobot dataset.

    Args:
        name: Dataset name, e.g. ``"cube_picking"``.  Also the folder name.
        fps: Control loop rate in Hz.  Used to timestamp samples.
        task: Default natural-language instruction stored with every sample,
            e.g. ``"pick up the red cube"``.  Can be overridden per episode.
        root: Folder that holds datasets.  Defaults to ``./datasets``.
        owner: Hugging Face style owner used to build the dataset id.
        use_videos: Store camera streams as MP4 video (small) instead of PNG
            images (large but simpler to inspect).
        verbose: Print what the recorder is doing.

    A dataset that already exists on disk is **resumed**: new episodes are
    appended to it.  When resuming, you may re-declare the fields exactly as
    before (this has no effect) or declare nothing at all and inherit them.
    Declaring a field the existing dataset does not have is an error.
    """

    def __init__(
        self,
        name: str,
        fps: int,
        task: str | None = None,
        root: str | Path = "datasets",
        owner: str = "student",
        use_videos: bool = True,
        verbose: bool = True,
    ):
        if "/" in name:
            owner, name = name.split("/", 1)
        if not name:
            raise RecorderError("The dataset needs a name, e.g. Recorder('cube_picking', fps=30).")
        if int(fps) <= 0:
            raise RecorderError(f"fps must be a positive whole number, got {fps!r}.")

        self.name = name
        self.fps = int(fps)
        self.task = task
        self.repo_id = f"{owner}/{name}"
        self.path = Path(root).expanduser().resolve() / name
        self.use_videos = bool(use_videos)
        self.verbose = bool(verbose)

        self._cameras: dict[str, _Camera] = {}
        self._state: dict[str, _Part] = {}
        self._action: dict[str, _Part] = {}
        self._pending: dict[str, object] = {}

        self._dataset: LeRobotDataset | None = None
        self._recording = False
        self._episode_task: str | None = None
        self._episode_frames = 0
        self._closed = False

        # Is there a dataset here already?
        self.resumed = (self.path / "meta" / "info.json").is_file()
        self._existing: dict | None = None
        if self.resumed:
            self._load_existing()

        atexit.register(self._at_exit)

    # -- declaring fields --------------------------------------------------- #

    def add_camera(self, name: str, height: int, width: int, channels: int = 3) -> "Recorder":
        # """Declare a camera.  Stored as ``observation.images.<name>``."""
        self._check_new_name(name)
        for size, label in ((height, "height"), (width, "width"), (channels, "channels")):
            if int(size) <= 0:
                raise RecorderError(f"Camera '{name}' has an invalid {label}: {size!r}.")
        camera = _Camera(name, _IMAGE_PREFIX + name, int(height), int(width), int(channels))

        if self.resumed:
            self._check_resumed_camera(camera)
            return self

        self._cameras[name] = camera
        return self

    def add_state(self, name: str, dim: int = 1) -> "Recorder":
        """Declare part of the robot's observed state.

        All state parts are joined, in declaration order, into LeRobot's single
        ``observation.state`` vector.
        """
        return self._add_part(name, dim, self._state, "state")

    def add_action(self, name: str, dim: int = 1) -> "Recorder":
        # """Declare part of the action / command sent to the robot.

        # All action parts are joined, in declaration order, into LeRobot's single
        # ``action`` vector.
        # """
        return self._add_part(name, dim, self._action, "action")

    def _add_part(self, name: str, dim: int, group: dict, kind: str) -> "Recorder":
        self._check_new_name(name)
        if "." in name:
            raise RecorderError(
                f"Field names cannot contain a dot: {name!r}. Try {name.replace('.', '_')!r}."
            )
        if int(dim) <= 0:
            raise RecorderError(f"'{name}' needs a dim of at least 1, got {dim!r}.")

        if self.resumed:
            self._check_resumed_part(name, int(dim), kind)
            return self

        start = sum(part.dim for part in group.values())
        group[name] = _Part(name, int(dim), start)
        return self

    def _check_new_name(self, name: str) -> None:
        if self._dataset is not None:
            raise RecorderError(
                # "Fields must be declared before the first start_episode()."
            )
        if not isinstance(name, str) or not name:
            raise RecorderError(f"Field names must be non-empty strings, got {name!r}.")
        if "/" in name:
            raise RecorderError(f"Field names cannot contain '/': {name!r}.")
        if name in _RESERVED:
            raise RecorderError(
                f"'{name}' is reserved by LeRobot and filled in automatically. "
                f"Please pick another name."
            )
        # When resuming, the fields are already loaded from disk and re-declaring
        # them is allowed (and checked against the dataset instead).
        if not self.resumed and name in self._known_names():
            raise RecorderError(
                f"'{name}' is already declared. Every field needs its own name, so that "
                f"log('{name}', ...) is unambiguous. For example, use "
                f"add_state('{name}', ...) together with add_action('{name}_target', ...)."
            )

    def _known_names(self) -> set[str]:
        return set(self._cameras) | set(self._state) | set(self._action)

    # -- resume checks ------------------------------------------------------ #

    def _load_existing(self) -> None:
        # """Read the fields of the dataset we are about to resume."""
        meta = LeRobotDatasetMetadataProbe(self.path)
        self._existing = meta.summary

        for column, spec in meta.features.items():
            if column.startswith(_IMAGE_PREFIX):
                shape = tuple(spec.get("shape") or ())
                if len(shape) != 3:
                    continue
                short = column[len(_IMAGE_PREFIX):]
                self._cameras[short] = _Camera(short, column, *(int(s) for s in shape))
            elif column in (_STATE_COLUMN, _ACTION_COLUMN):
                group = self._state if column == _STATE_COLUMN else self._action
                total = int((spec.get("shape") or [0])[0])
                parts = _split_names(spec.get("names"), total)
                if not parts:
                    parts = {"state" if column == _STATE_COLUMN else "action": total}
                start = 0
                for part_name, dim in parts.items():
                    group[part_name] = _Part(part_name, dim, start)
                    start += dim

        if self.fps != meta.fps:
            if self.verbose:
                print(
                    f"[recorder] note: '{self.name}' was recorded at {meta.fps} fps; "
                    f"keeping {meta.fps} fps so the episodes stay consistent."
                )
            self.fps = meta.fps

        if self.verbose:
            print(
                f"[recorder] resuming '{self.name}' "
                f"({meta.summary['episodes']} episodes, {meta.summary['frames']} frames) "
                f"at {self.path}"
            )
            print(f"[recorder] fields to log each step: {', '.join(self.fields) or '(none)'}")

    def _check_resumed_camera(self, camera: _Camera) -> None:
        found = self._cameras.get(camera.name)
        if found is None:
            raise RecorderError(self._missing_field_message(camera.name, "camera"))
        expected = (found.height, found.width, found.channels)
        given = (camera.height, camera.width, camera.channels)
        if expected != given:
            raise RecorderError(
                f"Camera '{camera.name}' in the existing dataset '{self.name}' is "
                f"{expected[0]}x{expected[1]} with {expected[2]} channel(s), but you declared "
                f"{given[0]}x{given[1]} with {given[2]}. Resumed datasets cannot change shape.\n"
                f"Either declare it the same way, or record into a new dataset name."
            )

    def _check_resumed_part(self, name: str, dim: int, kind: str) -> None:
        group = self._state if kind == "state" else self._action
        found = group.get(name)
        if found is None:
            raise RecorderError(self._missing_field_message(name, kind))
        if found.dim != dim:
            raise RecorderError(
                f"The {kind} field '{name}' in the existing dataset '{self.name}' has "
                f"{found.dim} number(s), but you declared {dim}. "
                f"Resumed datasets cannot change shape.\n"
                f"Either declare it as add_{kind}('{name}', {found.dim}), "
                f"or record into a new dataset name."
            )

    def _missing_field_message(self, name: str, kind: str) -> str:
        return (
            f"The dataset '{self.name}' that you are resuming does not have a {kind} "
            f"field called '{name}'.\n"
            f"It has: {self._describe_fields()}.\n"
            f"You cannot add new fields to an existing dataset. Either drop this "
            f"add_{'camera' if kind == 'camera' else kind}() call, or choose a new "
            f"dataset name to start a fresh recording."
        )

    def _describe_fields(self) -> str:
        described = [f"{c.name} (camera {c.height}x{c.width})" for c in self._cameras.values()]
        described += [f"{p.name} (state, {p.dim})" for p in self._state.values()]
        described += [f"{p.name} (action, {p.dim})" for p in self._action.values()]
        return ", ".join(described) if described else "no fields"

    # -- what to log -------------------------------------------------------- #

    @property
    def fields(self) -> list[str]:
        # """Names that must be logged before every ``commit()``."""
        return list(self._cameras) + list(self._state) + list(self._action)

    def describe(self) -> str:
        """A human-readable summary of this dataset."""
        lines = [
            f"dataset : {self.name}  ({'resuming' if self.resumed else 'new'})",
            f"folder  : {self.path}",
            f"fps     : {self.fps}",
            f"task    : {self.task!r}",
            f"fields  : {self._describe_fields()}",
        ]
        if self._state:
            lines.append(f"  -> observation.state is {self._total(self._state)} numbers")
        if self._action:
            lines.append(f"  -> action is {self._total(self._action)} numbers")
        if self._existing:
            lines.append(
                f"already recorded: {self._existing['episodes']} episodes, "
                f"{self._existing['frames']} frames"
            )
        return "\n".join(lines)

    @staticmethod
    def _total(group: dict[str, _Part]) -> int:
        return sum(part.dim for part in group.values())

    # -- recording ---------------------------------------------------------- #

    def start_episode(self, task: str | None = None) -> "Recorder":
        # """Begin a new episode.

        # Args:
        #     task: Instruction for this episode.  Defaults to the ``task`` given
        #         when the Recorder was built.
        # """
        self._require_open()
        print("lerobot is working")
        if self._recording:
            raise RecorderError(
                "An episode is already being recorded. Call stop_episode() first."
            )

        chosen = task if task is not None else self.task
        if not chosen:
            raise RecorderError(
                "Every episode needs a task description, e.g.\n"
                "  Recorder('my_data', fps=30, task='pick up the red cube')\n"
                "or\n"
                "  rec.start_episode(task='pick up the red cube')"
            )

        if self._dataset is None:
            self._open_dataset()

        self._episode_task = chosen
        self._episode_frames = 0
        self._pending.clear()
        self._recording = True
        if self.verbose:
            print(f"[recorder] recording episode {self.episode_count} - task: {chosen!r}")
        return self

    def log(self, name: str, value) -> "Recorder":
        # """Remember one value for the sample being built.

        # Logging the same name twice before ``commit()`` keeps only the last
        # value.  The value is checked immediately, so mistakes point at the line
        # that caused them.
        # """
        if not self._recording:
            raise RecorderError(
                f"log('{name}', ...) was called before start_episode(). "
                f"Call rec.start_episode() first."
            )

        if name in self._cameras:
            camera = self._cameras[name]
            self._pending[name] = _as_image(
                value, name, camera.height, camera.width, camera.channels
            )
        elif name in self._state:
            self._pending[name] = _as_vector(value, name, self._state[name].dim)
        elif name in self._action:
            self._pending[name] = _as_vector(value, name, self._action[name].dim)
        else:
            known = ", ".join(self.fields) or "none declared yet"
            raise RecorderError(
                f"'{name}' is not a field of this dataset. Declared fields are: {known}.\n"
                f"Declare it before recording with add_camera('{name}', ...), "
                f"add_state('{name}', ...) or add_action('{name}', ...)."
            )
        return self

    def commit(self) -> "Recorder":
        # """Store everything logged since the last commit as one sample."""
        if not self._recording:
            raise RecorderError(
                "commit() was called before start_episode(). Call rec.start_episode() first."
            )

        missing = [name for name in self.fields if name not in self._pending]
        if missing:
            raise RecorderError(
                f"Cannot commit this sample - these fields were never logged: "
                f"{', '.join(missing)}.\n"
                f"Every declared field must be logged once per loop, before commit()."
            )

        frame: dict[str, object] = {"task": self._episode_task}
        for camera in self._cameras.values():
            frame[camera.column] = self._pending[camera.name]
        if self._state:
            frame[_STATE_COLUMN] = self._join(self._state)
        if self._action:
            frame[_ACTION_COLUMN] = self._join(self._action)

        self._dataset.add_frame(frame)
        self._episode_frames += 1
        self._pending.clear()
        return self

    def _join(self, group: dict[str, _Part]) -> np.ndarray:
        return np.concatenate([self._pending[name] for name in group]).astype(np.float32)

    def stop_episode(self) -> "Recorder":
        # """Save the episode being recorded."""
        if not self._recording:
            raise RecorderError("stop_episode() was called without a matching start_episode().")

        self._recording = False
        self._pending.clear()

        if self._episode_frames == 0:
            self._dataset.clear_episode_buffer()
            if self.verbose:
                print("[recorder] episode had no samples - nothing saved.")
            return self

        self._dataset.save_episode()
        if self.verbose:
            seconds = self._episode_frames / self.fps
            print(
                f"[recorder] saved episode {self.episode_count - 1} "
                f"({self._episode_frames} samples, {seconds:.1f}s)"
            )
        return self

    def cancel_episode(self) -> "Recorder":
        # """Throw away the episode being recorded (for a bad take)."""
        if not self._recording:
            return self
        self._recording = False
        self._pending.clear()
        self._dataset.clear_episode_buffer()
        if self.verbose:
            print(f"[recorder] discarded episode ({self._episode_frames} samples).")
        self._episode_frames = 0
        return self

    @contextlib.contextmanager
    def episode(self, task: str | None = None):
        # """Record one episode using ``with``.

        #     with rec.episode():
        #         for _ in range(100):
        #             ...
        #             rec.commit()

        # The episode is saved when the block ends.  Pressing Ctrl-C also saves
        # it; any other error discards it.
        # """
        self.start_episode(task)
        try:
            yield self
        except KeyboardInterrupt:
            self.stop_episode()
            raise
        except BaseException:
            self.cancel_episode()
            raise
        else:
            self.stop_episode()

    # -- finishing ---------------------------------------------------------- #

    def close(self) -> "Recorder":
        # """Finish the dataset.  Required, or the dataset stays incomplete."""
        if self._closed:
            return self

        if self._recording:
            if self.verbose:
                print("[recorder] close() called while recording - saving the episode first.")
            self.stop_episode()

        if self._dataset is not None:
            self._dataset.finalize()
            if self.verbose:
                episodes, frames = self.episode_count, self._dataset.meta.total_frames
                print(
                    f"[recorder] closed '{self.name}': {episodes} episodes, "
                    f"{frames} samples at {self.path}"
                )

        self._closed = True
        with contextlib.suppress(Exception):
            atexit.unregister(self._at_exit)
        return self

    def _at_exit(self) -> None:
        # """Safety net so a forgotten close() does not destroy the dataset."""
        if self._closed or self._dataset is None:
            return
        with contextlib.suppress(Exception):
            print("[recorder] close() was never called - saving the dataset now.")
            self.close()

    def __enter__(self) -> "Recorder":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    # -- internals ---------------------------------------------------------- #

    @property
    def episode_count(self) -> int:
        # """Number of episodes saved in this dataset so far."""
        if self._dataset is not None:
            return self._dataset.meta.total_episodes
        return self._existing["episodes"] if self._existing else 0

    def _require_open(self) -> None:
        if self._closed:
            raise RecorderError(
                f"This Recorder is closed. Build a new Recorder('{self.name}', ...) "
                f"to add more episodes."
            )

    def _open_dataset(self) -> None:
        if not self.fields:
            raise RecorderError(
                "Declare at least one field before recording, for example:\n"
                "  rec.add_camera('wrist', height=480, width=640)\n"
                "  rec.add_state('arm', 6)\n"
                "  rec.add_action('arm_target', 6)"
            )

        threads = 4 if self._cameras else 0

        if self.resumed:
            self._dataset = LeRobotDataset.resume(
                repo_id=self.repo_id,
                root=self.path,
                image_writer_threads=threads,
            )
            return

        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            # An empty leftover folder is harmless; anything else is not ours.
            if any(self.path.iterdir()):
                raise RecorderError(
                    f"{self.path} already exists but is not a LeRobot dataset "
                    f"(no meta/info.json). Please move it away, or pick another name."
                )
            self.path.rmdir()

        self._dataset = LeRobotDataset.create(
            repo_id=self.repo_id,
            fps=self.fps,
            features=self._build_features(),
            root=self.path,
            use_videos=self.use_videos,
            image_writer_threads=threads,
        )
        if self.verbose:
            print(f"[recorder] created '{self.name}' at {self.path}")

    def _build_features(self) -> dict:
        features: dict[str, dict] = {}

        for camera in self._cameras.values():
            features[camera.column] = {
                "dtype": "video" if self.use_videos else "image",
                "shape": (camera.height, camera.width, camera.channels),
                "names": ["height", "width", "channels"],
            }

        for column, group in ((_STATE_COLUMN, self._state), (_ACTION_COLUMN, self._action)):
            if not group:
                continue
            # `names` records which slice of the vector belongs to which field,
            # so the parts can be recovered when the dataset is resumed or read.
            names = [f"{part.name}.{i}" for part in group.values() for i in range(part.dim)]
            features[column] = {
                "dtype": "float32",
                "shape": (self._total(group),),
                "names": names,
            }

        return features

    def __repr__(self) -> str:
        state = "closed" if self._closed else ("recording" if self._recording else "ready")
        return (
            f"Recorder(name={self.name!r}, fps={self.fps}, "
            f"episodes={self.episode_count}, fields={self.fields}, {state})"
        )


class LeRobotDatasetMetadataProbe:
    # """Reads ``meta/info.json`` directly.

    # Used to inspect a dataset before deciding how to open it, without paying
    # for a full LeRobot metadata load.
    # """

    def __init__(self, path: Path):
        import json

        info = json.loads((path / "meta" / "info.json").read_text())
        self.features: dict = info.get("features", {})
        self.fps: int = int(info.get("fps", 30))
        self.summary = {
            "episodes": int(info.get("total_episodes", 0)),
            "frames": int(info.get("total_frames", 0)),
        }
