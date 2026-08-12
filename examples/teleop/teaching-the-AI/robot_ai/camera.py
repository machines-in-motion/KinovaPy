"""The camera block: an Intel RealSense D435i that hands you one picture at a time.

Grabbing a frame takes a while, and the robot loop cannot afford to wait, so a
background thread pulls frames constantly and `get_image()` just takes whichever
one is newest. If nothing new has arrived since last time you get the previous
picture again - a slightly stale image is much better than a stalled robot.

Adapted from the teleop recording script (examples/teleop/worker.py), trimmed
down to what a policy needs.
"""

import threading
import time

import numpy as np

from . import vision

CAMERA_NAME_MATCH = "D435I"
OPEN_RETRIES = 10
RETRY_DELAY = 0.5


class Camera:
    """The robot's eye. Use it with `with`, so it always gets closed."""

    def __init__(self, dry_run=False):
        self.dry_run = dry_run
        self._pipeline = None
        self._thread = None
        self._running = False
        self._frame = None
        self._lock = threading.Lock()
        self.frames_seen = 0

    # --------------------------------------------------------------- lifetime

    def start(self):
        if self.dry_run:
            print("[camera] dry run: returning grey pictures instead of real ones")
            self._running = True
            return self
        import pyrealsense2 as rs

        serial = None
        for dev in rs.context().query_devices():
            if CAMERA_NAME_MATCH in dev.get_info(rs.camera_info.name).upper():
                serial = dev.get_info(rs.camera_info.serial_number)
                break
        if serial is None:
            raise RuntimeError(
                f"no {CAMERA_NAME_MATCH} camera found. Is it plugged in? Try `rs-enumerate-devices`."
            )

        last_error = None
        for attempt in range(1, OPEN_RETRIES + 1):
            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_device(serial)
            config.enable_stream(
                rs.stream.color, vision.CAMERA_W, vision.CAMERA_H, rs.format.bgr8, 30
            )
            try:
                pipeline.start(config)
                self._pipeline = pipeline
                break
            except RuntimeError as exc:
                last_error = exc
                del pipeline, config
                if "busy" in str(exc).lower():
                    raise RuntimeError(
                        "the camera is already open in another program.\n"
                        "  Most likely a teleop run that did not exit cleanly:\n"
                        "    pkill -f 'python.*final\\.py'"
                    ) from exc
                print(f"[camera] not ready (attempt {attempt}/{OPEN_RETRIES}), retrying...")
                time.sleep(RETRY_DELAY)
        if self._pipeline is None:
            raise RuntimeError(f"could not open the camera: {last_error}")

        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

        # Wait for the first picture so the caller never gets a None.
        deadline = time.time() + 5.0
        while self.get_raw() is None:
            if time.time() > deadline:
                raise RuntimeError("the camera opened but never produced a picture")
            time.sleep(0.05)
        print("[camera] ready")
        return self

    def _capture_loop(self):
        failures = 0
        while self._running:
            try:
                frames = self._pipeline.wait_for_frames(timeout_ms=1000)
                color = frames.get_color_frame()
            except RuntimeError:
                color = None
            if not color:
                failures += 1
                if failures >= 30:
                    print("[camera] stopped responding")
                    self._running = False
                time.sleep(0.05)
                continue
            failures = 0
            with self._lock:
                self._frame = np.asanyarray(color.get_data())
                self.frames_seen += 1

    def stop(self):
        self._running = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except RuntimeError:
                pass
            self._pipeline = None

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()
        return False

    # ------------------------------------------------------------------ frames

    def get_raw(self):
        """The newest camera frame, exactly as the sensor gave it (480x640 BGR)."""
        if self.dry_run:
            return np.full((vision.CAMERA_H, vision.CAMERA_W, 3), 128, dtype=np.uint8)
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def get_image(self):
        """The newest picture, prepared exactly the way the training data was."""
        raw = self.get_raw()
        if raw is None:
            return None
        return vision.from_camera(raw)

    @property
    def is_alive(self):
        return self._running
