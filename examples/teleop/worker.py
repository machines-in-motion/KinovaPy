import os
import threading
import time
import sys
import queue
import numpy as np
import cv2
import pyrealsense2 as rs

from PIL import Image
from pathlib import Path

# Match by product name substring rather than a fixed serial number, so this
# keeps working across reconnects or if the camera is swapped for another
# unit of the same model.
CAMERA_NAME_MATCH = "D435I"

# Opening the device right after a previous run released it can transiently
# fail for a moment while the driver settles, so retry a few times.
CAMERA_OPEN_RETRIES = 10
CAMERA_OPEN_RETRY_DELAY = 0.5


def _find_camera_serial():
    ctx = rs.context()
    for dev in ctx.query_devices():
        name = dev.get_info(rs.camera_info.name)
        if CAMERA_NAME_MATCH in name.upper():
            return dev.get_info(rs.camera_info.serial_number)
    return None


def _video_device_holders():
    # Which processes currently hold a /dev/video* node, by scanning /proc
    # rather than shelling out to lsof (not installed everywhere).
    holders = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            fds = list((proc / "fd").iterdir())
        except OSError:
            continue  # process exited, or not ours
        for fd in fds:
            try:
                target = os.readlink(fd)
            except OSError:
                continue
            # "(deleted)" nodes are stale handles on an unplugged device, not
            # something competing for the camera we want.
            if target.startswith("/dev/video") and "(deleted)" not in target:
                try:
                    cmd = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode()
                except OSError:
                    cmd = "?"
                cmd = " ".join(cmd.split())[:100]  # one line, keep the message readable
                holders.append((int(proc.name), cmd, target))
                break
    return holders


def _open_camera():
    serial = _find_camera_serial()
    if serial is None:
        raise RuntimeError(
            f"[worker] no camera found matching {CAMERA_NAME_MATCH!r}. "
            f"Is it plugged in? (check `rs-enumerate-devices` or `lsusb`)"
        )

    for attempt in range(1, CAMERA_OPEN_RETRIES + 1):
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(serial)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        try:
            pipeline.start(config)
            return pipeline
        except RuntimeError as e:
            # half-started pipeline still holds device handles; drop our
            # references now so librealsense tears them down before the next
            # attempt (and so they are not still around at interpreter exit,
            # where destroying them tends to segfault).
            del pipeline, config

            # EBUSY means another *process* owns the video node. No amount of
            # waiting will free it, so fail immediately and name the culprit
            # instead of burning ten retries on a hopeless case.
            if "busy" in str(e).lower():
                holders = "\n".join(
                    f"    pid {pid}  {dev}  {cmd}" for pid, cmd, dev in _video_device_holders()
                )
                raise RuntimeError(
                    f"[worker] camera {serial} is busy - another process already has it open.\n"
                    f"  Processes holding a video device:\n{holders or '    (none visible to this user)'}\n"
                    f"  Usually these are leftover runs; kill them with:\n"
                    f"    pkill -f 'python.*final\\.py'"
                ) from e

            print(f"[worker] camera not ready yet (attempt {attempt}/{CAMERA_OPEN_RETRIES}): {e}, retrying...")
            time.sleep(CAMERA_OPEN_RETRY_DELAY)

    raise RuntimeError(f"[worker] could not open camera {serial} after {CAMERA_OPEN_RETRIES} attempts")

SAVEDATA = "savedata" in sys.argv
save_queue = queue.Queue()

frame_count = 0
x = frame_count + 1
OUT_DIR = Path("data/camera_data/run_x") #cd into teleop folder
OUT_DIR.mkdir(parents=True, exist_ok=True)

class Worker:
    def __init__(self, ):
        # Both threads are daemons: if the main script dies before reaching
        # stop() (an exception in its own `finally` is enough), a non-daemon
        # thread would keep the interpreter alive forever with the camera
        # still open, and the next run would fail with "device busy".
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.latest_frame_stamp = None
        self.frame = None
        self.writer_thread = threading.Thread(target=self.write, daemon=True)
        self._running = True
        self.display_queue = queue.Queue(maxsize=1)
        self.pipeline = _open_camera()

    def run(self):
        consecutive_failures = 0
        while self._running:
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=1000)
                color_frame = frames.get_color_frame()
            except RuntimeError:
                color_frame = None
            if not color_frame:
                consecutive_failures += 1
                if consecutive_failures >= 30:
                    print("[worker] camera stopped responding, reopening...")
                    self.pipeline.stop()
                    self.pipeline = _open_camera()
                    consecutive_failures = 0
                else:
                    time.sleep(0.05)
                continue
            consecutive_failures = 0
            frame = np.asanyarray(color_frame.get_data())
            global frame_count
            frame_count += 1

            if SAVEDATA and frame_count % 8 == 0:
                save_queue.put((OUT_DIR / f"camera_data_{frame_count:06d}.jpg", frame.copy()))
            else:
                None

            if self.display_queue.full():
                try:
                    self.display_queue.get_nowait()
                except queue.Empty:
                    pass
            self.display_queue.put_nowait(frame)

    def get_frame(self):
        # Pull the newest frame off queue without blocking control loop.
        # If camera hasn't produced new one since the last call,
        # previous frame is returned (None until the first one arrives).
        try:
            self.frame = self.display_queue.get_nowait()
            self.latest_frame_stamp = time.time()
        except queue.Empty:
            pass
        return self.frame

    def show(self):
        # cv2 GUI calls (imshow/waitKey) has to run on main thread,
        # the capture thread just hands frames off through queue.
        if self.get_frame() is None:
            return
        cv2.imshow("frame", self.frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            self.stop()

    def write(self):
        while True:
            item = save_queue.get()
            if item is None:
                break
            path, frame = item
            cv2.imwrite(str(path), frame)

    def start(self):
        self.thread.start()
        print("worker is working")
        if SAVEDATA:
            self.writer_thread.start()

    def stop(self):
        if not self._running:
            return  # already stopped; stopping the pipeline twice throws
        self._running = False

        # Let the capture thread leave wait_for_frames before the pipeline goes
        # away underneath it - stopping a pipeline another thread is blocked on
        # crashes librealsense. The timeout is one wait_for_frames period plus
        # slack, so a wedged camera cannot hang shutdown.
        if self.thread.is_alive():
            self.thread.join(timeout=2.0)

        try:
            self.pipeline.stop()
        except RuntimeError as e:
            print(f"[worker] pipeline already stopped: {e}")
        cv2.destroyAllWindows()
        if SAVEDATA and self.writer_thread.is_alive():
            save_queue.put(None)  # sentinel: unblocks write()'s get()
            self.writer_thread.join(timeout=5.0)



