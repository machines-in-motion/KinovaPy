"""Readers for the Gen3 built-in vision module, which streams over the arm's ethernet.

Color is plain H.264 and goes through cv2. Depth is 16-bit Z16 wrapped in a gstreamer RTP
payload, which cv2.VideoCapture cannot decode on any build -- so DepthStream shells out to
gst-launch-1.0 and reads raw frames back over a pipe. That avoids depending on the gobject
introspection bindings, which are installed for the system python but not for conda.

Both streams cap at two simultaneous connections and the arm disconnects a client that
stops reading for 30s.
"""
import os
import subprocess
import threading
import time

import numpy as np

DEFAULT_IP = "192.168.1.10"

# as published by the vision module
COLOR_SIZE = (1280, 720)
DEPTH_SIZE = (480, 270)


class DepthStream:
    """Yields (h, w) uint16 depth frames from rtsp://<ip>/depth.

    The pixel values are raw Z16 sensor units, not metres -- see `read()`.
    """

    def __init__(self, ip=DEFAULT_IP, size=DEPTH_SIZE, latency=30):
        self.width, self.height = size
        self._nbytes = self.width * self.height * 2
        # pinning width/height/format in the capsfilter makes gstreamer fail loudly if the
        # module ever publishes something else, instead of us silently misreading the pipe
        self._cmd = [
            "gst-launch-1.0", "-q",
            "rtspsrc", f"location=rtsp://{ip}/depth", f"latency={latency}",
            "!", "rtpgstdepay",
            "!", "videoconvert",
            "!", f"video/x-raw,format=GRAY16_LE,width={self.width},height={self.height}",
            "!", "fdsink", "fd=1",
        ]
        self._proc = None

    def open(self):
        if self._proc is not None:
            return self
        self._proc = subprocess.Popen(
            self._cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        return self

    def read(self):
        """Return the next frame as (h, w) uint16, or None if the stream ended.

        Values are raw Z16 units with 0 meaning "no return". Confirm the scale against a
        known distance before treating them as millimetres.
        """
        if self._proc is None:
            self.open()

        buf = bytearray(self._nbytes)
        view = memoryview(buf)
        got = 0
        while got < self._nbytes:
            # a pipe read returns whatever is in the buffer, so loop until the frame is whole
            n = self._proc.stdout.readinto(view[got:])
            if not n:
                return None
            got += n

        return np.frombuffer(bytes(buf), dtype="<u2").reshape(self.height, self.width)

    def close(self):
        if self._proc is None:
            return
        self._proc.terminate()
        try:
            self._proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self._proc.kill()
        self._proc = None

    def __enter__(self):
        return self.open()

    def __exit__(self, *exc):
        self.close()


class ColorStream:
    """Drop-old reader for rtsp://<ip>/color, yielding (h, w, 3) uint8 BGR frames.

    A background thread does nothing but drain the socket, keeping only the newest frame.
    Reading straight from cv2 in the display loop instead is a trap: the arm sends a fixed
    30fps and anything slower on the consumer side (cv2.imshow costs ~47ms/frame on this
    machine) backs up the TCP receive queue until the arm's encoder overruns and drops
    H.264 packets, which shows up as corrupted macroblocks -- not as a lag you would notice.
    Dropping frames here is what keeps the wire healthy.
    """

    def __init__(self, ip=DEFAULT_IP, transport="tcp"):
        self._ip = ip
        self._transport = transport
        self._cap = None
        self._thread = None
        self._lock = threading.Lock()
        self._latest = None
        self._stop = threading.Event()
        self.frames_read = 0     # decoded off the wire
        self.frames_taken = 0    # actually handed to the caller

    @property
    def frames_dropped(self):
        return self.frames_read - self.frames_taken

    def open(self):
        if self._cap is not None:
            return self
        self._cap = open_color(self._ip, self._transport)
        self._thread = threading.Thread(target=self._drain, daemon=True)
        self._thread.start()
        return self

    def _drain(self):
        while not self._stop.is_set():
            ok, frame = self._cap.read()
            if not ok:
                continue
            with self._lock:
                self._latest = frame
                self.frames_read += 1

    def read(self, timeout=5.0):
        """Return the most recent frame, or None if none arrived within `timeout`."""
        if self._cap is None:
            self.open()
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self._lock:
                frame = self._latest
                self._latest = None
                if frame is not None:
                    self.frames_taken += 1
                    return frame
            time.sleep(0.001)
        return None

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self):
        return self.open()

    def __exit__(self, *exc):
        self.close()


def open_color(ip=DEFAULT_IP, transport="tcp"):
    """cv2.VideoCapture on rtsp://<ip>/color, yielding the usual BGR uint8 frames.

    RTSP defaults to UDP, which drops H.264 packets and smears frames, so we ask for TCP.
    That option is read when the capture is constructed, hence setting it here.
    """
    import cv2

    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = f"rtsp_transport;{transport}"
    cap = cv2.VideoCapture(f"rtsp://{ip}/color", cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise RuntimeError(
            f"could not open rtsp://{ip}/color -- check the arm is reachable and that "
            "fewer than two clients are already on the stream"
        )
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap
