"""Live viewer for the Gen3 built-in vision module (streams over the arm's ethernet, not USB).

    python kinova_camera.py                 # color stream
    python kinova_camera.py depth           # depth stream, as uint16 numpy frames
    python kinova_camera.py --ip 192.168.1.12
    python kinova_camera.py --width 1280    # display size; frames are captured at 1280x720
                                            # regardless, and snapshots stay full res

Needs an opencv with GUI support -- `opencv-python-headless` has none, and if both it and
`opencv-python` are installed the headless one wins and imshow raises "not implemented".

Keys: [q] quit, [s] snapshot to data/camera_data/ (depth saves a 16-bit PNG + .npy)

The arm runs an RTSP server on port 554. Measured on this arm:
    rtsp://<ip>/color   1280x720 h264, reads at ~38fps
    rtsp://<ip>/depth   480x270 GRAY16_LE (Z16, in a gstreamer RTP payload). The caps
                        advertise 30fps but it actually delivers ~15.
Max 2 connections per stream, and the arm drops you after 30s of not reading, so don't
pause the loop.
"""
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from KinovaPy.camera import ColorStream, DepthStream

IP = sys.argv[sys.argv.index("--ip") + 1] if "--ip" in sys.argv else "192.168.1.10"
DEPTH = "depth" in sys.argv
# imshow costs ~47ms/frame at 1280x720 here (Qt backend, no OpenGL), which is slower than
# the 30fps source; 960 keeps the display loop comfortably ahead of the stream
WIDTH = int(sys.argv[sys.argv.index("--width") + 1]) if "--width" in sys.argv else 960
OUT_DIR = Path("data/camera_data")


def snapshot(frame, tag, count):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = OUT_DIR / f"kinova_{tag}_{count:06d}"
    if frame.dtype == np.uint16:
        # keep the raw units -- the 8-bit colormap on screen is only for looking at
        np.save(f"{stem}.npy", frame)
        cv2.imwrite(f"{stem}.png", frame)
    else:
        cv2.imwrite(f"{stem}.jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"wrote {stem}.*")


def colorize(depth):
    """uint16 depth -> BGR, scaled to the valid range of this frame."""
    valid = depth[depth > 0]
    if valid.size == 0:
        return np.zeros((*depth.shape, 3), np.uint8)
    # clip at a high percentile so a few saturated pixels don't flatten everything else
    lo, hi = valid.min(), np.percentile(valid, 95)
    scaled = np.clip((depth.astype(np.float32) - lo) / max(hi - lo, 1), 0, 1)
    out = cv2.applyColorMap((scaled * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    out[depth == 0] = 0  # no return -> black
    return out


frame_count = 0

if DEPTH:
    print(f"opening rtsp://{IP}/depth ...")
    stream = DepthStream(ip=IP)
    try:
        with stream:
            while True:
                depth = stream.read()
                if depth is None:
                    print("depth stream ended")
                    break
                frame_count += 1

                if frame_count == 1:
                    print(f"{depth.shape[1]}x{depth.shape[0]} {depth.dtype}")

                view = colorize(depth)
                # centre pixel in raw units, to sanity-check the scale against a tape measure
                cy, cx = depth.shape[0] // 2, depth.shape[1] // 2
                cv2.drawMarker(view, (cx, cy), (255, 255, 255), cv2.MARKER_CROSS, 12, 1)
                cv2.putText(view, f"centre={depth[cy, cx]}", (8, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                cv2.imshow("kinova depth", view)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("s"):
                    snapshot(depth, "depth", frame_count)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
    sys.exit()


print(f"opening rtsp://{IP}/color ...")
stream = ColorStream(ip=IP)
t_start = time.time()

try:
    with stream:
        while True:
            frame = stream.read()
            if frame is None:
                print("no frame for 5s -- stream stalled")
                break
            frame_count += 1

            if frame_count == 1:
                print(f"{frame.shape[1]}x{frame.shape[0]} {frame.dtype}, showing at {WIDTH}px")

            view = frame if WIDTH >= frame.shape[1] else cv2.resize(
                frame, (WIDTH, round(WIDTH * frame.shape[0] / frame.shape[1]))
            )
            cv2.imshow("kinova color", view)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                snapshot(frame, "color", frame_count)  # full res, not the display copy
except KeyboardInterrupt:
    pass
finally:
    cv2.destroyAllWindows()
    dt = time.time() - t_start
    print(f"shown {frame_count} frames in {dt:.1f}s ({frame_count / max(dt, 1e-9):.1f} fps), "
          f"{stream.frames_dropped} dropped to keep the socket drained")
