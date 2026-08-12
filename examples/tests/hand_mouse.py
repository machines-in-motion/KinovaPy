"""
Hand-tracking mouse: points the OS cursor at your index fingertip and
clicks/drags on a thumb pinch, using the Orbbec camera's color stream
(accessed as a V4L2/UVC device, same as examples/tests/obbec_camera.py).

Run:
    python examples/tests/hand_mouse.py

Controls (focus must be on the preview window):
    c        toggle mouse control on/off (starts OFF — see note below)
    q / ESC  quit

Gestures (only active while control is toggled on):
    move index fingertip           -> moves the cursor
    pinch thumb + index tip        -> left button down (release pinch to let go / click)
    pinch thumb + middle fingertip -> right click

Control starts OFF on purpose: the cursor is your real system cursor, so
raising your hand in front of the camera before you're ready would hijack
whatever you're doing. Press 'c' once the preview window has focus and your
hand is framed to engage it, and 'c' again to hand control back.
"""

import time
import urllib.request
from pathlib import Path

import cv2
from pynput.mouse import Button, Controller

import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

try:
    import tkinter

    _root = tkinter.Tk()
    _root.withdraw()
    SCREEN_W, SCREEN_H = _root.winfo_screenwidth(), _root.winfo_screenheight()
    _root.destroy()
except Exception:
    SCREEN_W, SCREEN_H = 1920, 1080

MODEL_DIR = Path(__file__).parent / "models"
MODEL_PATH = MODEL_DIR / "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)

# fraction of the camera frame (centered) that maps to the full screen;
# keeping this well inside [0, 1] means you don't have to reach the frame edges
ACTIVE_BOX = (0.15, 0.85, 0.15, 0.85)  # x_min, x_max, y_min, y_max, normalized
SMOOTHING = 0.4  # 0 = no smoothing, higher = smoother but laggier
PINCH_ON = 0.05  # normalized distance to engage a pinch
PINCH_OFF = 0.07  # normalized distance to release it (hysteresis)

THUMB_TIP, INDEX_TIP, MIDDLE_TIP = 4, 8, 12


def ensure_model():
    if MODEL_PATH.exists():
        return
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"downloading hand landmark model to {MODEL_PATH} ...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)


def dist(a, b):
    return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5


def to_screen(nx, ny):
    x_min, x_max, y_min, y_max = ACTIVE_BOX
    sx = (nx - x_min) / (x_max - x_min)
    sy = (ny - y_min) / (y_max - y_min)
    sx = min(max(sx, 0.0), 1.0)
    sy = min(max(sy, 0.0), 1.0)
    return sx * SCREEN_W, sy * SCREEN_H


def main():
    ensure_model()

    # matches the V4L2/UVC access pattern used elsewhere in this repo
    # (obbec_camera.py, teleop/worker.py) — the Orbbec color stream is
    # /dev/video1; raw pyorbbecsdk USB access needs udev rules this
    # machine doesn't have installed.
    cap = cv2.VideoCapture(1, cv2.CAP_V4L2)
    if not cap.isOpened():
        print("Error opening Orbbec camera at /dev/video1")
        return
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    landmarker = vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.6,
            min_tracking_confidence=0.5,
        )
    )

    mouse = Controller()
    control_on = False
    left_pinching = False
    right_pinching = False
    cursor = None  # smoothed (x, y) in screen space
    start_t = time.monotonic()

    print("Preview window opening. Press 'c' to engage mouse control, 'q' to quit.")

    try:
        while True:
            success, frame = cap.read()
            if not success:
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int((time.monotonic() - start_t) * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.hand_landmarks:
                lm = result.hand_landmarks[0]
                thumb, index, middle = lm[THUMB_TIP], lm[INDEX_TIP], lm[MIDDLE_TIP]

                target = to_screen(index.x, index.y)
                cursor = target if cursor is None else (
                    cursor[0] + SMOOTHING * (target[0] - cursor[0]),
                    cursor[1] + SMOOTHING * (target[1] - cursor[1]),
                )

                left_dist = dist(thumb, index)
                right_dist = dist(thumb, middle)

                if control_on:
                    mouse.position = (int(cursor[0]), int(cursor[1]))

                    if not left_pinching and left_dist < PINCH_ON:
                        left_pinching = True
                        mouse.press(Button.left)
                    elif left_pinching and left_dist > PINCH_OFF:
                        left_pinching = False
                        mouse.release(Button.left)

                    if not right_pinching and right_dist < PINCH_ON:
                        right_pinching = True
                        mouse.click(Button.right)
                    elif right_pinching and right_dist > PINCH_OFF:
                        right_pinching = False

                h, w = frame.shape[:2]
                for point in (thumb, index, middle):
                    cv2.circle(frame, (int(point.x * w), int(point.y * h)), 8, (0, 255, 0), -1)
            else:
                if control_on and left_pinching:
                    mouse.release(Button.left)
                left_pinching = False
                right_pinching = False

            status = f"CONTROL {'ON' if control_on else 'OFF'} (press c to toggle)"
            color = (0, 200, 0) if control_on else (0, 0, 200)
            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.imshow("Hand Mouse", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("c"):
                control_on = not control_on
                if not control_on and left_pinching:
                    mouse.release(Button.left)
                    left_pinching = False
    finally:
        if left_pinching:
            mouse.release(Button.left)
        landmarker.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
