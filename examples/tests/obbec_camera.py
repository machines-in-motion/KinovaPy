import cv2
import threading
import sys
import queue
from pathlib import Path
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("could not open /dev/video0")
#cv2.CAP_V4L2

cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)


SAVEDATA = "savedata" in sys.argv


# background writer so imwrite never blocks the capture loop
save_queue = queue.Queue()

def writer():
    while True:
        thing = save_queue.get()
        if thing is None:
            break
        path, img = thing
        cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 90])

writer_thread = threading.Thread(target=writer, daemon=True)
writer_thread.start()

frame_count = 0
OUT_DIR = Path("data/camera_data")

try:
    while True:
        success, frame = cap.read()
        if not success:
            print("unable to collect")
            continue
        frame_count += 1

        if SAVEDATA and frame_count % 8 == 0:
            save_queue.put((OUT_DIR / f"camera_data_{frame_count:06d}.jpg", frame.copy()))
        else:
            None

        cv2.imshow("frame", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            cap.release()
            break
except KeyboardInterrupt:
    pass
finally:
    cap.release()
    save_queue.put(None)
    writer_thread.join()
    
save_queue.put(None)
cap.release()
cv2.destroyAllWindows()