import threading
import cv2
import sys
import queue


from pathlib import Path

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

if not cap.isOpened():
    print(f"[worker] warning: camera failed to open (VideoCapture({cap.get(cv2.CAP_PROP_POS_FRAMES)}))")

cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)



SAVEDATA = "savedata" in sys.argv
save_queue = queue.Queue()

frame_count = 0
OUT_DIR = Path("data/camera_data") #cd into teleop folder
OUT_DIR.mkdir(parents=True, exist_ok=True)

class Worker:
    def __init__(self, ):
        self.thread = threading.Thread(target=self.run)
        self.writer_thread = threading.Thread(target=self.write)
        self._running = True
        self.display_queue = queue.Queue(maxsize=1)


    def run(self):
        while self._running:
            success, frame = cap.read()
            if not success:
                print("unable to collect")
                continue
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

    def show(self):
        # cv2 GUI calls (imshow/waitKey) must run on the main thread,
        # so the capture thread just hands frames off through a queue.
        try:
            frame = self.display_queue.get_nowait()
        except queue.Empty:
            return
        cv2.imshow("frame", frame)
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
        if SAVEDATA:
            self.writer_thread.start()

    def stop(self):
        self._running = False
        cap.release()
        cv2.destroyAllWindows()
        if SAVEDATA:
            save_queue.put(None)



