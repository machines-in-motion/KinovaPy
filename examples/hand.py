import os
import urllib.request

import cv2

#yolo
BASE_URL = "https://github.com/cansik/yolo-hand-detection/releases/download/pretrained/"
CFG_FILE = "cross-hands-yolov4-tiny.cfg"
WEIGHTS_FILE = "cross-hands-yolov4-tiny.weights"
MODEL_DIR = "models"
 
os.makedirs(MODEL_DIR, exist_ok=True)
cfg_path = os.path.join(MODEL_DIR, CFG_FILE)
weights_path = os.path.join(MODEL_DIR, WEIGHTS_FILE)
 
for fname, fpath in [(CFG_FILE, cfg_path), (WEIGHTS_FILE, weights_path)]:
    if not os.path.exists(fpath):
        print(f"Downloading {fname} ...")
        urllib.request.urlretrieve(BASE_URL + fname, fpath)
 
# ---- Load network ----
net = cv2.dnn.readNetFromTFLite(cfg_path, weights_path)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
 
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
 
CONF_THRESHOLD = 0.4
NMS_THRESHOLD = 0.3
INPUT_SIZE = 416  # try 256 for more speed, less accuracy
 
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam. Try changing VideoCapture(0) to VideoCapture(1).")
 
while True:
    success, frame = cap.read()
    if not success:
        break
 
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
 
    blob = cv2.dnn.blobFromImage(
        frame, 1 / 255.0, (INPUT_SIZE, INPUT_SIZE), swapRB=True, crop=False
    )
    net.setInput(blob)
    outputs = net.forward(output_layers)
 
    boxes, confidences = [], []
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            confidence = scores.max()
            if confidence > CONF_THRESHOLD:
                cx, cy, bw, bh = detection[0:4] * [w, h, w, h]
                x = int(cx - bw / 2)
                y = int(cy - bh / 2)
                boxes.append([x, y, int(bw), int(bh)])
                confidences.append(float(confidence))
 
    indices = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRESHOLD, NMS_THRESHOLD)
 
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, bw, bh = boxes[i]
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
            label = f"hand {confidences[i]:.2f}"
            cv2.putText(
                frame, label, (x, max(y - 10, 0)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
            )
 
    cv2.imshow("Hand Detection (press 'q' to quit)", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
 
cap.release()
cv2.destroyAllWindows()
 
