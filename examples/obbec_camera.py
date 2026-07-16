import cv2
import sys

cap = cv2.VideoCapture(0)
SAVEDATA = "savedata" in sys.argv


if not cap.isOpened():
    raise RuntimeError("Could not open webcam. Try changing VideoCapture(0) to VideoCapture(1).")

frame_count = 0
while True:

    success, frame = cap.read() #reads each frame
    frame_count += .125

    if frame_count.is_integer():
        cv2.imwrite(f"data/camera_data/camera_data_{frame_count}.jpg",frame)
    if not success: # if the frame isn't being read breaks out of while loop
        break

    frame = cv2.flip(frame, 1) # flips frame
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    #detect yellow
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    red_orange_yellow_1 = (0, 100, 100)
    red_orange_yellow_2 = (35, 255, 255)
    after_green_1 = (90, 100, 100)
    after_green_2 = (179, 255, 255)


    mask1 = cv2.inRange(hsv, red_orange_yellow_1, red_orange_yellow_2)
    mask2 = cv2.inRange(hsv, after_green_1, after_green_2)

    mask = cv2.bitwise_or(mask1, mask2)

    result = cv2.bitwise_and(frame, frame, mask=mask)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(frame, [largest], -1, (0, 255, 0), 2)  # draws onto frame
        cv2.imshow("frame", frame)                                # shows frame, now with contour drawn on it

    #if cv2.contourArea(largest) > 500:  # ignore tiny specks

    if cv2.waitKey(1) & 0xFF == ord("q"): #quits frame if q is pressed
        break



cap.release()
cv2.destroyAllWindows()
