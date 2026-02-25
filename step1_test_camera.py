import cv2
import time
import numpy as np

CAMERA_INDEX = 0

cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("[FAIL] Webcam not opened. Try CAMERA_INDEX=1")
    raise SystemExit

print("[OK] Webcam opened. Press Q to quit.")
time.sleep(0.5)

while True:
    ok, frame = cap.read()
    if not ok or frame is None:
        print("[WARN] Failed to grab frame")
        continue

    print("[DEBUG] frame shape:", frame.shape, "dtype:", frame.dtype, "mean:", np.mean(frame))
    cv2.imshow("Camera Test", frame)

    if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q')):
        break

cap.release()
cv2.destroyAllWindows()
print("[DONE] Camera test finished.")