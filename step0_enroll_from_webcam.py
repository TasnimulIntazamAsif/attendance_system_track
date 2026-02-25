import os, cv2, time
import numpy as np
import face_recognition

PERSON_NAME = "Asif"     # change this
CAMERA_INDEX = 0
SAVE_DIR = "known_faces"
TARGET_SAMPLES = 15      # 10-20 is great
MODEL = "hog"            # keep hog (fast)

os.makedirs(SAVE_DIR, exist_ok=True)

cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    raise RuntimeError("Camera not opened. Try CAMERA_INDEX=1")

print("[INFO] Look at the camera. Press SPACE to capture samples. Press Q to quit.")
count = 0

while True:
    ok, frame = cap.read()
    if not ok or frame is None:
        continue

    # detect face quickly on smaller frame
    small = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    rgb = np.require(rgb, dtype=np.uint8, requirements=["C"])

    locs = face_recognition.face_locations(rgb, model=MODEL)
    msg = f"Faces: {len(locs)} | Saved: {count}/{TARGET_SAMPLES}"
    cv2.putText(frame, msg, (20,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Enroll", frame)
    k = cv2.waitKey(1) & 0xFF

    if k in (ord('q'), ord('Q')):
        break

    # SPACE -> save ONLY if exactly 1 face found
    if k == 32:
        if len(locs) != 1:
            print("[WARN] Need exactly 1 face in frame. Try again.")
            continue
        count += 1
        out = os.path.join(SAVE_DIR, f"{PERSON_NAME}_{count:03d}.jpg")
        cv2.imwrite(out, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        print("[OK] Saved:", out)
        if count >= TARGET_SAMPLES:
            break

cap.release()
cv2.destroyAllWindows()
print("[DONE] Enrollment images captured.")