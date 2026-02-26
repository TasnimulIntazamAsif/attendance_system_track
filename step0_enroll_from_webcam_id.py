import os
import cv2
import numpy as np
import face_recognition

# =========================
# SETTINGS
# =========================
CAMERA_INDEX = 0
SAVE_DIR = "known_faces"
TARGET_SAMPLES = 15
MODEL = "hog"
JPEG_QUALITY = 95

os.makedirs(SAVE_DIR, exist_ok=True)

# =========================
# INPUT
# =========================
PERSON_ID = input("Enter ID (e.g., 570508): ").strip()
PERSON_NAME = input("Enter Name (e.g., Asif): ").strip()

if not PERSON_ID or not PERSON_NAME:
    raise SystemExit("[FAIL] ID and Name both required!")

# =========================
# CAMERA
# =========================
cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    raise RuntimeError("Camera not opened. Try CAMERA_INDEX=1")

print("\n[INFO] Press SPACE to capture (only if exactly 1 face detected)")
print("[INFO] Press Q to quit")
print(f"[INFO] Saving as: {PERSON_ID}_{PERSON_NAME}_###.jpg\n")

count = 0

while True:
    ok, frame = cap.read()
    if not ok or frame is None:
        continue

    # detect on smaller image (speed)
    small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    rgb = np.require(rgb, dtype=np.uint8, requirements=["C"])

    locs = face_recognition.face_locations(rgb, model=MODEL)

    msg = f"ID={PERSON_ID}  Name={PERSON_NAME}  Faces={len(locs)}  Saved={count}/{TARGET_SAMPLES}"
    cv2.putText(frame, msg, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Enroll (SPACE=Save, Q=Quit)", frame)

    k = cv2.waitKey(1) & 0xFF
    if k in (ord("q"), ord("Q")):
        break

    if k == 32:  # SPACE
        if len(locs) != 1:
            print("[WARN] Need exactly 1 face. Try again.")
            continue

        count += 1
        filename = f"{PERSON_ID}_{PERSON_NAME}_{count:03d}.jpg"
        path = os.path.join(SAVE_DIR, filename)

        cv2.imwrite(path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        print("[OK] Saved:", filename)

        if count >= TARGET_SAMPLES:
            print("[DONE] Enough samples collected.")
            break

cap.release()
cv2.destroyAllWindows()
print("[FINISHED]")