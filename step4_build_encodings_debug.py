import os
import cv2
import pickle
import numpy as np
import face_recognition

KNOWN_DIR = "known_faces"
DATA_DIR = "data"
ENC_PATH = os.path.join(DATA_DIR, "encodings.pkl")
os.makedirs(DATA_DIR, exist_ok=True)

MODEL = "hog"

valid_ext = (".jpg", ".jpeg", ".png", ".webp")
files = [f for f in os.listdir(KNOWN_DIR) if f.lower().endswith(valid_ext)]
print("[INFO] Found:", files)

names = []
encodings = []

for f in files:
    path = os.path.join(KNOWN_DIR, f)
    name = os.path.splitext(f)[0].strip()

    try:
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            print("[FAIL] imread:", f)
            continue

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # Force EXACT format dlib wants:
        rgb = np.require(rgb, dtype=np.uint8, requirements=["C"])

        print(f"[DEBUG] {f} rgb.shape={rgb.shape} dtype={rgb.dtype} contig={rgb.flags['C_CONTIGUOUS']}")

        locs = face_recognition.face_locations(rgb, model=MODEL)
        print(f"[DEBUG] {f} faces_found={len(locs)}")

        if len(locs) == 0:
            print("[WARN] No face:", f)
            continue

        enc = face_recognition.face_encodings(rgb, known_face_locations=locs)[0]
        names.append(name)
        encodings.append(enc)
        print("[OK] Encoded:", name)

    except Exception as e:
        print("[ERROR] Encoding failed:", f, "->", e)

with open(ENC_PATH, "wb") as fp:
    pickle.dump({"names": names, "encodings": encodings}, fp)

print("[DONE] Total encodings saved:", len(names))
print("[NAMES]", names)