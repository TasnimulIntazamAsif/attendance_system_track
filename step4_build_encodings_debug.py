import os
import cv2
import pickle
import numpy as np
import face_recognition
from collections import defaultdict

KNOWN_DIR = "known_faces"
DATA_DIR = "data"
ENC_PATH = os.path.join(DATA_DIR, "encodings.pkl")
os.makedirs(DATA_DIR, exist_ok=True)

MODEL = "hog"
NUM_JITTERS = 2
valid_ext = (".jpg", ".jpeg", ".png", ".webp")


def parse_id_name(filename: str):
    """
    Accepts:
      570508_Asif_001.jpg
      570508_Asif.jpg
    Returns:
      ("570508", "Asif") or (None, None)
    """
    base = os.path.splitext(filename)[0]
    parts = base.split("_")
    if len(parts) < 2:
        return None, None
    pid = parts[0].strip()
    name = parts[1].strip()
    if not pid or not name:
        return None, None
    return pid, name


files = [f for f in os.listdir(KNOWN_DIR) if f.lower().endswith(valid_ext)]
print("[INFO] Found:", files)

groups = defaultdict(list)
for f in files:
    pid, name = parse_id_name(f)
    if pid is None:
        print(f"[WARN] Skipped (bad filename): {f}  -> Use: ID_Name_001.jpg")
        continue
    groups[(pid, name)].append(f)

ids, names, encodings = [], [], []

for (pid, name), flist in groups.items():
    person_encs = []
    print(f"\n[INFO] Person: {pid}-{name} | images={len(flist)}")

    for f in flist:
        path = os.path.join(KNOWN_DIR, f)
        try:
            bgr = cv2.imread(path, cv2.IMREAD_COLOR)
            if bgr is None:
                print("[FAIL] imread:", f)
                continue

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb = np.require(rgb, dtype=np.uint8, requirements=["C"])

            locs = face_recognition.face_locations(rgb, model=MODEL)
            if len(locs) != 1:
                print(f"[WARN] {f} faces_found={len(locs)} -> skipped")
                continue

            enc = face_recognition.face_encodings(
                rgb, known_face_locations=locs, num_jitters=NUM_JITTERS
            )[0]
            person_encs.append(enc)
            print("[OK] Encoded:", f)

        except Exception as e:
            print("[ERROR] Encoding failed:", f, "->", e)

    if len(person_encs) == 0:
        print("[WARN] No encodings for:", pid, name)
        continue

    avg_enc = np.mean(person_encs, axis=0)
    ids.append(pid)
    names.append(name)
    encodings.append(avg_enc)

    print(f"[DONE] Saved avg encoding for {pid}-{name} from {len(person_encs)} images")

with open(ENC_PATH, "wb") as fp:
    pickle.dump({"ids": ids, "names": names, "encodings": encodings}, fp)

print("\n[SAVED] encodings.pkl ->", ENC_PATH)
print("[PEOPLE]", list(zip(ids, names)))