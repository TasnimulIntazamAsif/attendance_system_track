import os, cv2, pickle
import numpy as np
import face_recognition
from collections import defaultdict

KNOWN_DIR = "known_faces"
DATA_DIR = "data"
ENC_PATH = os.path.join(DATA_DIR, "encodings.pkl")
os.makedirs(DATA_DIR, exist_ok=True)

MODEL = "hog"
NUM_JITTERS = 2   # increase to 5 if you want slower but stronger encodings

valid_ext = (".jpg", ".jpeg", ".png", ".webp")
files = [f for f in os.listdir(KNOWN_DIR) if f.lower().endswith(valid_ext)]
print("[INFO] Found:", files)

# group by person: "Asif_001.jpg" -> "Asif"
groups = defaultdict(list)
for f in files:
    base = os.path.splitext(f)[0]
    name = base.split("_")[0].strip()
    groups[name].append(f)

names = []
encodings = []

for name, flist in groups.items():
    person_encs = []
    print(f"\n[INFO] Person: {name} | images={len(flist)}")

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
                print(f"[WARN] {f} faces_found={len(locs)} (need 1) -> skipped")
                continue

            enc = face_recognition.face_encodings(
                rgb, known_face_locations=locs, num_jitters=NUM_JITTERS
            )[0]
            person_encs.append(enc)
            print("[OK] Encoded from:", f)

        except Exception as e:
            print("[ERROR] Encoding failed:", f, "->", e)

    if len(person_encs) == 0:
        print("[WARN] No encodings for:", name)
        continue

    avg_enc = np.mean(person_encs, axis=0)
    names.append(name)
    encodings.append(avg_enc)
    print(f"[DONE] {name} avg-encoding built from {len(person_encs)} images")

with open(ENC_PATH, "wb") as fp:
    pickle.dump({"names": names, "encodings": encodings}, fp)

print("\n[DONE] Total people saved:", len(names))
print("[NAMES]", names)