import os
import cv2
import numpy as np

KNOWN_DIR = "known_faces"
valid_ext = (".jpg", ".jpeg", ".png", ".webp")

files = [f for f in os.listdir(KNOWN_DIR) if f.lower().endswith(valid_ext)]
print("[INFO] Found images:", files)

if not files:
    print("[FAIL] Put images inside known_faces/")
    raise SystemExit

for f in files:
    path = os.path.join(KNOWN_DIR, f)
    img = cv2.imread(path, cv2.IMREAD_COLOR)

    if img is None:
        print("[FAIL] cv2.imread failed:", f)
        continue

    print("[OK]", f, "shape:", img.shape, "dtype:", img.dtype, "mean:", np.mean(img))