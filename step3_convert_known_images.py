import os
import cv2

KNOWN_DIR = "known_faces"
OUT_DIR = "known_faces_fixed"
os.makedirs(OUT_DIR, exist_ok=True)

valid_ext = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff")

files = [f for f in os.listdir(KNOWN_DIR) if f.lower().endswith(valid_ext)]
print("[INFO] Converting:", files)

for f in files:
    in_path = os.path.join(KNOWN_DIR, f)
    name = os.path.splitext(f)[0]
    out_path = os.path.join(OUT_DIR, name + ".jpg")

    img = cv2.imread(in_path, cv2.IMREAD_COLOR)
    if img is None:
        print("[FAIL] Cannot read:", f)
        continue

    ok = cv2.imwrite(out_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if ok:
        print("[OK] Converted:", f, "->", out_path)
    else:
        print("[FAIL] Write failed:", f)

print("\nDONE. Move known_faces_fixed/*.jpg to known_faces/ (replace old).")