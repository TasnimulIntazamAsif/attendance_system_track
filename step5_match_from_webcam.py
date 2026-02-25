import cv2
import pickle
import numpy as np
import face_recognition
from collections import deque

ENC_PATH = "data/encodings.pkl"
MODEL = "hog"
TOLERANCE = 0.45
CAMERA_INDEX = 0
RESIZE = 0.25

AVG_FRAMES = 5   # average encodings across N frames (boost stability)

def distance_to_confidence(distance: float, good=0.35, ok=0.45) -> float:
    """
    Confidence mapping tuned for face_recognition distances.
    - <= good distance => 90..100%
    - good..ok => 70..90%
    - > ok => down to 0%
    (Still a heuristic; distance is the real metric.)
    """
    if distance is None:
        return 0.0
    if distance <= good:
        # map [0..good] -> [100..90]
        return float(max(90.0, 100.0 - (distance / max(1e-6, good)) * 10.0))
    if distance <= ok:
        # map [good..ok] -> [90..70]
        return float(90.0 - ((distance - good) / max(1e-6, (ok - good))) * 20.0)
    # map [ok..1.0] -> [70..0]
    return float(max(0.0, 70.0 - ((distance - ok) / max(1e-6, (1.0 - ok))) * 70.0))

def draw_label(frame, text, org=(20, 60), base_scale=1.3, thickness=3, pad=10):
    h, w = frame.shape[:2]
    x, y = org

    scale = base_scale
    while True:
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        if x + tw + pad * 2 <= w - 5 or scale <= 0.6:
            break
        scale -= 0.1

    x1, y1 = max(0, x - pad), max(0, y - th - pad)
    x2, y2 = min(w - 1, x + tw + pad), min(h - 1, y + baseline + pad)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), -1)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 255, 0), thickness, cv2.LINE_AA)

with open(ENC_PATH, "rb") as f:
    data = pickle.load(f)

print("[INFO] Known people:", data["names"])
if len(data["encodings"]) == 0:
    raise SystemExit("[FAIL] No encodings found. Run step4_build_encodings_debug.py")

cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
if not cap.isOpened():
    raise SystemExit("[FAIL] Camera not opened. Try CAMERA_INDEX=1")

WIN = "Match Test"
cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

print("[OK] Press Q to quit. (Tip: enroll from webcam for 90%+)")
enc_queue = deque(maxlen=AVG_FRAMES)

while True:
    ok, frame = cap.read()
    if not ok or frame is None:
        continue

    small = cv2.resize(frame, (0, 0), fx=RESIZE, fy=RESIZE)
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    rgb = np.require(rgb, dtype=np.uint8, requirements=["C"])

    locs = face_recognition.face_locations(rgb, model=MODEL)
    name = "Unknown"
    best_dist = None

    if locs:
        encs = face_recognition.face_encodings(rgb, locs)
        if encs:
            enc_queue.append(encs[0])

    # Only decide if we have enough frames
    if len(enc_queue) >= max(2, AVG_FRAMES // 2):
        avg_enc = np.mean(np.array(enc_queue), axis=0)

        dists = face_recognition.face_distance(data["encodings"], avg_enc)
        best = int(np.argmin(dists))
        best_dist = float(dists[best])

        if best_dist <= TOLERANCE:
            name = data["names"][best]

    conf = distance_to_confidence(best_dist, good=0.35, ok=TOLERANCE)
    if best_dist is None:
        text = "No face / stabilizing..."
    else:
        status = "Matched" if name != "Unknown" else "Not Matched"
        text = f"{status}: {name} | dist={best_dist:.3f} | acc={conf:.1f}%"

    draw_label(frame, text)

    cv2.imshow(WIN, frame)
    k = cv2.waitKey(1) & 0xFF
    if k in (ord('q'), ord('Q')):
        break

cap.release()
cv2.destroyAllWindows()
print("[DONE]")