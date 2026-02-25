import cv2
import pickle
import numpy as np
import face_recognition

ENC_PATH = "data/encodings.pkl"
MODEL = "hog"
TOLERANCE = 0.45
CAMERA_INDEX = 0
RESIZE = 0.25  # faster recognition

def distance_to_confidence(distance: float, tolerance: float) -> float:
    """Heuristic confidence percentage from face distance (not a true probability)."""
    if distance is None:
        return 0.0
    if distance <= tolerance:
        conf = 100.0 - (distance / tolerance) * 50.0   # 100..50
    else:
        conf = 50.0 - ((distance - tolerance) / max(1e-6, (1.0 - tolerance))) * 50.0  # 50..0
    return float(max(0.0, min(100.0, conf)))

def draw_label(frame, text, org=(20, 55), base_scale=1.2, thickness=3, pad=10):
    """
    Draw text with background box.
    Auto-shrinks font so the whole label fits in the frame.
    """
    h, w = frame.shape[:2]
    x, y = org

    scale = base_scale
    while True:
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        # Check if it fits horizontally
        if x + tw + pad * 2 <= w - 5 or scale <= 0.5:
            break
        scale -= 0.1

    # Background rectangle coords
    x1 = x - pad
    y1 = y - th - pad
    x2 = x + tw + pad
    y2 = y + baseline + pad

    # Clamp to frame
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(w - 1, x2); y2 = min(h - 1, y2)

    # Draw background (black box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), -1)

    # Draw text (green)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 255, 0), thickness, cv2.LINE_AA)

# Load encodings
with open(ENC_PATH, "rb") as f:
    data = pickle.load(f)

print("[INFO] Known people:", data["names"])
if len(data["encodings"]) == 0:
    print("[FAIL] No encodings found. Run step4_build_encodings_debug.py first.")
    raise SystemExit

# Open camera (Windows stable)
cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

if not cap.isOpened():
    print("[FAIL] Cannot open camera. Try CAMERA_INDEX=1")
    raise SystemExit

# Create FULLSCREEN window
WIN = "Match Test"
cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

print("[OK] Matching started. Press Q to quit.")

while True:
    ok, frame = cap.read()
    if not ok or frame is None:
        continue

    # Recognition on small frame
    small = cv2.resize(frame, (0, 0), fx=RESIZE, fy=RESIZE)
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    rgb = np.require(rgb, dtype=np.uint8, requirements=["C"])

    msg = "No face"
    best_dist = None
    acc = 0.0

    locs = face_recognition.face_locations(rgb, model=MODEL)
    if locs:
        encs = face_recognition.face_encodings(rgb, locs)
        if encs:
            dists = face_recognition.face_distance(data["encodings"], encs[0])
            best = int(np.argmin(dists))
            best_dist = float(dists[best])
            acc = distance_to_confidence(best_dist, TOLERANCE)

            if best_dist <= TOLERANCE:
                msg = f"Matched: {data['names'][best]}"
            else:
                msg = "Not Matched"

    # Build shorter text that ALWAYS fits
    if best_dist is None:
        text = f"{msg} | acc={acc:.1f}%"
    else:
        text = f"{msg} | dist={best_dist:.3f} | acc={acc:.1f}%"

    draw_label(frame, text, org=(20, 60), base_scale=1.3, thickness=3)

    # Show fullscreen
    cv2.imshow(WIN, frame)

    key = cv2.waitKey(1) & 0xFF
    if key in (ord('q'), ord('Q')):
        break
    # Optional: press F to toggle fullscreen/windowed
    if key in (ord('f'), ord('F')):
        cur = cv2.getWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN)
        if cur == cv2.WINDOW_FULLSCREEN:
            cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
        else:
            cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

cap.release()
cv2.destroyAllWindows()
print("[DONE]")