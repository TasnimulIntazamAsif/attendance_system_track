import os
import cv2
import time
import pickle
import threading
import numpy as np
import face_recognition
from datetime import datetime

# =========================
# PATHS
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
ENC_PATH = os.path.join(DATA_DIR, "encodings.pkl")
CSV_PATH = os.path.join(DATA_DIR, "attendance.csv")
os.makedirs(DATA_DIR, exist_ok=True)

# =========================
# CONFIG
# =========================
MODEL = "hog"
TOLERANCE = 0.45
CAMERA_INDEX = 0

RECOG_RESIZE = 0.20
WORKER_INTERVAL = 0.15
AVG_ENCODING_FRAMES = 3

# Unknown duplicate control (optional)
SAVE_COOLDOWN_UNKNOWN = 10  # seconds

# ✅ NEW: store each matched person only once (per program run)
SAVE_EACH_PERSON_ONCE = True

# =========================
# LOAD ENCODINGS
# =========================
if not os.path.exists(ENC_PATH):
    raise SystemExit(f"[FAIL] encodings.pkl not found: {ENC_PATH}")

with open(ENC_PATH, "rb") as f:
    known = pickle.load(f)

if len(known.get("encodings", [])) == 0:
    raise SystemExit("[FAIL] No encodings. Run step4_build_encodings_debug.py first.")

print("[INFO] Known people:", known.get("names", []))
print("[INFO] Encodings path:", ENC_PATH)
print("[INFO] CSV path:", CSV_PATH)

# =========================
# CONFIDENCE (heuristic)
# =========================
def distance_to_confidence(distance):
    if distance is None:
        return 0.0
    if distance <= 0.35:
        return float(max(90.0, 100.0 - distance * 100.0))
    elif distance <= 0.45:
        return float(90.0 - (distance - 0.35) * 200.0)
    else:
        return float(max(0.0, 70.0 - (distance - 0.45) * 100.0))

# =========================
# DRAW LABEL (auto-fit)
# =========================
def draw_label(frame, text, org=(20, 60), base_scale=1.2, thickness=2, pad=10):
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

# =========================
# ATTENDANCE LOGGING
# =========================
# ✅ NEW: already saved matched people (per run)
already_saved_people = set()

# for Unknown cooldown
last_saved_unknown_ts = 0.0

def log_attendance(name, status, distance, accuracy):
    """
    Rules:
      - Matched -> save person only ONCE (if enabled)
      - Not Matched -> save Unknown (cooldown)
      - No Face -> do not save
    """
    global last_saved_unknown_ts

    if status == "No Face":
        return

    now = datetime.now()
    header_needed = not os.path.exists(CSV_PATH)

    # ✅ matched person
    if status == "Matched":
        if SAVE_EACH_PERSON_ONCE and name in already_saved_people:
            return  # don't save again
        already_saved_people.add(name)

        with open(CSV_PATH, "a", encoding="utf-8") as f:
            if header_needed:
                f.write("date,time,name,status,distance,accuracy\n")
            f.write(f"{now:%Y-%m-%d},{now:%H:%M:%S},{name},{status},{distance:.4f},{accuracy:.2f}\n")

        print(f"[SAVED ONCE] {now:%H:%M:%S} | {name} | Matched | dist={distance:.4f} | acc={accuracy:.2f}%")
        return

    # ❌ not matched -> Unknown (cooldown)
    if status == "Not Matched":
        ts = time.time()
        if ts - last_saved_unknown_ts < SAVE_COOLDOWN_UNKNOWN:
            return
        last_saved_unknown_ts = ts

        with open(CSV_PATH, "a", encoding="utf-8") as f:
            if header_needed:
                f.write("date,time,name,status,distance,accuracy\n")
            dist_str = "" if distance is None else f"{distance:.4f}"
            acc_str = "" if accuracy is None else f"{accuracy:.2f}"
            f.write(f"{now:%Y-%m-%d},{now:%H:%M:%S},Unknown,Not Matched,{dist_str},{acc_str}\n")

        print(f"[SAVED] {now:%H:%M:%S} | Unknown | Not Matched")
        return

# =========================
# SHARED STATE
# =========================
lock = threading.Lock()
latest_frame = None

result_status = "Starting..."
result_name = "Unknown"
result_dist = None
result_acc = 0.0

enc_buf = []
running = True

# =========================
# WORKER THREAD (recognition)
# =========================
def worker_loop():
    global result_status, result_name, result_dist, result_acc, enc_buf

    while running:
        time.sleep(WORKER_INTERVAL)

        with lock:
            if latest_frame is None:
                continue
            frame = latest_frame.copy()

        small = cv2.resize(frame, (0, 0), fx=RECOG_RESIZE, fy=RECOG_RESIZE)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        rgb = np.require(rgb, dtype=np.uint8, requirements=["C"])

        locs = face_recognition.face_locations(rgb, model=MODEL)

        if not locs:
            result_status, result_name, result_dist, result_acc = "No Face", "Unknown", None, 0.0
            enc_buf = []
            continue

        locs = [locs[0]]  # only first face for speed
        encs = face_recognition.face_encodings(rgb, locs)
        if not encs:
            result_status, result_name, result_dist, result_acc = "No Face", "Unknown", None, 0.0
            enc_buf = []
            continue

        enc_buf.append(encs[0])
        if len(enc_buf) > AVG_ENCODING_FRAMES:
            enc_buf = enc_buf[-AVG_ENCODING_FRAMES:]

        avg_enc = np.mean(np.array(enc_buf), axis=0)

        dists = face_recognition.face_distance(known["encodings"], avg_enc)
        best = int(np.argmin(dists))
        best_dist = float(dists[best])
        acc = distance_to_confidence(best_dist)

        if best_dist <= TOLERANCE:
            status = "Matched"
            name = known["names"][best]
        else:
            status = "Not Matched"
            name = "Unknown"

        result_status, result_name, result_dist, result_acc = status, name, best_dist, acc

        # ✅ log (will save matched person only once)
        log_attendance(name, status, best_dist, acc)

# =========================
# CAMERA + DISPLAY LOOP
# =========================
cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    raise SystemExit("[FAIL] Camera not opened. Try CAMERA_INDEX=1")

WIN = "Attendance System (One Save Per Person)"
cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

t = threading.Thread(target=worker_loop, daemon=True)
t.start()

print("[OK] Running. Matched person will be saved only ONCE. Q=quit, F=fullscreen toggle")

fps_t0 = time.time()
fps_count = 0
shown_fps = 0

while True:
    ok, frame = cap.read()
    if not ok or frame is None:
        continue

    with lock:
        latest_frame = frame

    fps_count += 1
    if time.time() - fps_t0 >= 1.0:
        shown_fps = fps_count
        fps_count = 0
        fps_t0 = time.time()

    if result_dist is None:
        text = f"{result_status} | FPS={shown_fps}"
    else:
        text = f"{result_status}: {result_name} | dist={result_dist:.3f} | acc={result_acc:.1f}% | FPS={shown_fps}"

    draw_label(frame, text, org=(20, 60))
    cv2.imshow(WIN, frame)

    key = cv2.waitKey(1) & 0xFF
    if key in (ord('q'), ord('Q')):
        break
    if key in (ord('f'), ord('F')):
        cur = cv2.getWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN)
        if cur == cv2.WINDOW_FULLSCREEN:
            cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
        else:
            cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

running = False
cap.release()
cv2.destroyAllWindows()
print("[DONE]")