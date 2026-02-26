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

# ✅ Prevent repeated saving for same person (in seconds)
PERSON_EVENT_COOLDOWN = 10

# Unknown anti-spam
UNKNOWN_COOLDOWN = 10

CSV_HEADER = "timeframe,id,name,status,accuracy\n"

# =========================
# CSV HEADER ENFORCER
# =========================
def ensure_csv_header():
    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH, "w", encoding="utf-8") as f:
            f.write(CSV_HEADER)
        return

    try:
        with open(CSV_PATH, "r", encoding="utf-8") as f:
            first = f.readline()
    except Exception:
        first = ""

    if first != CSV_HEADER:
        backup = os.path.join(DATA_DIR, "attendance_old_backup.csv")
        try:
            if os.path.exists(backup):
                os.remove(backup)
        except:
            pass

        os.replace(CSV_PATH, backup)
        with open(CSV_PATH, "w", encoding="utf-8") as f:
            f.write(CSV_HEADER)

        print(f"[INFO] Old CSV backed up to: {backup}")
        print(f"[INFO] New CSV created with correct header: {CSV_PATH}")

ensure_csv_header()

# =========================
# LOAD ENCODINGS
# =========================
if not os.path.exists(ENC_PATH):
    raise SystemExit("[FAIL] encodings.pkl not found. Run step4 first.")

with open(ENC_PATH, "rb") as f:
    known = pickle.load(f)

known_ids = known.get("ids", [])
known_names = known.get("names", [])
known_encs = known.get("encodings", [])

if len(known_encs) == 0:
    raise SystemExit("[FAIL] No encodings found in encodings.pkl")

print("[INFO] Loaded people:", list(zip(known_ids, known_names)))
print("[INFO] CSV ->", CSV_PATH)

# =========================
# ACCURACY (heuristic)
# =========================
def distance_to_accuracy(distance):
    if distance is None:
        return 0.0
    if distance <= 0.35:
        return float(max(90.0, 100.0 - distance * 100.0))
    elif distance <= 0.45:
        return float(90.0 - (distance - 0.35) * 200.0)
    else:
        return float(max(0.0, 70.0 - (distance - 0.45) * 100.0))

# =========================
# DRAW LABEL
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
# ✅ CHECK-IN/CHECK-OUT STATE
# =========================
# checked_state[id] = True means currently checked-in, next event will be Check Out
checked_state = {}
last_event_ts = {}  # last time we saved an event for this id
last_unknown_ts = 0.0

def load_state_from_csv():
    """
    Reads existing CSV and sets checked_state based on last status for each ID:
      - Last status "Check In" => checked_state[id]=True
      - Last status "Check Out" => checked_state[id]=False
    """
    if not os.path.exists(CSV_PATH):
        return

    try:
        with open(CSV_PATH, "r", encoding="utf-8") as f:
            header = f.readline()
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 5:
                    continue
                _, pid, _, status, _ = parts[0], parts[1], parts[2], parts[3], parts[4]
                if pid and pid != "Unknown":
                    if status == "Check In":
                        checked_state[pid] = True
                    elif status == "Check Out":
                        checked_state[pid] = False
    except Exception as e:
        print("[WARN] Could not load state from CSV:", e)

load_state_from_csv()

def append_csv(timeframe, pid, name, status, accuracy):
    with open(CSV_PATH, "a", encoding="utf-8") as f:
        f.write(f"{timeframe},{pid},{name},{status},{accuracy:.2f}\n")

def log_check_in_out(pid, name, accuracy):
    """
    Toggle logic:
      if checked_state[pid] is False/missing => Check In
      if checked_state[pid] is True         => Check Out
    Applies per-person cooldown to avoid spam.
    """
    now_ts = time.time()
    last_ts = last_event_ts.get(pid, 0.0)
    if now_ts - last_ts < PERSON_EVENT_COOLDOWN:
        return None  # no new event

    last_event_ts[pid] = now_ts

    currently_in = checked_state.get(pid, False)
    status = "Check Out" if currently_in else "Check In"

    # toggle state
    checked_state[pid] = (status == "Check In")

    timeframe = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    append_csv(timeframe, pid, name, status, accuracy)
    print(f"[SAVED] {pid}-{name} | {status} | acc={accuracy:.2f}%")

    return status

def log_unknown(accuracy):
    global last_unknown_ts
    ts = time.time()
    if ts - last_unknown_ts < UNKNOWN_COOLDOWN:
        return

    last_unknown_ts = ts
    timeframe = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    append_csv(timeframe, "Unknown", "Unknown", "Not Matched", accuracy)
    print(f"[SAVED] Unknown | Not Matched | acc={accuracy:.2f}%")

# =========================
# THREAD STATE
# =========================
lock = threading.Lock()
latest_frame = None

result_text = "Starting..."
running = True
enc_buf = []

def worker_loop():
    global result_text, enc_buf

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
            result_text = "No Face"
            enc_buf = []
            continue

        locs = [locs[0]]  # first face only
        encs = face_recognition.face_encodings(rgb, locs)
        if not encs:
            result_text = "No Face"
            enc_buf = []
            continue

        enc_buf.append(encs[0])
        if len(enc_buf) > AVG_ENCODING_FRAMES:
            enc_buf = enc_buf[-AVG_ENCODING_FRAMES:]

        avg_enc = np.mean(np.array(enc_buf), axis=0)

        dists = face_recognition.face_distance(known_encs, avg_enc)
        best = int(np.argmin(dists))
        best_dist = float(dists[best])
        acc = distance_to_accuracy(best_dist)

        if best_dist <= TOLERANCE:
            pid = known_ids[best]
            name = known_names[best]

            # decide / save event if cooldown passed
            saved_status = log_check_in_out(pid, name, acc)

            # show status even if not saved (based on current state)
            # If saved_status is None, show what would be next? Better show "Detected"
            current_in = checked_state.get(pid, False)
            display_status = saved_status if saved_status else ("Checked In" if current_in else "Checked Out")
            result_text = f"{pid}_{name} | {display_status} | acc={acc:.1f}%"

        else:
            log_unknown(acc)
            result_text = f"Not Matched | acc={acc:.1f}%"

# =========================
# CAMERA + DISPLAY
# =========================
cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    raise SystemExit("[FAIL] Camera not opened. Try CAMERA_INDEX=1")

WIN = "Attendance System (Check In / Check Out)"
cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

threading.Thread(target=worker_loop, daemon=True).start()

print("[OK] Running... Q=quit")

while True:
    ok, frame = cap.read()
    if not ok or frame is None:
        continue

    with lock:
        latest_frame = frame

    draw_label(frame, result_text, org=(20, 60))
    cv2.imshow(WIN, frame)

    key = cv2.waitKey(1) & 0xFF
    if key in (ord('q'), ord('Q')):
        break

running = False
cap.release()
cv2.destroyAllWindows()
print("[DONE]")