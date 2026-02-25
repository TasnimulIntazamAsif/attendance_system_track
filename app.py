import os
import cv2
import time
import pickle
import threading
from datetime import datetime

import numpy as np
import face_recognition
from flask import Flask, render_template, Response, jsonify


# =========================
# CONFIG
# =========================
APP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(APP_DIR, "data")
ENC_PATH = os.path.join(DATA_DIR, "encodings.pkl")
CSV_PATH = os.path.join(DATA_DIR, "attendance.csv")

os.makedirs(DATA_DIR, exist_ok=True)

CAMERA_INDEX = 0

# Stream
STREAM_WIDTH = 800
STREAM_FPS = 15
JPEG_QUALITY = 70

# Recognition
MODEL = "hog"
TOLERANCE = 0.45
RECOG_RESIZE = 0.25

app = Flask(__name__)


# =========================
# Load encodings
# =========================
def load_encodings():
    if not os.path.exists(ENC_PATH):
        print("[WARN] No encodings.pkl found. Run step4_build_encodings_debug.py first.")
        return {"names": [], "encodings": []}

    with open(ENC_PATH, "rb") as f:
        data = pickle.load(f)

    print("[INFO] Loaded known faces:", data.get("names", []))
    return data


known = load_encodings()


# =========================
# Confidence (heuristic)
# =========================
def distance_to_confidence(distance: float, tolerance: float) -> float:
    if distance is None:
        return 0.0

    if distance <= tolerance:
        conf = 100.0 - (distance / tolerance) * 50.0
    else:
        conf = 50.0 - ((distance - tolerance) / max(1e-6, (1.0 - tolerance))) * 50.0

    return float(max(0.0, min(100.0, conf)))


# =========================
# Attendance logging
# =========================
def log_attendance(name: str, status: str, distance=None, accuracy=None):
    """
    RULE:
      - If matched -> store real name
      - If not matched/no face/no known -> store Unknown
    """
    if status != "Matched":
        name = "Unknown"

    now = datetime.now()
    header_needed = not os.path.exists(CSV_PATH)

    with open(CSV_PATH, "a", encoding="utf-8") as f:
        if header_needed:
            f.write("date,time,name,status,distance,accuracy\n")

        dist_str = "" if distance is None else f"{distance:.4f}"
        acc_str = "" if accuracy is None else f"{accuracy:.2f}"

        f.write(f"{now:%Y-%m-%d},{now:%H:%M:%S},{name},{status},{dist_str},{acc_str}\n")


# =========================
# Camera thread
# =========================
class Camera:
    def __init__(self, idx=0):
        self.idx = idx
        self.cap = None
        self.running = False
        self.lock = threading.Lock()
        self.last_frame = None
        self.last_jpeg = None

    def start(self):
        self.cap = cv2.VideoCapture(self.idx, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open webcam. Try CAMERA_INDEX=1 and close Zoom/Camera apps.")

        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        for _ in range(10):
            self.cap.read()
            time.sleep(0.03)

        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()
        print("[INFO] Camera started.")

    def _loop(self):
        interval = 1.0 / float(STREAM_FPS)
        last_t = 0.0

        while self.running:
            now = time.time()
            if now - last_t < interval:
                time.sleep(0.002)
                continue
            last_t = now

            ok, frame = self.cap.read()
            if not ok or frame is None:
                continue

            h, w = frame.shape[:2]
            if w > STREAM_WIDTH:
                scale = STREAM_WIDTH / float(w)
                stream = cv2.resize(frame, (int(w * scale), int(h * scale)))
            else:
                stream = frame

            ok2, jpeg = cv2.imencode(
                ".jpg", stream,
                [int(cv2.IMWRITE_JPEG_QUALITY), int(JPEG_QUALITY)]
            )
            if not ok2:
                continue

            with self.lock:
                self.last_frame = frame
                self.last_jpeg = jpeg.tobytes()

    def get_frame(self):
        with self.lock:
            return None if self.last_frame is None else self.last_frame.copy()

    def get_jpeg(self):
        with self.lock:
            return self.last_jpeg


camera = Camera(CAMERA_INDEX)
scan_lock = threading.Lock()


# =========================
# Face match
# =========================
def match_face(frame_bgr):
    if len(known["encodings"]) == 0:
        return {"status": "No Known Faces", "name": "Unknown", "distance": None, "accuracy": 0.0}

    small = cv2.resize(frame_bgr, (0, 0), fx=RECOG_RESIZE, fy=RECOG_RESIZE)
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    rgb = np.require(rgb, dtype=np.uint8, requirements=["C"])

    locs = face_recognition.face_locations(rgb, model=MODEL)
    if not locs:
        return {"status": "No Face", "name": "Unknown", "distance": None, "accuracy": 0.0}

    encs = face_recognition.face_encodings(rgb, locs)
    if not encs:
        return {"status": "No Face", "name": "Unknown", "distance": None, "accuracy": 0.0}

    dists = face_recognition.face_distance(known["encodings"], encs[0])
    best = int(np.argmin(dists))
    best_dist = float(dists[best])

    accuracy = distance_to_confidence(best_dist, TOLERANCE)

    if best_dist <= TOLERANCE:
        return {"status": "Matched", "name": known["names"][best], "distance": best_dist, "accuracy": accuracy}

    return {"status": "Not Matched", "name": "Unknown", "distance": best_dist, "accuracy": accuracy}


# =========================
# MJPEG stream
# =========================
def gen_mjpeg():
    while True:
        jpeg = camera.get_jpeg()
        if jpeg is None:
            time.sleep(0.03)
            continue
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n")


# =========================
# Routes
# =========================
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(gen_mjpeg(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/scan", methods=["POST"])
def scan():
    if not scan_lock.acquire(blocking=False):
        return jsonify({"ok": True, "message": "⏳ Scanning..."}), 200

    try:
        frame = camera.get_frame()
        if frame is None:
            return jsonify({"ok": False, "message": "Camera not ready"}), 500

        res = match_face(frame)

        # ✅ Always log:
        # - Matched -> store real name
        # - Others  -> store Unknown
        log_attendance(res["name"], res["status"], res["distance"], res["accuracy"])

        if res["status"] == "Matched":
            msg = f"✅ Matched: {res['name']} | Accuracy: {res['accuracy']:.1f}% | Distance: {res['distance']:.3f}"
        elif res["status"] == "No Face":
            msg = "⚠️ No face detected (saved as Unknown)"
        elif res["status"] == "No Known Faces":
            msg = "⚠️ No known faces loaded (saved as Unknown)"
        else:
            msg = f"❌ Not Matched (saved as Unknown) | Accuracy: {res['accuracy']:.1f}% | Distance: {res['distance']:.3f}"

        return jsonify({"ok": True, "result": res, "message": msg})
    finally:
        scan_lock.release()


@app.route("/reload", methods=["POST"])
def reload():
    global known
    known = load_encodings()
    return jsonify({"ok": True, "count": len(known["names"])})


if __name__ == "__main__":
    camera.start()
    app.run(host="127.0.0.1", port=5000, debug=True, threaded=True, use_reloader=False)