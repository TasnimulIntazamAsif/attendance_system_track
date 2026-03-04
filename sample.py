import os
import cv2
import torch
import time
import threading
import math
import csv
import uuid
from datetime import datetime
from ultralytics import YOLO

# =========================
# SETTINGS
# =========================
CCTV_URL = "rtsp://admin:boss321%23@192.168.2.25:554/cam/realmonitor?channel=1&subtype=0"
BASE_SAVE_DIR = "known_faces"

CONF_THRESHOLD = 0.4
FACE_MIN_SIZE = 60
SAVE_COOLDOWN = 1

RESIZE_WIDTH = 416
YOLO_IMG_SIZE = 416
MOVEMENT_THRESHOLD = 15

os.makedirs(BASE_SAVE_DIR, exist_ok=True)

# =========================
# LOAD MODEL
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO("yolo26s.pt")

if device == "cuda":
    model.to(device)
    model.model.half()
    torch.backends.cudnn.benchmark = True

print("[INFO] Device:", device)

# =========================
# FACE DETECTORS (FRONTAL + PROFILE)
# =========================
frontal_face = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

profile_face = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_profileface.xml"
)

# =========================
# UNIQUE ID STORAGE
# =========================
trackid_to_unique = {}
unique_image_count = {}

def generate_unique_id():
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    short_uuid = uuid.uuid4().hex[:6].upper()
    return f"UID_{timestamp}_{short_uuid}"

def log_to_csv(folder_path, unique_id, status, image_name):
    csv_name = f"{unique_id}_log.csv"
    csv_path = os.path.join(folder_path, csv_name)
    file_exists = os.path.isfile(csv_path)

    with open(csv_path, mode="a", newline="") as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow(["timestamp", "unique_id", "status", "image_name"])

        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            unique_id,
            status,
            image_name
        ])

# =========================
# CAMERA THREAD
# =========================
class CameraStream:
    def __init__(self, url):
        self.cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.ret, self.frame = self.cap.read()
        self.running = True
        threading.Thread(target=self.update, daemon=True).start()

    def update(self):
        while self.running:
            self.cap.grab()
            self.ret, self.frame = self.cap.read()

    def read(self):
        return self.ret, self.frame

    def stop(self):
        self.running = False
        self.cap.release()

stream = CameraStream(CCTV_URL)

if not stream.cap.isOpened():
    print("[FAIL] CCTV not connected")
    exit()

print("[INFO] CCTV Connected")

cv2.namedWindow("ULTRA LIVE TRACKING", cv2.WINDOW_NORMAL)

last_save_time = {}
previous_positions = {}

# =========================
# MAIN LOOP
# =========================
while True:

    ret, frame = stream.read()
    if not ret:
        continue

    frame_small = cv2.resize(
        frame,
        (RESIZE_WIDTH, int(frame.shape[0] * RESIZE_WIDTH / frame.shape[1]))
    )

    scale_x = frame.shape[1] / frame_small.shape[1]
    scale_y = frame.shape[0] / frame_small.shape[0]

    results = model.track(
        frame_small,
        persist=True,
        classes=[0],
        imgsz=YOLO_IMG_SIZE,
        conf=CONF_THRESHOLD,
        tracker="bytetrack.yaml",
        verbose=False
    )

    display_frame = frame

    if results[0].boxes.id is None:
        cv2.imshow("ULTRA LIVE TRACKING", display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    boxes = results[0].boxes

    for box, conf, track_id in zip(boxes.xyxy, boxes.conf, boxes.id):

        if float(conf) < CONF_THRESHOLD:
            continue

        x1, y1, x2, y2 = box.tolist()
        x1 = int(x1 * scale_x)
        y1 = int(y1 * scale_y)
        x2 = int(x2 * scale_x)
        y2 = int(y2 * scale_y)

        track_id = int(track_id)

        # ===== ASSIGN UNIQUE ID =====
        if track_id not in trackid_to_unique:
            unique_id = generate_unique_id()
            trackid_to_unique[track_id] = unique_id
            unique_image_count[unique_id] = 0
            print(f"[INFO] New Unique ID Created: {unique_id}")

        unique_id = trackid_to_unique[track_id]

        # ===== MOVEMENT =====
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        status = "STAY"
        color = (0, 255, 0)

        if track_id in previous_positions:
            prev_x, prev_y = previous_positions[track_id]
            distance = math.hypot(center_x - prev_x, center_y - prev_y)

            if distance > MOVEMENT_THRESHOLD:
                status = "MOVING"
                color = (0, 0, 255)

        previous_positions[track_id] = (center_x, center_y)

        cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(display_frame, f"{unique_id} - {status}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 2)

        current_time = time.time()
        if track_id in last_save_time:
            if current_time - last_save_time[track_id] < SAVE_COOLDOWN:
                continue

        person_crop = frame[y1:y2, x1:x2]
        if person_crop.size == 0:
            continue

        gray = cv2.cvtColor(person_crop, cv2.COLOR_BGR2GRAY)

        # Detect frontal + profile faces
        faces1 = frontal_face.detectMultiScale(gray, 1.2, 5)
        faces2 = profile_face.detectMultiScale(gray, 1.2, 5)

        faces = list(faces1) + list(faces2)

        if len(faces) == 0:
            continue

        # Pick largest detected face
        faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
        fx, fy, fw, fh = faces[0]

        if fw < FACE_MIN_SIZE or fh < FACE_MIN_SIZE:
            continue

        face_img = person_crop[fy:fy+fh, fx:fx+fw]

        folder_path = os.path.join(BASE_SAVE_DIR, unique_id)
        os.makedirs(folder_path, exist_ok=True)

        unique_image_count[unique_id] += 1
        image_name = f"{unique_id}_{unique_image_count[unique_id]:03d}.jpg"

        image_path = os.path.join(folder_path, image_name)
        cv2.imwrite(image_path, face_img)

        log_to_csv(folder_path, unique_id, status, image_name)

        last_save_time[track_id] = time.time()

        print(f"[INFO] Face Saved {image_name}")

    cv2.imshow("ULTRA LIVE TRACKING", display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

stream.stop()
cv2.destroyAllWindows()