import os
import cv2
import torch
import threading
import time
from collections import deque
from ultralytics import YOLO

# =========================
# SETTINGS
# =========================
CCTV_URL = "rtsp://admin:boss321%23@192.168.2.62:554/cam/realmonitor?channel=1&subtype=0"
SAVE_DIR = "known_faces"
PERSON_ID = "new_person"
TARGET_SAMPLES = 1500

FRAME_SKIP = 5
CONF_THRESHOLD = 0.5
FACE_MIN_SIZE = 80
BLUR_THRESHOLD = 60
RESIZE_WIDTH = 640
SAVE_COOLDOWN = 3  # seconds

os.makedirs(SAVE_DIR, exist_ok=True)

# =========================
# LOAD YOLO (PERSON ONLY)
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO("yolo26s.pt")
model.to(device)

if device == "cuda":
    model.model.half()

print("[INFO] Device:", device)

# =========================
# FACE DETECTOR
# =========================
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# =========================
# CONNECT CCTV
# =========================
cap = cv2.VideoCapture(CCTV_URL, cv2.CAP_FFMPEG)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print("[FAIL] CCTV not connected")
    exit()

print("[INFO] CCTV Connected")

# =========================
# FRAME THREAD
# =========================
frame_queue = deque(maxlen=1)
capture_flag = True

def capture_frames():
    global capture_flag
    while capture_flag:
        ret, frame = cap.read()
        if ret:
            frame_queue.append(frame)

threading.Thread(target=capture_frames, daemon=True).start()

# =========================
# BLUR CHECK
# =========================
def is_blurry(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() < BLUR_THRESHOLD

# =========================
# PERSON DETECTION
# =========================
def detect_person(frame):

    h, w = frame.shape[:2]
    scale = RESIZE_WIDTH / w
    resized = cv2.resize(frame, (RESIZE_WIDTH, int(h * scale)))

    results = model(resized, verbose=False)

    boxes = []

    for box in results[0].boxes:
        conf = float(box.conf[0])
        cls = int(box.cls[0])

        if conf > CONF_THRESHOLD and cls == 0:
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            x1 = int(x1 / scale)
            y1 = int(y1 / scale)
            x2 = int(x2 / scale)
            y2 = int(y2 / scale)

            boxes.append((x1, y1, x2, y2))

    return boxes

# =========================
# MAIN LOOP
# =========================
count = 0
frame_counter = 0
last_save_time = 0

while True:

    if not frame_queue:
        continue

    original_frame = frame_queue[-1].copy()
    display_frame = original_frame.copy()

    frame_counter += 1
    if frame_counter % FRAME_SKIP != 0:
        continue

    person_boxes = detect_person(original_frame)

    for (x1, y1, x2, y2) in person_boxes:

        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(display_frame, "Person",
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 0, 0), 2)

        # Safe boundary check
        h, w = original_frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        person_crop = original_frame[y1:y2, x1:x2]
        if person_crop.size == 0:
            continue

        gray = cv2.cvtColor(person_crop, cv2.COLOR_BGR2GRAY)

        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(FACE_MIN_SIZE, FACE_MIN_SIZE)
        )

        # ✅ Only save if exactly ONE face detected
        if len(faces) != 1:
            continue

        fx, fy, fw, fh = faces[0]

        face_crop = person_crop[fy:fy+fh, fx:fx+fw]
        if face_crop.size == 0:
            continue

        cv2.rectangle(display_frame,
                      (x1 + fx, y1 + fy),
                      (x1 + fx + fw, y1 + fy + fh),
                      (0, 255, 0), 2)

        current_time = time.time()

        # ✅ ONLY FACE SAVING HERE
        if not is_blurry(face_crop) and (current_time - last_save_time) > SAVE_COOLDOWN:

            count += 1
            filename = f"{PERSON_ID}_{count:04d}.jpg"
            cv2.imwrite(os.path.join(SAVE_DIR, filename), face_crop)

            last_save_time = current_time

            cv2.putText(display_frame, "FACE SAVED",
                        (x1 + fx, y1 + fy - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0,255,0), 2)

            print("[INFO] Face Saved:", filename)

            if count >= TARGET_SAMPLES:
                capture_flag = False
                break

    cv2.putText(display_frame, f"Saved: {count}/{TARGET_SAMPLES}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0,255,0), 2)

    cv2.imshow("PERSON + FACE DETECTION", display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        capture_flag = False
        break

cap.release()
cv2.destroyAllWindows()
print("[FINISHED]")