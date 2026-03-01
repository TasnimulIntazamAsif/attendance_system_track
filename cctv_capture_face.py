import os
import cv2
import numpy as np
from datetime import datetime
import threading
from ultralytics import YOLO

# =========================
# SETTINGS
# =========================
CCTV_URL = "rtsp://admin:boss321%23@192.168.2.42:554/cam/realmonitor?channel=1&subtype=0"  # Your CCTV camera URL with credentials
SAVE_DIR = "known_faces"
PERSON_ID = "new_person"  # You can update this with some identifier
PERSON_NAME = "CCTV_Face"  # Automatically tagged faces
TARGET_SAMPLES = 15  # Number of samples to capture
FRAME_SKIP = 10  # Process every 10th frame for faster performance

os.makedirs(SAVE_DIR, exist_ok=True)

# =========================
# Load YOLOv8 Model
# =========================
model = YOLO("yolov8n.pt")  # Load the pre-trained YOLOv8 model (small version for fast performance)

# =========================
# Connect to CCTV Stream
# =========================
cap = cv2.VideoCapture(CCTV_URL)

if not cap.isOpened():
    print("[FAIL] Could not connect to CCTV camera. Check URL.")
    exit(1)

print("[INFO] Connected to CCTV Stream.")

# =========================
# THREADS for Optimized Capture
# =========================
frame_queue = []
frame_lock = threading.Lock()
capture_flag = True

def capture_frames():
    global capture_flag

    while capture_flag:
        ret, frame = cap.read()
        if not ret:
            continue

        with frame_lock:
            frame_queue.append(frame)

        # Limit the frame queue size to avoid memory overload
        if len(frame_queue) > 10:
            with frame_lock:
                frame_queue.pop(0)

# Start frame capturing in a separate thread
capture_thread = threading.Thread(target=capture_frames, daemon=True)
capture_thread.start()

# =========================
# Main face detection loop
# =========================
count = 0  # Counter for saved faces
frame_counter = 0

def detect_face(frame):
    """Detect faces in the frame using YOLOv8 and return bounding box locations."""
    # Run YOLOv8 inference on the frame (detect faces)
    results = model(frame)

    # Extract face bounding boxes from results (YOLOv8 format)
    detections = results[0].boxes  # Access the first result
    boxes = []
    confidences = []
    class_ids = []

    # Iterate through detections and filter for "person" class (class 0)
    for det in detections:
        confidence = det.conf[0].item()  # Confidence score
        class_id = int(det.cls[0].item())  # Class ID (0 for person)

        if confidence > 0.5 and class_id == 0:  # Person class (ID 0 in COCO)
            x_center, y_center, w, h = det.xywh[0].tolist()  # Center x, y, width, height
            x = int((x_center - w / 2) * frame.shape[1])
            y = int((y_center - h / 2) * frame.shape[0])
            w = int(w * frame.shape[1])
            h = int(h * frame.shape[0])

            boxes.append([x, y, w, h])
            confidences.append(confidence)
            class_ids.append(class_id)

    return boxes, confidences, class_ids

while True:
    if len(frame_queue) == 0:
        continue

    with frame_lock:
        frame = frame_queue[-1]  # Get the latest frame from the queue

    frame_counter += 1
    if frame_counter % FRAME_SKIP != 0:
        continue  # Skip frames to reduce processing load

    # Detect faces using YOLOv8
    boxes, confidences, class_ids = detect_face(frame)

    # Show information on the screen
    msg = f"ID={PERSON_ID}  Name={PERSON_NAME}  Faces={len(boxes)}  Saved={count}/{TARGET_SAMPLES}"
    cv2.putText(frame, msg, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    if len(boxes) > 0:  # If faces are detected
        for i in range(len(boxes)):
            x, y, w, h = boxes[i]
            confidence = confidences[i]
            if confidence > 0.5:  # If confidence is high enough (threshold can be adjusted)
                # Draw bounding box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Save the face if detected and it's the 1st time (one face per frame)
                count += 1
                filename = f"{PERSON_ID}_CCTV_Face_{count:03d}.jpg"
                path = os.path.join(SAVE_DIR, filename)

                # Crop face and save
                face_image = frame[y:y + h, x:x + w]
                cv2.imwrite(path, face_image)  # Save image as .jpg
                print(f"[INFO] Saved: {filename}")

                if count >= TARGET_SAMPLES:
                    print("[DONE] Enough samples collected.")
                    break

    # Show the frame with bounding boxes
    cv2.imshow("CCTV Face Capture with YOLOv8", frame)

    # Exit on Q key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture_flag = False
cap.release()
cv2.destroyAllWindows()
print("[FINISHED] Face capture from CCTV using YOLOv8 completed.")