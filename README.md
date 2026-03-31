<img width="700" height="1322" alt="Screenshot 2026-03-31 131002" src="https://github.com/user-attachments/assets/d1a50b01-8b62-4ba8-b527-4a478622f892" /># 🎯 Flask Face Recognition Attendance System with Smart CCTV Face Capture & Multi-Person Tracking System
📌 Overview

This project is a real-time CCTV-based face capture and multi-person tracking system built using YOLO (Ultralytics) and OpenCV.
The system detects multiple people from a live CCTV stream, assigns each person a unique ID, and saves only clear face images into separate folders for each individual.

A real-time **face recognition based attendance system** built with **Flask, OpenCV, and face_recognition (dlib)**.  
The system uses a webcam to detect and recognize faces, mark attendance, and display matching accuracy.

---
<img width="2246" height="1475" alt="Screenshot 2026-02-25 115108" src="https://github.com/user-attachments/assets/df201ce6-d5f9-4991-95b3-fc6194354935" />

<img width="2248" height="1472" alt="Screenshot 2026-02-25 115436" src="https://github.com/user-attachments/assets/5c236a3d-89c1-4303-bf6c-431259ae116d" />


## iMAGE CAPTURING THROUGH TRACKING

<img width="2210" height="1207" alt="Screenshot 2026-03-31 130236" src="https://github.com/user-attachments/assets/f6215dc7-0365-45dc-8337-9cef3a768ad5" />


<img width="2201" height="1173" alt="Screenshot 2026-03-31 130428" src="https://github.com/user-attachments/assets/f0935496-a4fa-4c45-8b5b-8c0cf839fb73" />






## 🚀 Features

- 📷 Real-time webcam streaming
- 🧠 Face recognition using dlib (face_recognition)
- 📊 Attendance logging (CSV)
- 🎯 Match / Not Matched detection
- 📈 Accuracy (%) and distance display
- ⚡ Fast processing using frame resizing
- 🖥 Fullscreen webcam display (OpenCV)
- 🔄 Auto-scan functionality (Flask UI)

🎯 Real-time multi-person detection & tracking
🧍 Unique ID assigned to each person (BoT-SORT tracker)
😊 Detects and saves only face images (not full body)
📁 Automatic folder creation per person
⚡ Optimized for low latency CCTV streaming
🧠 Blur detection to avoid saving unclear images
🔄 Cooldown system to prevent duplicate captures
📉 Frame skipping for performance optimization
---

## 📂 Project Structure
<img width="700" height="1322" alt="Screenshot 2026-03-31 131002" src="https://github.com/user-attachments/assets/3ac5dc96-69a2-45b5-8ded-b5399021a312" />

## Technologies Used
Python
OpenCV
PyTorch
Ultralytics YOLOv8
Haar Cascade (Face Detection)

## ⚙️ Installation
1️⃣ Clone the repository
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
2️⃣ Create virtual environment
python -m venv .venv
.venv\Scripts\activate
3️⃣ Install dependencies
pip install ultralytics opencv-python torch torchvision


## ▶️ How to Run
python cctv_capture_face.py

## 📸 Output
Each detected person gets a folder:
known_faces/person_1/
known_faces/person_2/
Face images are saved like:
0001.jpg
0002.jpg

## 🧠 How It Works
📷 Capture frame from CCTV (RTSP stream)
🔍 Detect persons using YOLO
🧍 Assign unique ID using tracking
✂️ Crop person region
😊 Detect face using Haar Cascade
🧹 Filter blurry images
💾 Save face into respective folder

## ⚠️ Known Limitations
Very small or distant faces may not be detected
Haar Cascade may miss side faces
ID may change if person leaves and re-enters frame

## 🔥 Future Improvements
✅ Face Recognition (Deep Learning based)
✅ DeepSORT / ByteTrack integration
✅ Re-identification (Re-ID)
✅ Attendance system integration
✅ Web dashboard (Django/Flask)

## 👨‍💻 Author

# Tasnimul Intazam Asif

## 📄 License

This project is open-source and available under the MIT License.
