from flask import Flask, render_template, Response
import cv2
import face_recognition
import os

app = Flask(__name__)

# Path to the known faces directory
KNOWN_FACES_DIR = 'known_faces'

# List of known faces and their names
known_face_encodings = []
known_face_names = []


# Load known faces from the directory
def load_known_faces():
    global known_face_encodings, known_face_names
    for filename in os.listdir(KNOWN_FACES_DIR):
        if filename.endswith('.jpeg') or filename.endswith('.png'):
            image_path = os.path.join(KNOWN_FACES_DIR, filename)
            # Load the image using OpenCV
            image = cv2.imread(image_path)

            if image is None:
                print(f"Error: Unable to load image {filename}")
                continue  # Skip if image can't be loaded

            # Convert image to RGB for face_recognition
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Find face encodings for all faces in the image
            face_encodings = face_recognition.face_encodings(image_rgb)

            if face_encodings:
                # Use the first face found in the image (assumes one face per image)
                encoding = face_encodings[0]
                known_face_encodings.append(encoding)
                known_face_names.append(os.path.splitext(filename)[0])  # Use filename as the name
            else:
                print(f"No faces found in {filename}. Skipping this file.")


# Initialize known faces
load_known_faces()


# Webcam feed generator
def gen_frames():
    cap = cv2.VideoCapture(0)  # Open webcam

    if not cap.isOpened():
        print("Error: Unable to access the webcam.")
        return

    while True:
        ret, frame = cap.read()  # Read frame from webcam
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Convert frame to RGB for face_recognition
        rgb_frame = frame[:, :, ::-1]

        # Find faces in the frame
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        # Process each face found
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

            name = "Unknown"
            status = "Not Matched"
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]
                status = "Matched"

            # Draw rectangle around the face and display the name/status
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.putText(frame, f"{name} - {status}", (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Convert frame to JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Return the frame as a streaming response
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()  # Release the webcam after the loop is finished


# Route for the webcam feed
@app.route('/video')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


# Route for the main page
@app.route('/')
def index():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)