import os
import time
import cv2
import numpy as np
from flask import Flask, jsonify, render_template, Response, send_from_directory
import pygame
import sounddevice as sd
from threading import Thread
import mediapipe as mp
import dlib
import face_recognition
import logging
from mouth import start_mouth_detection_with_alarm, detect_mouth_opening
import queue

status_queue = queue.Queue()
initial_photo = None  # To store the initial photo

# Configure logging
logging.basicConfig(level=logging.INFO)
# Initialize Pygame for playing the alarm sound
pygame.mixer.init()

# Flask app setup
app = Flask(__name__)

# Load face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Constants for mouth detection
MOUTH_AR_THRESH = 0.7
TALK_TIME_THRESH = 1

# Load the alarm sound
alarm_sound = pygame.mixer.Sound("alarm.wav")  # Replace with your alarm sound file

# Mediapipe for head pose detection
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Variables for alarm management
alarm_active = False
start_talking_time = None

mic = 1
threshold = 0.02
stream = sd.InputStream(device=mic, channels=1, samplerate=44100, blocksize=1024)

def start_alarm():
    global alarm_active
    pygame.mixer.music.load("alarm.wav")
    pygame.mixer.music.play(loops=-1)

def stop_alarm():
    global alarm_active
    pygame.mixer.music.stop()
    alarm_active = False

def generate_frames():
    global alarm_active, start_talking_time, initial_photo

    try:
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            logging.error("Failed to open camera.")
            return

        # Capture initial photo
        if initial_photo is None:
            success, initial_photo = camera.read()
            if not success:
                logging.error("Failed to capture initial photo.")
                return
            logging.info("Initial photo captured.")

        while True:
            success, frame = camera.read()
            if not success:
                logging.error("Failed to read frame from camera.")
                break

            frame = detect_head_pose(frame)

            lighting_status = "Lighting conditions are good."
            if not check_lighting(frame):
                lighting_status = "Lighting conditions are not ideal."
                # logging.info(lighting_status)

            blur_status = "Camera quality is good."
            if not check_blur(frame):
                blur_status = "Camera is blurry."
                # logging.info(blur_status)

            status_queue.put({"lighting": lighting_status, "blur": blur_status})

            if not check_face(frame):
                start_alarm()

            if detect_mouth_opening(frame):
                if start_talking_time is None:
                    start_talking_time = time.time()
                elif time.time() - start_talking_time > TALK_TIME_THRESH and not alarm_active:
                    start_alarm()
            else:
                start_talking_time = None
                stop_alarm()

            # Compare current frame with initial photo
            if initial_photo is not None:
                if is_tampered(frame, initial_photo):
                    start_alarm()
                else:
                    stop_alarm()

            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                logging.error("Failed to encode frame.")
                break

            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    except Exception as e:
        logging.exception("An error occurred while generating frames.")
    finally:
        camera.release()

def detect_head_pose(frame):
    global alarm_active

    image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    img_h, img_w, img_c = image.shape
    face_3d = []
    face_2d = []

    face_ids = [33, 263, 1, 61, 291, 199]

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for idx, lm in enumerate(face_landmarks.landmark):
                if idx in face_ids:
                    if idx == 1:
                        nose_2d = (lm.x * img_w, lm.y * img_h)
                        nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 8000)

                    x, y = int(lm.x * img_w), int(lm.y * img_h)

                    face_2d.append([x, y])
                    face_3d.append([x, y, lm.z])

            face_2d = np.array(face_2d, dtype=np.float64)
            face_3d = np.array(face_3d, dtype=np.float64)
            focal_length = 1 * img_w
            cam_matrix = np.array([[focal_length, 0, img_h / 2],
                                   [0, focal_length, img_w / 2],
                                   [0, 0, 1]])
            dist_matrix = np.zeros((4, 1), dtype=np.float64)
            success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
            rmat, jac = cv2.Rodrigues(rot_vec)
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

            x = angles[0] * 360
            y = angles[1] * 360

            if y < -10 or y > 10 or x < -5:
                if not alarm_active:
                    alarm_active = True
                    start_alarm()
                else:
                    if alarm_active:
                        stop_alarm()

                text = f"x: {int(x)} | y: {int(y)}"
                cv2.putText(image, text, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    return image

def check_face(frame):
    global alarm_active
    rgb_frame = frame[:, :, ::-1]
    face_locations = face_recognition.face_locations(rgb_frame)
    if len(face_locations) == 0:
        if not alarm_active:
            alarm_active = True
            start_alarm()
        return False
    else:
        if alarm_active:
            stop_alarm()
        return True

def check_lighting(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    if mean_brightness < 50 or mean_brightness > 200:
        return False
    return True

def check_blur(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if laplacian_var < 100:
        return False
    return True

def is_tampered(current_frame, initial_frame):
    current_face_locations = face_recognition.face_locations(current_frame)
    initial_face_locations = face_recognition.face_locations(initial_frame)
    
    if not current_face_locations or not initial_face_locations:
        return True

    current_encoding = face_recognition.face_encodings(current_frame, current_face_locations)[0]
    initial_encoding = face_recognition.face_encodings(initial_frame, initial_face_locations)[0]

    match = face_recognition.compare_faces([initial_encoding], current_encoding)
    return not match[0]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_camera', methods=['GET'])
def start_camera():
    return "Camera started."

@app.route('/stop_camera', methods=['GET'])
def stop_camera():
    return "Camera stopped."

@app.route('/status')
def status():
    try:
        status = status_queue.get_nowait()
    except queue.Empty:
        status = {"lighting": "Unknown", "blur": "Unknown"}
    return jsonify(status)

@app.route('/capture', methods=['POST'])
def capture():
    global initial_photo
    try:
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            logging.error("Failed to open camera.")
            return "Failed to open camera.", 500
        success, initial_photo = camera.read()
        if not success:
            logging.error("Failed to capture initial photo.")
            return "Failed to capture initial photo.", 500
        camera.release()
        logging.info("Initial photo captured successfully.")
        return "Initial photo captured successfully."
    except Exception as e:
        logging.exception("An error occurred while capturing initial photo.")
        return "An error occurred while capturing initial photo.", 500


@app.route('/stop_exam')
def stop_exam():
    # Implement your logic to stop the exam, such as logging the user out or marking the exam as completed
    return "Exam stopped due to tab switching.", 200


if __name__ == "__main__":
    app.run(debug=False, use_reloader=False)  # Disable debug and reloader
