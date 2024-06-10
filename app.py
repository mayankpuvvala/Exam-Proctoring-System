import os
import sys
import time
import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
from flask import Flask, jsonify, render_template, Response
import pygame
import sounddevice as sd
from threading import Thread
import mediapipe as mp
import tensorflow as tf
import logging
import face_recognition

from mouth import start_mouth_detection_with_alarm, detect_mouth_opening

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

class AudioProcessor:
    def __init__(self):
        self.active_voice_time = 0
        self.active = False
        self.start_time = None

        self.audio_thread = Thread(target=self.update_active_voice_time)
        self.audio_thread.daemon = True

    def start_audio(self):
        global stream
        stream.start()
        self.audio_thread.start()

    def stop_audio(self):
        global stream
        stream.stop()
        stream.close()

    def update_active_voice_time(self):
        global stream
        while stream.active:
            data, overflowed = stream.read(1024)
            rms = np.sqrt(np.mean(data**2))

            if rms > threshold and not self.active:
                self.active = True
                self.start_time = time.time()
            elif rms <= threshold and self.active:
                self.active = False
                end_time = time.time()
                self.active_voice_time += end_time - self.start_time

audio_processor = AudioProcessor()

def start_alarm():
    global alarm_active
    pygame.mixer.music.load("alarm.wav")
    pygame.mixer.music.play(loops=-1)

def stop_alarm():
    global alarm_active
    pygame.mixer.music.stop()
    alarm_active = False

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
    # Convert the image from BGR color (which OpenCV uses) to RGB color
    rgb_frame = frame[:, :, ::-1]
    # Find all the faces in the current frame of video
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

def check_camera_quality(frame):
    if not check_lighting(frame):
        print("Lighting conditions are not ideal.")
    if not check_blur(frame):
        print("Camera is blurry or has oil on it.")
    if check_lighting(frame) and check_blur(frame):
        print("Camera quality is good.")

def generate_frames():
    global alarm_active, start_talking_time

    camera = cv2.VideoCapture(0)
    while True:
        success, frame = camera.read()
        if not success:
            break

        frame = detect_head_pose(frame)

        # Check for lighting and blur
        if not check_lighting(frame):
            print("Lighting conditions are not ideal.")
        if not check_blur(frame):
            print("Camera is blurry or has oil on it.")

        if not check_face(frame):
            # Play alarm if no face detected
            start_alarm()

        if detect_mouth_opening(frame):
            if start_talking_time is None:
                start_talking_time = time.time()
            elif time.time() - start_talking_time > TALK_TIME_THRESH and not alarm_active:
                start_alarm()
        else:
            start_talking_time = None
            stop_alarm()

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

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

@app.route('/start_audio', methods=['GET'])
def start_audio():
    audio_processor.start_audio()
    return "Audio started."

@app.route('/stop_audio', methods=['GET'])
def stop_audio():
    audio_processor.stop_audio()
    return "Audio stopped."

@app.route('/active_voice_time', methods=['GET'])
def active_voice_time():
    return jsonify({"active_voice_time": audio_processor.active_voice_time})

if __name__ == "__main__":
    app.run(debug=False, use_reloader=False)  # Disable debug and reloader
