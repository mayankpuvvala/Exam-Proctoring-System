import cv2
import os
import dlib
import numpy as np
import face_recognition
from flask import Flask, render_template, Response, redirect, url_for, request
import pygame

# Initialize Pygame for playing the alarm sound
pygame.mixer.init()

app = Flask(__name__)

# Load face detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Load the alarm sound
alarm_sound = pygame.mixer.Sound("alarm.wav")  # Replace with your alarm sound file

# Global variables
user_face_encoding = None
captured_photo_path = "captured_face.jpg"
camera = cv2.VideoCapture(0)

def start_alarm():
    pygame.mixer.music.load("alarm.wav")
    pygame.mixer.music.play(loops=-1)

def stop_alarm():
    pygame.mixer.music.stop()

def capture_photo():
    global user_face_encoding

    ret, frame = camera.read()
    if not ret:
        return False
    
    rgb_frame = frame[:, :, ::-1]
    face_locations = face_recognition.face_locations(rgb_frame)

    if len(face_locations) == 0:
        return False

    user_face_encoding = face_recognition.face_encodings(rgb_frame, face_locations)[0]
    cv2.imwrite(captured_photo_path, frame)
    return True

def check_face(frame):
    global user_face_encoding

    rgb_frame = frame[:, :, ::-1]
    face_locations = face_recognition.face_locations(rgb_frame)
    if len(face_locations) == 0:
        return False
    
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces([user_face_encoding], face_encoding)
        if not any(matches):
            start_alarm()
            return False
    stop_alarm()
    return True

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/capture', methods=['POST'])
def capture():
    if capture_photo():
        return 'Photo captured', 200
    return 'Failed to capture photo', 500

@app.route('/confirm')
def confirm():
    return render_template('confirm.html', image_path=captured_photo_path)

@app.route('/reset', methods=['POST'])
def reset():
    global user_face_encoding
    user_face_encoding = None
    if os.path.exists(captured_photo_path):
        os.remove(captured_photo_path)
    return 'Photo reset', 200

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/start_app', methods=['POST'])
def start_app():
    global camera
    camera.release()
    os.system("python app.py")
    return 'App started', 200

if __name__ == '__main__':
    app.run(debug=True, port=5001)
