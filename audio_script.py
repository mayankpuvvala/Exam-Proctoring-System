import cv2
import numpy as np
from flask import Flask, render_template, Response
import pygame
from threading import Thread
import time
import mediapipe as mp
from app import audio_processor  # Import the audio processor

app = Flask(__name__)

# Initialize Pygame for playing the alarm sound
pygame.mixer.init()

# Load the alarm sound
alarm_sound = pygame.mixer.Sound("alarm.wav")  # Replace "alarm.wav" with your alarm sound file

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

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

def start_alarm():
    global alarm_active
    pygame.mixer.music.load("alarm.wav")
    pygame.mixer.music.play(loops=-1)  # Play in an infinite loop (-1)

def stop_alarm():
    global alarm_active
    pygame.mixer.music.stop()
    alarm_active = False

def generate_frames():
    global alarm_active

    camera = cv2.VideoCapture(0)  # 0 for default camera
    while True:
        success, frame = camera.read()
        if not success:
            break

        frame = detect_head_pose(frame)

        if audio_processor.is_talking() and not alarm_active:
            start_alarm()
        elif not audio_processor.is_talking() and alarm_active:
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

@app.route('/start_audio', methods=['GET'])
def start_audio():
    audio_processor.start_audio()
    return "Audio started."

@app.route('/stop_audio', methods=['GET'])
def stop_audio():
    audio_processor.stop_audio()
    return "Audio stopped."

if __name__ == '__main__':
    app.run(debug=True)
