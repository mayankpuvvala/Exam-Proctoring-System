import cv2
import dlib
import numpy as np
from flask import Flask, render_template, Response, jsonify
from flask_cors import CORS
import pygame
import mediapipe as mp
import time
import face_recognition

app = Flask(__name__)
CORS(app)

# Global variables
camera = None
tab_switch_count = 0
TAB_SWITCH_LIMIT = 5
alarm_active = False
verified_user = None

# Initialize Pygame for playing the alarm sound
pygame.mixer.init()
alarm_sound = pygame.mixer.Sound("alarm.wav")

# Load face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def get_camera():
    global camera
    if camera is None:
        camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    return camera

def release_camera():
    global camera
    if camera is not None:
        camera.release()
        camera = None

def generate_frames():
    while True:
        try:
            success, frame = get_camera().read()
            if not success:
                print("Failed to capture frame")
                time.sleep(0.1)  # Wait a bit before trying again
                continue

            frame = detect_head_pose(frame)

            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                print("Failed to encode frame")
                continue
            
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        except Exception as e:
            print(f"Error in generate_frames: {str(e)}")
            time.sleep(1)  # Wait a bit before trying again

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

            # print(f"Head pose angles: x={x}, y={y}")

            if y < -10 or y > 10 or x < -5:
                if not alarm_active:
                    print("Starting alarm")
                    alarm_active = True
                    start_alarm()
            else:
                if alarm_active:
                    print("Stopping alarm")
                    stop_alarm()

            text = f"x: {int(x)} | y: {int(y)}"
            cv2.putText(image, text, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    return image

def start_alarm():
    global alarm_active
    alarm_sound.play(loops=-1)  # Play in an infinite loop (-1)

def stop_alarm():
    global alarm_active
    alarm_sound.stop()
    alarm_active = False

def check_lighting(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    if mean_brightness < 50:
        return "Too Dark"
    elif mean_brightness > 200:
        return "Too Bright"
    return "Good"

def check_blur(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if laplacian_var < 100:
        return "Blurry"
    return "Clear"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    print("Status route called")
    global verified_user
    if verified_user is None:
        print("No verified user")
        return jsonify({
            'lighting': 'N/A',
            'blur': 'N/A',
            'face_detected': False,
            'verified_user': False
        })

    print("Attempting to get camera")
    success, frame = get_camera().read()
    print(f"Camera read success: {success}")
    if not success:
        print("Failed to capture frame")
        return jsonify({'error': 'Failed to capture frame'}), 500

    print("Checking lighting")
    lighting = check_lighting(frame)
    print("Checking blur")
    blur = check_blur(frame)
    print("Detecting faces")
    face_locations = face_recognition.face_locations(frame)
    face_detected = len(face_locations) > 0
    print(f"Face detected: {face_detected}")

    print("Returning status")
    return jsonify({
        'lighting': lighting,
        'blur': blur,
        'face_detected': face_detected,
        'verified_user': verified_user is not None
    })

@app.route('/tab_switch')
def tab_switch():
    global tab_switch_count
    tab_switch_count += 1
    if tab_switch_count >= TAB_SWITCH_LIMIT:
        return jsonify({"message": "Exam stopped due to excessive tab switching."}), 403
    return jsonify({"tab_switches": tab_switch_count}), 200

@app.route('/stop_camera', methods=['GET'])
def stop_camera():
    release_camera()
    return "Camera stopped."

if __name__ == '__main__':
    app.run(debug=True)
