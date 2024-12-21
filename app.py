<<<<<<< HEAD
import cv2
import dlib
import numpy as np
from flask import Flask, render_template, Response, jsonify, request
from flask_cors import CORS
import pygame
import mediapipe as mp
import time
import speedtest  # Add speedtest-cli for internet speed check

app = Flask(__name__)
CORS(app)

# Global variables
camera = None
tab_switch_count = 0
TAB_SWITCH_LIMIT = 15
alarm_active = False
score = 0  # Score for malpractice
phones = 0  # Number of detected phones
inactive_time = 0  # Time spent inactive
last_used_website = "None"  

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
        print("Attempting to initialize the camera...")
        camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not camera.isOpened():
            print("Error: Unable to access the camera.")
            raise Exception("Camera not accessible.")
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        camera.set(cv2.CAP_PROP_FPS, 30)
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

import time

alarm_start_time = None  
def detect_head_pose(frame):
    global alarm_active, alarm_start_time

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

            if y < -10 or y > 5 or x < -5:
                if not alarm_active:
                    alarm_start_time = time.time()  # Record the time when the user first looks away
                    alarm_active = True
                    start_alarm()
                else:
                    elapsed_time = time.time() - alarm_start_time
                    if elapsed_time > 3:  # Check if the user has been looking away for more than 3 seconds
                        update_score(True)
                        malpractice_reasons.append("Looking away from the screen for too long.")
            else:
                if alarm_active:
                    stop_alarm()
                    alarm_active = False
                    alarm_start_time = None  # Reset the start time

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
    if frame is None or frame.size == 0:
        return "Error: Invalid Frame"
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    if mean_brightness < 50:
        return "Too Dark"
    elif mean_brightness > 100:
        return "Too Bright"
    return "Good"



def check_blur(frame):
    if frame is None or frame.size == 0:
        return "Error: Invalid Frame"
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if laplacian_var < 50:
        return "Blurry"
    return "Clear"
def update_score(condition_met):
    global score
    if condition_met:
        score += 1




@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')



malpractice_reasons = []

@app.route('/status', methods=['GET'])
def status():
    global score, human_count, phones, last_used_website, inactive_time, alarm_active

    try:
        # Capture a frame for analysis
        success, frame = get_camera().read()
        if not success:
            return jsonify({"error": "Failed to capture frame"}), 500

        # Detect faces and phones using Roboflow
        human_count, phones = count_faces_and_phones_in_frame(frame)
        smoothed_human_count = consistent_human_count(human_count)

        # Initialize lists for malpractice reasons and user suggestions
        malpractice_reasons = []
        user_suggestions = []

        # Check for malpractice conditions
        if smoothed_human_count > 1:
            update_score(True)
            malpractice_reasons.append("Multiple faces detected in the frame.")
        if phones > 0:
            update_score(True)
            malpractice_reasons.append(f"{phones} phone(s) detected in the frame.")
        if alarm_active:
            update_score(True)
            malpractice_reasons.append("Looking away from the screen for too long.")

        # Check lighting conditions
        lighting = check_lighting(frame)
        if lighting == "Too Dark":
            user_suggestions.append("Increase brightness.")
        elif lighting == "Too Bright":
            user_suggestions.append("Reduce brightness.")

        # Check for blur
        blur_status = check_blur(frame)
        if blur_status == "Blurry":
            user_suggestions.append("Adjust focus or clean lens.")

        # Return status with malpractice reasons and user suggestions
        return jsonify({
            "human_count": smoothed_human_count,
            "phones": phones,
            "lighting": lighting,
            "blur_status": blur_status,
            "score": score,
            "malpractice_reasons": malpractice_reasons,
            "user_suggestions": user_suggestions,
            "last_used_website": last_used_website,
            "inactive_time": inactive_time,
            "tab_switches": tab_switch_count
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500




total_inactive_time = 0
last_accessed_website = "None"

@app.route('/tab_switch', methods=['POST'])
def tab_switch():
    global total_inactive_time, tab_switch_count, last_accessed_website

    try:
        # Parse the data from the request
        data = request.get_json()
        inactive_duration = data.get('inactive_duration', 0)
        accessed_website = data.get('last_accessed_website', 'Unknown')

        # Increment tab switch count and update total inactive time
        if inactive_duration > 0:  # Ensure we only add meaningful durations
            total_inactive_time += inactive_duration

        if accessed_website and accessed_website != "http://127.0.0.1:5000/":
            tab_switch_count += 1
            last_accessed_website = accessed_website

        # Return consistent data structure
        return jsonify({
            'tab_switches': tab_switch_count,
            'total_inactive_time': round(total_inactive_time, 2),  # Round to 2 decimal places
            'last_accessed_website': last_accessed_website or "Unknown"
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500




@app.route('/stop_camera', methods=['GET'])
def stop_camera():
    release_camera()
    return "Camera stopped."

import speedtest

def update_speed_test():
    try:
        st = speedtest.Speedtest()
        st.get_best_server()  # Select the best server based on ping
        download_speed = st.download() / 1_000_000  # Convert to Mbps
        upload_speed = st.upload() / 1_000_000  # Convert to Mbps
        return round(download_speed, 2), round(upload_speed, 2)
    except Exception as e:
        print(f"Error in speed test: {e}")
        return 0.0, 0.0  # Return default values in case of error


@app.route('/speed_test', methods=['GET'])
def speed_test():
    download_speed, upload_speed = update_speed_test()
    return jsonify({
        'download': download_speed,
        'upload': upload_speed
    }), 200

from roboflow import Roboflow

# Initialize Roboflow
rf = Roboflow(api_key="JtOmdCES5A6PX2B69t4o")
project = rf.workspace().project("exam-proctoring-zewbp")
model = project.version(2).model

def count_faces_and_phones_in_frame(frame):
    global face_mesh

    # Mediapipe for face detection
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)

    # Count faces
    face_count = len(results.multi_face_landmarks) if results.multi_face_landmarks else 0

    # Save the frame as a temporary image for Roboflow
    temp_image_path = "temp_frame.jpg"
    cv2.imwrite(temp_image_path, frame)

    # Run Roboflow inference
    predictions = model.predict(temp_image_path).json()

    # Count phones detected by Roboflow
    phones = 0
    for prediction in predictions.get("predictions", []):
        if prediction["class"] == "Phone" and prediction["confidence"] > 0.8:
            phones += 1


    return face_count, phones



human_count_buffer = []

def consistent_human_count(current_count):
    global human_count_buffer
    human_count_buffer.append(current_count)
    if len(human_count_buffer) > 5:  # Maintain a buffer of the last 5 frames
        human_count_buffer.pop(0)
    return round(sum(human_count_buffer) / len(human_count_buffer))  # Average count


@app.route('/human_count', methods=['GET'])
def human_count():
    try:
        success, frame = get_camera().read()
        if not success or frame is None:
            return jsonify({'error': 'Failed to capture frame'}), 500

        # Use the new combined function for face and phone detection
        faces, phones = count_faces_and_phones_in_frame(frame)

        # Malpractice logic: More than one face or any phone detected
        malpractice = faces > 1 or phones > 0

        return jsonify({
            'face_count': faces,
            'phones': phones,
            'malpractice': malpractice
        }), 200
    except Exception as e:
        print(f"Error in /human_count: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    try:
        app.run(debug=True)
    except Exception as e:
        print(f"An error occurred during app run: {e}")
=======
from flask import Flask, render_template, Response, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO
from camera_manager import generate_frames, get_camera, release_camera
from monitoring import check_lighting, check_blur, detect_face
from flask import Flask, jsonify
import speedtest
import threading
import time


app = Flask(__name__)
CORS(app)
socketio = SocketIO(app)

# Global variables
tab_switch_count = 0
TAB_SWITCH_LIMIT = 5
inactive_time = 0

@app.route('/')
def index():
    return render_template('index1.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    success, frame = get_camera().read()
    if not success:
        return jsonify({'error': 'Failed to capture frame'}), 500

    lighting = check_lighting(frame)
    blur = check_blur(frame)
    face_detected = detect_face(frame)

    return jsonify({
        'lighting': lighting,
        'blur': blur,
        'face_detected': face_detected,
        'tab_switch_count': tab_switch_count,
        'inactive_time': inactive_time
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


# Global variable to store current speeds
current_speed = {
    "download": 0,
    "upload": 0
}


def speed_test():
    global current_speed
    st = speedtest.Speedtest()

    while True:
        st.get_best_server()
        current_speed["download"] = st.download() / 1_000_000  # Convert to Mbps
        current_speed["upload"] = st.upload() / 1_000_000  # Convert to Mbps
        time.sleep(8)  # Update speed every 8 seconds

@app.route('/speed')
def get_speed():
    return jsonify(current_speed)

if __name__ == '__main__':
    # Start the speed test in a separate thread
    speed_thread = threading.Thread(target=speed_test, daemon=True)
    speed_thread.start()
    app.run(debug=True)

>>>>>>> 816673b1081b2fdf23d7b16ba26059b70dde7678
