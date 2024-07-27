from flask import Flask, render_template, Response, jsonify, request
from flask_cors import CORS
import cv2
import numpy as np
import time
import pygame
import face_recognition
import base64
import os
import signal

app = Flask(__name__)
CORS(app)

# Global variables
camera = None
tab_switch_count = 0
TAB_SWITCH_LIMIT = 5
alarm_active = False
face_verifier = None
verified_user = None

# Initialize Pygame for playing the alarm sound
pygame.mixer.init()
alarm_sound = pygame.mixer.Sound("alarm.wav")

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

def start_alarm():
    global alarm_active
    if not alarm_active:
        alarm_sound.play(loops=-1)
        alarm_active = True

def stop_alarm():
    global alarm_active
    if alarm_active:
        alarm_sound.stop()
        alarm_active = False

class FaceVerification:
    def __init__(self, database_path):
        self.database_path = database_path
        self.known_faces = {}
        self.load_database()

    def load_database(self):
        for filename in os.listdir(self.database_path):
            if filename.endswith(".jpg"):
                name = os.path.splitext(filename)[0]
                image_path = os.path.join(self.database_path, filename)
                image = face_recognition.load_image_file(image_path)
                encodings = face_recognition.face_encodings(image)
                if encodings:
                    self.known_faces[name] = encodings[0]

    def is_username_taken(self, username):
        return username in self.known_faces

    def add_face(self, name, encoding):
        self.known_faces[name] = encoding
        self.save_face(name, encoding)

    def save_face(self, name, encoding):
        filename = os.path.join(self.database_path, f"{name}.jpg")
        with open(filename, "wb") as f:
            f.write(encoding.tobytes())

    def verify_face(self, image_path):
        unknown_image = face_recognition.load_image_file(image_path)
        unknown_encodings = face_recognition.face_encodings(unknown_image)
        if not unknown_encodings:
            return False, None

        unknown_encoding = unknown_encodings[0]
        for name, known_encoding in self.known_faces.items():
            matches = face_recognition.compare_faces([known_encoding], unknown_encoding)
            if matches[0]:
                return True, name
        return False, None

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
@app.route('/register', methods=['POST'])
def register_user():
    print("Register route called")
    global face_verifier
    if face_verifier is None:
        print("Initializing face verifier")
        face_verifier = FaceVerification("database")
    
    name = request.json.get('name')
    photo_data = request.json.get('photo')
    
    print(f"Received data - Name: {bool(name)}, Photo: {bool(photo_data)}")
    if not name or not photo_data:
        print("Missing name or photo")
        return jsonify({'error': 'Name and photo are required'}), 400
    
    try:
        print("Processing photo data")
        photo_data = photo_data.split(',')[1]
        photo_bytes = base64.b64decode(photo_data)
        
        # Save the image permanently
        filename = os.path.join("database", f"{name}.jpg")
        with open(filename, "wb") as f:
            f.write(photo_bytes)
        
        # Check if the photo is optimal for face recognition
        image_data = face_recognition.load_image_file(filename)
        face_encoding = face_recognition.face_encodings(image_data)
        if not face_encoding:
            os.remove(filename)
            return jsonify({'error': 'No face detected in the photo. Please capture again.'}), 400
        
        # Save the face encoding with the user's name
        face_verifier.add_face(name, face_encoding[0])
        
        print("Registration successful")
        return jsonify({'message': f'User {name} registered successfully'}), 200
    except Exception as e:
        print(f"Registration failed: {str(e)}")
        return jsonify({'error': f'Registration failed: {str(e)}'}), 500



@app.route('/verify_user', methods=['POST'])
def verify_user():
    global face_verifier, verified_user
    if face_verifier is None:
        face_verifier = FaceVerification("database")
    
    photo_data = request.form.get('photo')
    
    if not photo_data:
        return jsonify({'error': 'Photo is required'}), 400
    
    try:
        # Remove the data URL prefix
        photo_data = photo_data.split(',')[1]
        
        # Decode the base64 string
        photo_bytes = base64.b64decode(photo_data)
        
        # Save the image temporarily
        temp_filename = 'temp_verification.jpg'
        with open(temp_filename, "wb") as f:
            f.write(photo_bytes)
        
        is_verified, result = face_verifier.verify_face(temp_filename)
        os.remove(temp_filename)
        
        if is_verified:
            verified_user = result
            return jsonify({'message': f'Verification successful. Welcome, {result}!'}), 200
        else:
            return jsonify({'error': 'Verification failed. Unknown person.'}), 403
    except Exception as e:
        return jsonify({'error': f'Verification failed: {str(e)}'}), 500

@app.route('/tab_switch')
def tab_switch():
    global tab_switch_count
    tab_switch_count += 1
    if tab_switch_count >= TAB_SWITCH_LIMIT:
        return jsonify({"message": "Exam stopped due to excessive tab switching."}), 403
    return jsonify({"tab_switches": tab_switch_count}), 200

@app.route('/stop_camera')
def stop_camera():
    global camera
    if camera is not None:
        camera.release()
        camera = None
    return jsonify({"message": "Camera stopped."}), 200

@app.route('/kill_app')
def kill_app():
    os.kill(os.getpid(), signal.SIGINT)
    return jsonify({"message": "Application terminated."}), 200


if __name__ == "__main__":
    print("Initializing camera...")
    camera = get_camera()
    print("Camera initialized successfully")
    app.run(debug=True, threaded=True)
