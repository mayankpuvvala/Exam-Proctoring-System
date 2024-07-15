import cv2
import numpy as np
from flask import Flask, jsonify, render_template, Response, request
import face_recognition
from flask_cors import CORS
import pygame
import os
import signal
from face_recognition1 import FaceVerification
import time

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
alarm_sound = pygame.mixer.Sound("alarm.wav")  # Make sure you have this file in your project directory

# def get_camera():
#     global camera
#     if camera is None:
#         for i in range(10):  # Try camera indices 0 to 9
#             camera = cv2.VideoCapture(i)
#             if camera.isOpened():
#                 camera.set(cv2.CAP_PROP_FPS, 30)
#                 camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
#                 camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
#                 return camera
#         raise Exception("No working camera found")
#     return camera

# def generate_frames():
#     while True:
#         success, frame = get_camera().read()
#         if not success:
#             break
#         else:
#             ret, buffer = cv2.imencode('.jpg', frame)
#             frame = buffer.tobytes()
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')




def get_camera():
    global camera
    if camera is None:
        for i in range(10):  # Try different camera indices
            camera = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if camera.isOpened():
                try:
                    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                    ret, frame = camera.read()
                    if ret:
                        return camera
                except cv2.error:
                    pass
            camera.release()
        raise Exception("No working camera found")
    return camera

# Global variables
show_tick = False
tick_start_time = 0

def generate_frames():
    global show_tick, tick_start_time
    while True:
        try:
            success, frame = get_camera().read()
            if not success:
                print("Failed to capture frame")
                time.sleep(0.1)  # Wait a bit before trying again
                continue
            
            if show_tick:
                current_time = time.time()
                if current_time - tick_start_time <= 2:
                    cv2.putText(frame, '✓', (frame.shape[1]//2, frame.shape[0]//2), 
                                cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 0), 5)
                else:
                    show_tick = False

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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

import base64

@app.route('/register', methods=['POST'])
def register_user():

    print("Registration attempt received")  # Add this line for debugging
    global face_verifier
    if face_verifier is None:
        face_verifier = FaceVerification("database")
    
    name = request.form.get('name')
    photo_data = request.form.get('photo')
    
    if not name or not photo_data:
        print(f"Missing data: name={bool(name)}, photo={bool(photo_data)}")  # Add this line for debugging
        return jsonify({'error': 'Name and photo are required'}), 400
    
    try:
        # Remove the data URL prefix
        photo_data = photo_data.split(',')[1]
        
        # Decode the base64 string
        photo_bytes = base64.b64decode(photo_data)
        
        # Save the image
        filename = f"{name}.jpg"
        with open(os.path.join("database", filename), "wb") as f:
            f.write(photo_bytes)
        
        # Reload the database
        face_verifier.load_database()
        print(f"User {name} registered successfully")  # Add this line for debugging
        return jsonify({'message': f'User {name} registered successfully'}), 200
    except Exception as e:
        print(f"Registration failed: {str(e)}")  # Add this line for debugging
        return jsonify({'error': f'Registration failed: {str(e)}'}), 500
    
    
    
@app.route('/verify_user', methods=['POST'])
def verify_user():
    global face_verifier, verified_user, show_tick, tick_start_time
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
            show_tick = True
            tick_start_time = time.time()
            return jsonify({'message': f'Verification successful. Welcome, {result}!'}), 200
        else:
            return jsonify({'error': 'Verification failed. Unknown person.'}), 403
    except Exception as e:
        return jsonify({'error': f'Verification failed: {str(e)}'}), 500
    
print("Initializing camera...")
camera = get_camera()
print("Camera initialized successfully")


@app.route('/status')
def status():
    global verified_user
    success, frame = get_camera().read()
    if success:
        # Convert frame to BGR if it's not already
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        
        lighting = check_lighting(frame)
        blur = check_blur(frame)
        
        face_locations = face_recognition.face_locations(frame)
        face_detected = len(face_locations) > 0
        
        if not face_detected:
            start_alarm()
        else:
            stop_alarm()
            
        if face_detected and verified_user:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_encoding = face_recognition.face_encodings(rgb_frame, face_locations)[0]
            is_verified, _ = face_verifier.verify_face_encoding(face_encoding)
            if not is_verified:
                start_alarm()
        
        return jsonify({
            'lighting': lighting,
            'blur': blur,
            'face_detected': face_detected,
            'verified_user': verified_user is not None
        })
    else:
        return jsonify({'error': 'Failed to capture frame'}), 500

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
    app.run(debug=True, threaded=True)