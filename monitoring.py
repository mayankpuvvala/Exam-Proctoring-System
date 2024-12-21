<<<<<<< HEAD
import cv2
import numpy as np
import face_recognition

def check_lighting(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    if mean_brightness < 25:
        return "Too Dark"
    elif mean_brightness > 100:
        return "Too Bright"
    return "Good"

def check_blur(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if laplacian_var < 50:
        return "Blurry"
    return "Clear"

def detect_face(frame):
    face_locations = face_recognition.face_locations(frame)
    return len(face_locations) > 0
=======
import cv2
import numpy as np
import face_recognition

def check_lighting(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    if mean_brightness < 25:
        return "Too Dark"
    elif mean_brightness > 100:
        return "Too Bright"
    return "Good"

def check_blur(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if laplacian_var < 50:
        return "Blurry"
    return "Clear"

def detect_face(frame):
    face_locations = face_recognition.face_locations(frame)
    return len(face_locations) > 0
>>>>>>> 816673b1081b2fdf23d7b16ba26059b70dde7678
