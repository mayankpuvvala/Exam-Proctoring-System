import cv2
import dlib
from imutils import face_utils
from scipy.spatial import distance as dist

# Load face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

MOUTH_AR_THRESH = 0.7  # Threshold for mouth aspect ratio to determine if mouth is open

def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[2], mouth[10])  # 51, 59
    B = dist.euclidean(mouth[4], mouth[8])   # 53, 57
    C = dist.euclidean(mouth[0], mouth[6])   # 49, 55
    mar = (A + B) / (2.0 * C)
    return mar

def detect_mouth_opening(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)
    
    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        
        mouth = shape[48:68]
        mar = mouth_aspect_ratio(mouth)
        
        if mar > MOUTH_AR_THRESH:
            cv2.putText(frame, "Mouth Open!", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
    
    return frame
