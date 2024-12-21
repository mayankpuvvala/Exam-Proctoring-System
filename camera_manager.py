<<<<<<< HEAD
import cv2
import time

camera = None

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
                time.sleep(0.1)  # Wait a bit before trying again
                continue
            
            # Encode the frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                print("Error encoding frame")
                continue

            # Convert the buffer to bytes
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        except Exception as e:
            print(f"Error in generate_frames: {str(e)}")
            time.sleep(1)  # Wait a bit before trying again
=======
import cv2
import time

camera = None

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
                time.sleep(0.1)  # Wait a bit before trying again
                continue
            
            # Encode the frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                print("Error encoding frame")
                continue

            # Convert the buffer to bytes
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        except Exception as e:
            print(f"Error in generate_frames: {str(e)}")
            time.sleep(1)  # Wait a bit before trying again
>>>>>>> 816673b1081b2fdf23d7b16ba26059b70dde7678
