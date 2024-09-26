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

