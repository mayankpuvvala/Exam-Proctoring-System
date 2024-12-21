import cv2
import numpy as np
from flask import Flask, jsonify, render_template, Response

app = Flask(__name__)

def check_lighting(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    return mean_brightness

def check_blur(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var

@app.route('/lighting_blur_check')
def lighting_blur_check():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return jsonify({"error": "Could not open video capture"}), 500

    ret, frame = cap.read()
    cap.release()
    if not ret:
        return jsonify({"error": "Could not read frame from camera"}), 500

    brightness = check_lighting(frame)
    blur = check_blur(frame)

    # Define thresholds
    brightness_threshold = 100  # Adjust as needed for your environment
    blur_threshold = 100  # Adjust as needed based on testing

    ideal_lighting = brightness > brightness_threshold
    lens_clean = blur > blur_threshold

    response = {
        "ideal_lighting": ideal_lighting,
        "brightness": brightness,
        "lens_clean": lens_clean,
        "blur": blur
    }

    return jsonify(response)

@app.route('/')
def index():
    return render_template('lighting_blur_check.html')

@app.route('/check_lighting_blur', methods=['GET'])
def check_lighting_blur():
    return Response(lighting_blur_check(), mimetype='application/json')

if __name__ == '__main__':
    app.run(debug=True)
