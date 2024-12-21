# Watcher: Online Student Monitoring and Exam Proctoring System

![Watcher Demo](https://github.com/mayankpuvvala/Exam-Proctoring-System/blob/main/Watcher_IMG.png)

## Demo of the Project
[![Watch the demo](https://github.com/mayankpuvvala/Exam-Proctoring-System/blob/main/Watcher_IMG.png)](https://youtu.be/5waUyec8q7s)

---

## Description

**Watcher** is a robust solution for online student monitoring and exam proctoring, designed to ensure integrity and engagement during virtual sessions. Leveraging state-of-the-art machine learning, computer vision, and audio processing techniques, Watcher offers advanced features such as real-time behavior analysis, tab activity tracking, and voice participation monitoring.

The system also detects unauthorized behaviors such as the presence of phones, talking, and impersonation during exams. It incorporates custom-trained models to enforce exam security by tracking human count and monitoring for environmental manipulations like low visibility, blur, and lighting exploits. All results are visualized for ease of analysis.

---

## Features

### Student Monitoring

- **Attention Detection**: Monitors attentiveness using eye gaze tracking and head pose estimation.
- **Inactive Tab Monitoring**: Tracks time spent on inactive browser tabs during online classes.
- **Voice Participation Analysis**: Measures active speaking time, analyzing frequency and pitch.

### Exam Proctoring

- **Phone Detection**: Identifies phone presence in the video feed using a custom-trained YOLOv11 model.
- **Human Count Tracking**: Detects the number of individuals in the camera frame to prevent impersonation.
- **Tab Activity Logging**: Tracks unauthorized tab changes during exams and logs details for supervisors.
- **Environmental Manipulation Detection**:
  - **Lighting Issues**: Uses grayscale conversion to adjust brightness dynamically.
  - **Blur Detection**: Implements Laplacian blur detection to prevent camera manipulation.

---

## Technical Stack

### Machine Learning & Deep Learning

- **YOLOv11**: Custom-trained model for detecting phones and counting individuals.
- **MobileNetV2**: Used for facial recognition.
- **MediaPipe Face Mesh**: For 3D facial landmark detection.
- **TensorFlow/Keras**: Deep learning frameworks for additional model development.
- **Scikit-learn**: Preprocessing and model evaluation.

### Computer Vision

- **OpenCV**: Real-time image processing and video capture.
- **PnP Algorithm**: Used for precise face tracking and head pose estimation.
- **Grayscale Conversion**: For dynamic lighting adjustments.
- **Laplacian Blur Detection**: Identifies blur to counter camera manipulations.

### Audio Processing

- **Short-Time Fourier Transform (STFT)**: Detects background noise and speech violations.
- **Librosa**: Extracts frequency and pitch data for voice analysis.
- **PyAudio**: Captures and processes audio input.

### Web Technologies

- **Flask**: Backend framework for handling server-side logic.
- **Socket.IO**: Enables real-time communication between client and server.

### Visualization

- **Matplotlib/Plotly**: Generates graphical insights for attentiveness, voice participation, and violations.

---

## Implementation Overview

### 1. Real-Time Camera Input
Captures video frames using OpenCV for analysis of attention, human count, and environmental conditions.

### 2. Attention Detection
Combines facial recognition, eye gaze tracking, and head pose estimation for engagement analysis.

### 3. Exam Security
- **Human Detection**: Tracks human count using YOLOv11 to prevent impersonation.
- **Phone Detection**: Detects phones in the camera feed and stops the exam upon identification.
- **Tab Logging**: Monitors and logs any tab changes during the exam.

### 4. Audio Processing
- **Voice Activity**: Monitors speaking duration using STFT.
- **Noise Detection**: Identifies background voices to detect unauthorized help.

### 5. Visualization
Real-time metrics are visualized through graphs, offering insights into attention levels, violations, and other key parameters.

---

## Prerequisites

Ensure you have Python 3.12 installed, along with the necessary libraries:

```bash
pip install opencv-python flask flask-socketio numpy pandas tensorflow keras pyaudio librosa matplotlib
