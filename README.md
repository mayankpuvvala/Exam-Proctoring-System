# Watcher: Online Student Monitoring System

## Demo of the project
[![Watch the demo](https://github.com/mayankpuvvala/Exam-Proctoring-System/blob/main/Watcher_IMG.png)](https://youtu.be/5waUyec8q7s)

## Description

The **Watcher** is an innovative solution designed to enhance the learning experience during online classes. This project leverages real-time camera input from students' laptops or phones to assess their attentiveness during online meetings. It employs advanced techniques such as attention detection based on eye gaze and head pose, as well as tab activity tracking. Additionally, the system measures the active voice participation of students and analyzes voice signals for frequency and pitch. All results are presented through graphical visualizations for easy interpretation.

---

## Features

- **Real-Time Camera Input**: Capture live video feed from the student's laptop or mobile device.
- **Attention Detection**: Utilize eye gaze tracking and head pose estimation to determine student attentiveness.
- **Inactive Tab Time Measurement**: Track the duration a student has spent on inactive browser tabs.
- **Active Voice Time Measurement**: Monitor the duration of student participation during discussions using microphone input.
- **Frequency and Pitch Analysis**: Analyze voice signals to determine the frequency and pitch of the student’s contributions.
- **Graphical Visualization**: Present results and metrics through informative graphs for better understanding and analysis.

---

## Technical Stack

### Machine Learning (ML) & Deep Learning (DL)

- **OpenCV**: Used for image processing and real-time video capture.
- **TensorFlow/Keras**: For developing and training models for attention detection (if applicable).
- **Scikit-learn**: For data preprocessing and model evaluation (if applicable).
- **NumPy/Pandas**: For data manipulation and analysis.

### Computer Vision (CV)

- **Face Detection**: Implemented using Haar cascades or DNN models to detect faces in real-time.
- **Head Pose Estimation**: Utilizes geometric transformations to estimate head orientation.
- **Eye Gaze Tracking**: Analyzes eye movement to gauge student focus and attentiveness.

### Audio Processing

- **PyAudio**: For capturing and processing microphone input.
- **Librosa**: For analyzing audio signals, measuring frequency and pitch.

### Web Technologies

- **Flask**: A lightweight WSGI web application framework to create the server for handling real-time data.
- **Socket.IO**: To enable real-time bi-directional communication between the client and the server.

---

## Implementation Overview

### 1. Real-Time Camera Input

The system captures real-time video using OpenCV's `VideoCapture`, providing continuous frames for analysis.

### 2. Attention Detection

By combining face detection and eye gaze tracking, we determine the student's focus level. Head pose estimation is implemented using facial landmark detection.

### 3. Tab Activity Monitoring

Browser activity is monitored using JavaScript on the client side, sending periodic updates to the server to log inactive tab time.

### 4. Active Voice Monitoring

Using PyAudio, the application records audio input, processing it to calculate the duration of active voice participation.

### 5. Frequency and Pitch Analysis

Voice signals are analyzed using Librosa to extract key metrics such as frequency and pitch, offering insights into student engagement.

### 6. Graphical Visualization

Results are visualized using libraries like Matplotlib or Plotly, providing real-time feedback through graphs that depict attentiveness, active voice time, and more.

---

### Prerequisites

Ensure you have Python 3.12 installed, along with the following libraries:

```bash
pip install opencv-python flask flask-socketio numpy pandas tensorflow keras pyaudio librosa matplotlib
