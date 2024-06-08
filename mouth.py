import time
import cv2
import dlib
import numpy as np
from imutils import face_utils
from scipy.spatial import distance as dist
import threading
import simpleaudio as sa
import pygame
import pyaudio
import webrtcvad
import collections
import sys

# Load face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

MOUTH_AR_THRESH = 0.7  # Threshold for mouth aspect ratio to determine if mouth is open
TALK_TIME_THRESH = 0.8  # Time threshold in seconds for continuous talking to trigger alarm
LIP_MOVE_THRESH = 0.8  # Time threshold in seconds for continuous lip movement to trigger alarm

def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[2], mouth[10])  # 51, 59
    B = dist.euclidean(mouth[4], mouth[8])   # 53, 57
    C = dist.euclidean(mouth[0], mouth[6])   # 49, 55
    mar = (A + B) / (2.0 * C)
    return mar

def play_alarm():
    wave_obj = sa.WaveObject.from_wave_file("alarm.wav")
    play_obj = wave_obj.play()
    play_obj.wait_done()

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
            return True
    
    return False

class VADAudioProcessor:
    def __init__(self):
        self.vad = webrtcvad.Vad(3)
        self.chunk_duration_ms = 30
        self.padding_duration_ms = 300
        self.chunk_size = int(16000 * self.chunk_duration_ms / 1000)
        self.padding_chunks = int(self.padding_duration_ms / self.chunk_duration_ms)
        self.num_padding_chunks = self.padding_chunks
        self.buffer = collections.deque(maxlen=self.padding_chunks)
        self.triggered = False

        self.pa = pyaudio.PyAudio()
        self.stream = self.pa.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=self.chunk_size)
    
    def read_chunk(self):
        return self.stream.read(self.chunk_size)

    def process_audio(self):
        while True:
            chunk = self.read_chunk()
            active = self.vad.is_speech(chunk, 16000)

            if active:
                self.buffer.append((chunk, active))
                num_voiced = len([f for f, speech in self.buffer if speech])
                if num_voiced > 0.9 * self.num_padding_chunks:
                    self.triggered = True
                    self.buffer = collections.deque(maxlen=self.padding_chunks)
            else:
                self.buffer.append((chunk, active))
                if self.triggered:
                    num_unvoiced = len([f for f, speech in self.buffer if not speech])
                    if num_unvoiced > 0.9 * self.num_padding_chunks:
                        self.triggered = False
                        self.buffer = collections.deque(maxlen=self.padding_chunks)
            
            yield self.triggered

def start_mouth_detection_with_alarm():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video capture.")
        return

    start_talking_time = None
    start_lip_move_time = None
    alarm_active = False

    vad_audio_processor = VADAudioProcessor()
    audio_thread = threading.Thread(target=vad_audio_processor.process_audio)
    audio_thread.daemon = True
    audio_thread.start()

    # Initialize Pygame
    pygame.init()
    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    pygame.display.set_caption("Mouth Detection Lock Screen")

    clock = pygame.time.Clock()
    running = True

    while running:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from camera.")
            break

        mouth_open = detect_mouth_opening(frame)
        person_talking = next(vad_audio_processor.process_audio())

        # Check for continuous talking
        if mouth_open or person_talking:
            if start_talking_time is None:
                start_talking_time = time.time()
            elif time.time() - start_talking_time > TALK_TIME_THRESH and not alarm_active:
                threading.Thread(target=play_alarm).start()
                alarm_active = True
        else:
            start_talking_time = None

        # Check for continuous lip movement
        if mouth_open:
            if start_lip_move_time is None:
                start_lip_move_time = time.time()
            elif time.time() - start_lip_move_time > LIP_MOVE_THRESH and not alarm_active:
                threading.Thread(target=play_alarm).start()
                alarm_active = True
        else:
            start_lip_move_time = None

        # Reset alarm state when no talking or lip movement is detected
        if not mouth_open and not person_talking:
            alarm_active = False

        # Convert frame to Pygame surface
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = np.rot90(frame)
        frame = pygame.surfarray.make_surface(frame)

        screen.blit(frame, (0, 0))
        pygame.display.update()

        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False

        clock.tick(30)  # Limit to 30 frames per second

    cap.release()
    pygame.quit()

