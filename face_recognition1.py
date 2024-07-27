import cv2
import face_recognition
import os
import numpy as np

class FaceVerification:
    def __init__(self, database_path):
        self.database_path = database_path
        self.known_face_encodings = []
        self.known_face_names = []
        self.load_database()

    def load_database(self):
        for filename in os.listdir(self.database_path):
            if filename.endswith((".jpg", ".jpeg", ".png")):
                path = os.path.join(self.database_path, filename)
                name = os.path.splitext(filename)[0]
                image = face_recognition.load_image_file(path)
                encoding = face_recognition.face_encodings(image)[0]
                self.known_face_encodings.append(encoding)
                self.known_face_names.append(name)

    def capture_passport_photo(self):
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            cv2.imshow('Capture Passport Photo', frame)
            if cv2.waitKey(1) & 0xFF == ord('c'):
                cv2.imwrite('passport_photo.jpg', frame)
                break
        cap.release()
        cv2.destroyAllWindows()

    def verify_face(self, image_path):
        image = face_recognition.load_image_file(image_path)
        face_locations = face_recognition.face_locations(image)
        if len(face_locations) == 0:
            return False, "No face detected"

        face_encoding = face_recognition.face_encodings(image)[0]
        matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
        face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)

        if matches[best_match_index]:
            return True, self.known_face_names[best_match_index]
        else:
            return False, "Unknown person"

    def continuous_verification(self):
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            rgb_frame = frame[:, :, ::-1]
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                name = "Unknown"

                face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = self.known_face_names[best_match_index]

                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

            cv2.imshow('Video', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    verifier = FaceVerification("database")
    verifier.capture_passport_photo()
    is_verified, result = verifier.verify_face("passport_photo.jpg")
    if is_verified:
        print(f"Verification successful. Welcome, {result}!")
        verifier.continuous_verification()
    else:
        print(f"Verification failed. {result}")