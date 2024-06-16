import cv2

camera = cv2.VideoCapture(0)
if not camera.isOpened():
    print("Failed to open camera.")
    exit()

while True:
    success, frame = camera.read()
    if not success:
        print("Failed to read frame from camera.")
        break

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
