import cv2
from deepface import DeepFace

# Load face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


# Start capturing video from file
cap = cv2.VideoCapture("./file.mp4")


while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        break  # Break the loop when the video ends

    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    emotions = []
    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in faces:
        # Extract the face ROI (Region of Interest)
        face_roi = frame[y:y + h, x:x + w]

        # Perform emotion analysis on the face ROI
        result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)

        # Determine the dominant emotion
        emotion = result[0]['dominant_emotion']

        # Print the detected emotion
        emotions.append(emotion)
print(f"Detected Emotion: {max(set(emotions), key = emotions.count)} {emotions}")
# Release the capture
cap.release()
