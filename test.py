import cv2
from google.cloud import vision
import threading
import time

# Initialize the Google Cloud Vision client
client = vision.ImageAnnotatorClient()

# Load the cascade classifier for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Initialize the camera
cap = cv2.VideoCapture(0)

# Define the names for emotion likelihoods
likelihood_name = (
    "UNKNOWN",
    "VERY_UNLIKELY",
    "UNLIKELY",
    "POSSIBLE",
    "LIKELY",
    "VERY_LIKELY",
)

# List to store face images for emotion detection
face_images = []

# Function to detect emotions in the face
def detect_emotions(face_img):
    image = vision.Image(content=face_img)
    response = client.face_detection(image=image)
    faces = response.face_annotations

    # Print the detected emotions
    for face in faces:
        print(f"Emotion in the face: {likelihood_name[face.anger_likelihood]}")
        print(f"Happiness: {likelihood_name[face.joy_likelihood]}")
        print(f"Surprise: {likelihood_name[face.surprise_likelihood]}")
        print(f"Sorrow: {likelihood_name[face.sorrow_likelihood]}")

# Thread for emotion detection
def emotion_detection_thread():
    while True:
        if len(face_images) > 0:
            face_img = face_images.pop(0)
            detect_emotions(face_img)

# Create and start the emotion detection thread
emotion_thread = threading.Thread(target=emotion_detection_thread)
emotion_thread.daemon = True
emotion_thread.start()

# Initialize variables for face capture and emotion detection
last_detection_time = time.time()
detection_interval = 1  # Emotion detection every 1 second

while True:
    _, img = cap.read()

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        # Draw a rectangle around the detected face
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Capture the face region
        face_img = img[y:y + h, x:x + w]

        # Convert the face region to binary format
        _, buffer = cv2.imencode('.jpg', face_img)
        face_content = buffer.tobytes()

        # Add the face image to the queue for emotion detection
        face_images.append(face_content)

    # Show the image with detected faces
    cv2.imshow('img', img)

    # Check for emotion detection every second
    current_time = time.time()
    if current_time - last_detection_time >= detection_interval:
        last_detection_time = current_time

    # Wait for the 'ESC' key to exit
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

# Release the camera
cap.release()
