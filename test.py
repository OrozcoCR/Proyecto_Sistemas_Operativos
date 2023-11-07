import cv2
import os
from google.cloud import vision
import threading

dataPath = './DATA'
imagePath = os.listdir(dataPath)
print('imgPath', imagePath)

face_recognizer = cv2.face_LBPHFaceRecognizer.create()
face_recognizer.read('ModeloFaceFrontalData.xml')
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

faceClassif = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Initialize the Google Cloud Vision client
client = vision.ImageAnnotatorClient()

# List to store face images for emotion detection
face_images = []

# Function to detect emotions in the face
def detect_emotions(face_img):
    image = vision.Image(content=face_img)
    response = client.face_detection(image=image)
    faces = response.face_annotations

    # Print the detected emotions
    for face in faces:
        print(f"Emotion in the face: {face.anger_likelihood.name}")
        print(f"Happiness: {face.joy_likelihood.name}")
        print(f"Surprise: {face.surprise_likelihood.name}")
        print(f"Sorrow: {face.sorrow_likelihood.name}")

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

while True:
    ret, frame = cap.read()
    if ret == False:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    auxFrame = gray.copy()

    faces = faceClassif.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        rostro = auxFrame[y:y + h, x:x + w]
        rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
        result = face_recognizer.predict(rostro)
        cv2.putText(frame, '{}'.format(result), (x, y - 5), 1, 1.3, (255, 255, 0), 1, cv2.LINE_AA)

        if result[1] < 85:
            cv2.putText(frame, '{}'.format(imagePath[result[0]]), (x, y - 25), 2, 1.1, (0, 255, 0), 0, cv2.LINE_AA)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Capture the face region for emotion detection
            face_img = frame[y:y + h, x:x + w]
            _, buffer = cv2.imencode('.jpg', face_img)
            face_content = buffer.tobytes()

            # Add the face image to the queue for emotion detection
            face_images.append(face_content)
        else:
            cv2.putText(frame, 'Desconocido', (x, y - 20), 2, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    cv2.imshow('frame', frame)
    k = cv2.waitKey(1)
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
