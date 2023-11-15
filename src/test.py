import cv2
import os
import threading
from google.cloud import vision
import tkinter as tk
from tkinter import Button, Label
from PIL import Image, ImageTk

classifier_path = ""
dataPath = os.path.join(os.path.dirname(__file__), "..", "DATA")
imagePath = os.listdir(dataPath)
print("imgPath", imagePath)
camera_label = None


def detect_emotions(face_img):
    image = vision.Image(content=face_img)
    response = client.face_detection(image=image)
    faces = response.face_annotations

    for face in faces:
        print(f"Anger: {face.anger_likelihood.name}")
        print(f"Joy: {face.joy_likelihood.name}")
        print(f"Surprise: {face.surprise_likelihood.name}")
        print(f"Sorrow: {face.sorrow_likelihood.name}")
        print("\n")


def emotion_detection_thread():
    while True:
        if len(face_images) > 0:
            face_img = face_images.pop(0)
            detect_emotions(face_img)


def stop_camera_feed(cap):
    cap.release()
    cv2.destroyAllWindows()


def update_camera_feed():
    ret, frame = cap.read()
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        auxFrame = gray.copy()

        faces = faceClassif.detectMultiScale(gray, 1.3, 4)

        for x, y, w, h in faces:
            face_img = frame[y : y + h, x : x + w]
            _, buffer = cv2.imencode(".jpg", face_img)
            face_content = buffer.tobytes()
            face_images.append(face_content)

        frame = cv2.resize(frame, (window_width, window_height))

        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        camera_label.imgtk = imgtk
        camera_label.config(image=imgtk)

        camera_label.after(10, update_camera_feed)


def start_camera_feed():
    global camera_label
    camera_label = Label(root)
    camera_label.grid(row=1, column=0, columnspan=3)
    update_camera_feed()


def main():
    global cap, faceClassif, client, face_images, window_width, window_height, root, camera_label
    face_recognizer = cv2.face_LBPHFaceRecognizer.create()
    classifier_path = os.path.join(
        os.path.dirname(__file__), "../models/haarcascade_frontalface_default.xml"
    )
    model_path = os.path.join(
        os.path.dirname(__file__), "../models/ModeloFaceFrontalData.xml"
    )
    face_recognizer.read(model_path)
    cap = cv2.VideoCapture(0)
    faceClassif = cv2.CascadeClassifier(classifier_path)
    client = vision.ImageAnnotatorClient()

    face_images = []

    root = tk.Tk()
    root.title("Emotions")

    stop_button = Button(
        root, text="Stop Camera", command=lambda: stop_camera_feed(cap)
    )
    start_button = Button(root, text="Start Camera", command=start_camera_feed)
    exit_button = Button(root, text="Exit", command=root.destroy)

    start_button.grid(row=0, column=0)
    stop_button.grid(row=0, column=1)
    exit_button.grid(row=0, column=2)

    window_width = 420
    window_height = 320
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x = (screen_width - window_width) // 2
    y = (screen_height - window_height) // 2
    root.geometry(f"{window_width}x{window_height}+{x}+{y}")
    root.state("zoomed")

    emotion_thread = threading.Thread(target=emotion_detection_thread)
    emotion_thread.daemon = True
    emotion_thread.start()

    root.mainloop()


if __name__ == "__main__":
    main()