import cv2
import os
import threading
from google.cloud import vision
import tkinter as tk
from tkinter import Button, Label, filedialog
from PIL import Image, ImageTk
import json

classifier_path = ""
dataPath = os.path.join(os.path.dirname(__file__), "..", "DATA")
imagePath = os.listdir(dataPath)
print("imgPath", imagePath)
camera_label = None
data = []

event = threading.Event()


def detect_emotions(face_img):
    global data
    image = vision.Image(content=face_img)
    response = client.face_detection(image=image)
    faces = response.face_annotations

    for face in faces:
        emotionSet = {
            "anger": face.anger_likelihood,
            "joy": face.joy_likelihood,
            "surprise": face.surprise_likelihood,
            "sorrow": face.sorrow_likelihood,
        }
        print(f"Anger: {face.anger_likelihood.name}")
        print(f"Joy: {face.joy_likelihood.name}")
        print(f"Surprise: {face.surprise_likelihood.name}")
        print(f"Sorrow: {face.sorrow_likelihood.name}")
        print("\n")
        data.append(emotionSet)

def emotion_detection_thread(event: threading.Event):
    while True:
        if event.is_set():
            print('The thread was stopped prematurely.')
            break
        if len(face_images) > 0:
            face_img = face_images.pop(0)
            detect_emotions(face_img)


def stop_camera_feed(cap, path, data):
    cap.release()
    cv2.destroyAllWindows()
    event.set()
    print('stop')
    with open(os.path.join(path, 'data.json'), 'w') as file:
        json.dump(data, file)


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


def mostrar_porcentajes():
    file_path = filedialog.askopenfilename(title="Seleccionar archivo JSON", filetypes=[("Archivos JSON", "*.json")])
    # Leer datos desde el archivo JSON
    with open(file_path, 'r') as file:
        data = json.load(file)

    totals = [sum(emotions.values()) for emotions in data]

    # Calcular los porcentajes para cada emoción en cada registro
    percentages = [{emotion: count / total * 100 for emotion, count in emotions.items()} for emotions, total in zip(data, totals)]

    # Calcular el promedio de los porcentajes para cada emoción
    average_percentages = {emotion: sum(p[emotion] for p in percentages) / len(percentages) for emotion in data[0]}

    # Imprimir los resultados
    print("Porcentajes promedio:")
    for emotion, percentage in average_percentages.items():
        print(f"{emotion}: {percentage:.2f}%")


def emotionsMain(path):
    global cap, faceClassif, client, face_images, window_width, window_height, root, camera_label, data, emotion_detection_thread, emotion_thread
    
    print(path)
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
        root, text="Generar reporte", command=lambda: stop_camera_feed(cap, path, data)
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

    emotion_thread = threading.Thread(target=emotion_detection_thread, args=(event,))
    emotion_thread.daemon = True
    emotion_thread.start()

    root.mainloop()


if __name__ == "__main__":
    emotionsMain()