import cv2
import os
import imutils
import tkinter as tk
from tkinter import Button, simpledialog, messagebox, Label, filedialog
from PIL import Image, ImageTk
from entrenamiento import train_model
from reconocimiento import test
from test import emotionsMain
from cloudStorage import download_entire_bucket, upload_folder_contents
import json

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

count = 0
PWD_PATH = os.path.dirname(__file__)
camera_label = None  # Define camera_label as a global variable
nextPath = ''


def update_camera_frame():
    global count, camera_label
    ret, frame = cap.read()
    if ret:
        frame = imutils.resize(frame, width=1080)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        auxFrame = frame.copy()
        faces = faceClassif.detectMultiScale(gray, 1.3, 4)
        for x, y, w, h in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            rostro = auxFrame[y: y + h, x: x + w]
            rostro = cv2.resize(rostro, (720, 720),
                                interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(personPath + "/rostro_{}.jpg".format(count), rostro)
            count = count + 1
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        camera_label.imgtk = imgtk
        camera_label.config(image=imgtk)

        if count >= 301:
            camera_label.destroy()
            messagebox.showwarning("Warning", "Actualizando modelo...")
            train_model()
            messagebox.showinfo("Info", "{} fue agreado/a".format(personName))
            # if test():
            #     root.quit()
            #     main()
            # else:
            #     messagebox.showinfo("Info", "La persona no fue reconocida")
            # return

        # Update every 10 milliseconds
        camera_label.after(10, update_camera_frame)
    else:
        # Retry every 10 milliseconds if frame capture fails
        camera_label.after(10, update_camera_frame)


def start_camera_capture():
    global cap, faceClassif, personPath, count, personName
    personName = simpledialog.askstring("Nuevo usuario", "Cual es su nombre?")
    if not personName:
        return
    personPath = os.path.join(PWD_PATH, "..", "DATA", personName)
    if not os.path.exists(personPath):
        print("CARPETA CREADA", personPath)
        os.makedirs(personPath)

    cap = cv2.VideoCapture(0)
    classifier_path = os.path.join(
        os.path.dirname(
            __file__), "../models/haarcascade_frontalface_default.xml"
    )
    faceClassif = cv2.CascadeClassifier(classifier_path)
    count = 0
    update_camera_frame()


def regonized():
    name = test()
    if name != None:
        root.destroy()
        emotionsMain(os.path.join(PWD_PATH, "..", "DATA", name))
    else:
        print("No reconocida")


def mostrar_porcentajes():
    file_path = filedialog.askopenfilename(
        title="Seleccionar archivo JSON", filetypes=[("Archivos JSON", "*.json")])
    # Leer datos desde el archivo JSON
    with open(file_path, 'r') as file:
        data = json.load(file)

    totals = [sum(emotions.values()) for emotions in data]

    # Calcular los porcentajes para cada emoción en cada registro
    percentages = [{emotion: count / total * 100 for emotion,
                    count in emotions.items()} for emotions, total in zip(data, totals)]

    # Calcular el promedio de los porcentajes para cada emoción
    average_percentages = {emotion: sum(
        p[emotion] for p in percentages) / len(percentages) for emotion in data[0]}

    # Imprimir los resultados
    print("Porcentajes promedio:")
    result_string = "\n".join(
        [f"{emotion}: {percentage:.2f}%" for emotion, percentage in average_percentages.items()])

    messagebox.showinfo('Porcentajes', result_string)



def create_gui():
    global camera_label, root
    root = tk.Tk()
    root.title("Emotions")
    # Create a label for the camera window
    camera_label = Label(root)
    camera_label.grid(row=1, columnspan=3)  # Use pack for the camera frame
    start_button = Button(root, text="Agregar persona",
                          command=start_camera_capture)
    button2 = Button(root, text="Iniciar reconocimiento", command=regonized)
    exit_button = Button(root, text="Salir", command=root.destroy)
    percentage = Button(root, text="Porcentaje", command=mostrar_porcentajes)
    guardar_datos = Button(root, text="Guardar datos", command=upload_folder_contents)
    cargar_datos = Button(root, text="Cargar datos", command=download_entire_bucket)


    start_button.grid(row=0, column=0)
    button2.grid(row=0, column=1)
    exit_button.grid(row=0, column=2)
    percentage.grid(row=0, column=3)
    guardar_datos.grid(row=0, column=4)
    cargar_datos.grid(row=0, column=5)


    window_width = 420
    window_height = 320
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x = (screen_width - window_width) // 2
    y = (screen_height - window_height) // 2
    root.geometry(f"{window_width}x{window_height}+{x}+{y}")
    root.state("zoomed")

    root.mainloop()


if __name__ == "__main__":
    create_gui()
