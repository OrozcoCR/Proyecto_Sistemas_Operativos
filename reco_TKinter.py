import cv2
import os
import imutils
import tkinter as tk
from tkinter import Button, Label
from PIL import Image, ImageTk

count = 0

def update_camera_frame():
    global count  # Declare count as a global variable

    ret, frame = cap.read()
    if ret:
        frame = imutils.resize(frame, width=320)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        auxFrame = frame.copy()
        faces = faceClassif.detectMultiScale(gray, 1.1, 4)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            rostro = auxFrame[y:y + h, x: x+w]
            rostro = cv2.resize(rostro, (720, 720), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(personPath + '/rostro_{}.jpg'.format(count), rostro)
            count = count + 1
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        camera_label.imgtk = imgtk
        camera_label.config(image=imgtk)
        
        if count >= 301:
            #Add extra code that jumps to other functions
            return  # Stop updating

        # Update every 10 milliseconds
        camera_label.after(10, update_camera_frame)
    else:
        # Retry every 10 milliseconds if frame capture fails
        camera_label.after(10, update_camera_frame)

def start_camera_capture():
    global cap, faceClassif, personPath, count
    personName = 'Estefanía'
    dataPath = '/Users/estefania/Documents/GitHub/Proyecto_Sistemas_Operativos/DATA'
    personPath = dataPath + '/' + personName

    if not os.path.exists(personPath):
        print('CARPETA CREADA', personPath)
        os.makedirs(personPath)

    cap = cv2.VideoCapture(0)
    faceClassif = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    count = 0
    update_camera_frame()

# Create a Tkinter window
root = tk.Tk()
root.title("Camera Capture")

# Create buttons
start_button = Button(root, text="Agregar persona", command=start_camera_capture)
button2 = Button(root, text="Iniciar reconocimiento")
exit_button = Button(root, text="Salir", command=root.destroy)  # Exit button

# Pack buttons
start_button.pack()
button2.pack()
exit_button.pack()  # Add the exit button

# Create a label for the camera window
camera_label = Label(root)
camera_label.pack()  # Use pack for the camera frame

# Center the window on the screen
window_width = 420  # Set the desired width
window_height = 320  # Set the desired height
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
x = (screen_width - window_width) // 2
y = (screen_height - window_height) // 2
root.geometry(f"{window_width}x{window_height}+{x}+{y}")

# Start the Tkinter main loop
root.mainloop()