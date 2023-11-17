import cv2
import os
import time

dataPath = os.path.join(os.path.dirname(__file__), "..", "DATA")
imagePath = os.listdir(dataPath)
print("imgPath", imagePath)


def test():
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
    recognized_start_time = None  # Initialize the start time for recognition

    while True:
        ret, frame = cap.read()
        if ret == False:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        auxFrame = gray.copy()

        faces = faceClassif.detectMultiScale(gray, 1.3, 4)

        recognized = False  # Flag to track if recognition occurs
        name = None

        for x, y, w, h in faces:
            rostro = auxFrame[y : y + h, x : x + w]
            rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
            result = face_recognizer.predict(rostro)
            cv2.putText(
                frame,
                "{}".format(result),
                (x, y - 5),
                1,
                1.3,
                (255, 255, 0),
                1,
                cv2.LINE_AA,
            )

            if result[1] < 84:
                name = imagePath[result[0]]
                cv2.putText(
                    frame,
                    "{}".format(imagePath[result[0]]),
                    (x, y - 25),
                    2,
                    1.1,
                    (0, 255, 0),
                    0,
                    cv2.LINE_AA,
                )
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                if recognized_start_time is None:
                    recognized_start_time = (
                        time.time()
                    )  # Start the timer for recognition
                elif (
                    time.time() - recognized_start_time >= 3
                ):  # Check if 3 seconds have passed
                    recognized = True  # Set recognized to True

            else:
                cv2.putText(
                    frame,
                    "Desconocido",
                    (x, y - 20),
                    2,
                    0.8,
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA,
                )
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                recognized_start_time = None  # Reset the timer for recognition

        if recognized:
            break  # Exit the loop when recognition occurs for 3 seconds

        cv2.imshow("frame", frame)
        k = cv2.waitKey(1)
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print(recognized)
    print(name)
    return name  # Return True if recognized for 3 seconds, otherwise False
