import cv2
import os

dataPath = os.path.join(os.path.dirname(__file__), "..","DATA")
imagePath = os.listdir(dataPath)
print('imgPath', imagePath)

def test():
    face_recognizer = cv2.face.LBPHFaceRecognizer.create()
    classifier_path = os.path.join(os.path.dirname(__file__), '../models/haarcascade_frontalface_default.xml')

    model_path = os.path.join(os.path.dirname(__file__), '../models/ModeloFaceFrontalData.xml')

    face_recognizer.read(model_path)
    cap = cv2.VideoCapture(0)

    faceClassif = cv2.CascadeClassifier(classifier_path)

    while True:
        ret, frame = cap.read()
        if ret == False:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        auxFrame = gray.copy()

        faces = faceClassif.detectMultiScale(gray, 1.3, 4)

        for (x, y, w, h) in faces:
            rostro = auxFrame[y:y+h, x:x+w]
            rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
            result = face_recognizer.predict(rostro)
            cv2.putText(frame, '{}'.format(result), (x, y-5),
                        1, 1.3, (255, 255, 0), 1, cv2.LINE_AA)

            if result[1] < 5800:
                cv2.putText(frame, '{}'.format(
                    imagePath[result[0]]), (x, y-25), 2, 1.1, (0, 255, 0), 0, cv2.LINE_AA)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            else:
                cv2.putText(frame, 'Desconocido', (x, y-20), 2,
                            0.8, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.imshow('frame', frame)
        k = cv2.waitKey(1)
        if k == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
