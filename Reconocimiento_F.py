import cv2
import os

dataPath= 'F:/Git/Proyecto_Sistemas_Operativos/DATA'
imagePath = os.listdir(dataPath)
print('imgPath', imagePath)

face_recognizer = cv2.face.LBPHFaceRecognizer.create()

face_recognizer.read('ModeloFaceFrontalData.xml')
cap=cv2.VideoCapture(0,cv2.CAP_DSHOW)

faceClassif = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while True:
    ret,frame= cap.read()
    if ret == False:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    auxFrame = gray.copy()

    faces = faceClassif.detectMultiScale(gray,1.1,4)

    for (x,y,w,h) in faces:
        rostro = auxFrame[y:y+h, x:x+w]
        rostro= cv2.resize(rostro,(150,150),interpolation=cv2.INTER_CUBIC)
        resutl = face_recognizer.predict(rostro)
        cv2.putText(frame,'{}'.format(resutl),(x,y-5),1,1.3,(255,255,0),1,cv2.LINE_AA)

        if resutl[1]<85:
            cv2.putText(frame,'{}'.format(imagePath[resutl[0]]),(x,y-25),2,1.1,(0,255,0),0,cv2.LINE_AA)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        else:
            cv2.putText(frame, 'Desconocido',(x,y-20),2,0.8,(0,0,255),1,cv2.LINE_AA )
            cv2.rectangle(frame,(x,y), (x+w,y+h),(0,0,255),2)
    cv2.imshow('frame',frame)
    k= cv2.waitKey(1)
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()  