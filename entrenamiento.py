import cv2
import os
import numpy as np

dataPath = 'F:/Git/Proyecto_Sistemas_Operativos/DATA'

peopleList = os.listdir(dataPath)
print(peopleList)

labels = []
facesData = []
label = 0

for nameDir in peopleList:
    personPath = dataPath + '/'+nameDir
    print('leyendo imagenes....')

    for fileName in os.listdir(personPath):
        #print('Rostros: ', nameDir +'/'+fileName)
        labels.append(label)

        facesData.append(cv2.imread(personPath +'/'+fileName,0))
        image = cv2.imread(personPath + '/'+fileName,0)

        #cv2.imshow('image',image)
        #cv2.waitKey(10)
    label= label +1

#cv2.destroyAllWindows()

#print('labels= ',labels)
#print('Numero de etiquetas: ', np.count_nonzero(np.array(labels)==0))
#print('Numero de etiquetas: ', np.count_nonzero(np.array(labels)==1))

face_recognizer = cv2.face.LBPHFaceRecognizer.create()
print("entrenando----")
face_recognizer.train(facesData,np.array(labels))
face_recognizer.write('ModeloFaceFrontalData.xml')
print("Modelo Guardado")
