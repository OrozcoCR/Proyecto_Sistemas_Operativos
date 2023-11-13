import cv2
import os
import numpy as np

data_path = os.path.join(os.path.dirname(__file__), "..","DATA")


def train_model():

    people_list = os.listdir(data_path)
    print(people_list)

    labels = []
    facesData = []
    label = 0

    for name in people_list:
        person_path = os.path.join(data_path, name)
        print(f"Entrenando con {name}...")

        for file_name in os.listdir(person_path):
            # print('Rostros: ', nameDir +'/'+fileName)
            labels.append(label)

            file_path = os.path.join(person_path, file_name)
            facesData.append(cv2.imread(file_path, 0))

            # cv2.imshow('image',image)
            # cv2.waitKey(10)
        label = label + 1

    # cv2.destroyAllWindows()

    # print('labels= ',labels)
    # print('Numero de etiquetas: ', np.count_nonzero(np.array(labels)==0))
    # print('Numero de etiquetas: ', np.count_nonzero(np.array(labels)==1))

    face_recognizer = cv2.face.LBPHFaceRecognizer.create()
    print("Entrenando...")
    face_recognizer.train(facesData, np.array(labels))
    face_recognizer.write('models/ModeloFaceFrontalData.xml')
    print("Modelo Guardado")
