import cv2
import os
import numpy as np

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def detect_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return None, None

    (x, y, w, h) = faces[0]
    return gray[y:y+h, x:x+w], (x, y, w, h)

def load_dataset(dataset_path):
    faces = []
    labels = []
    label_map = {}
    label_id = 0

    for person_name in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, person_name)
        if not os.path.isdir(person_path):
            continue

        label_map[label_id] = person_name

        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)
            img = cv2.imread(img_path)
            face, _ = detect_face(img)

            if face is not None:
                face = cv2.resize(face, (200, 200))
                faces.append(face)
                labels.append(label_id)

        label_id += 1

    return np.array(faces), np.array(labels), label_map
