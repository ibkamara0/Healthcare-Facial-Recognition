import os
import cv2

import numpy as np
from PIL import Image
import pickle

# Get the base directory and then get to the images folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "images")

face_cascade = cv2.CascadeClassifier('Cascades/data/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()


current_id = 0
label_ids = {}
y_labels = []
x_train = []

# see the images in the folder
for root, dirs, files in os.walk(image_dir):
    #Open each file
    for file in files:
        #looking for images in each file
        if file.endswith("png") or file.endswith("jpg"):
            path = os.path.join(root, file)
            #translating name of the file to a label
            label = os.path.basename(os.path.dirname(path)).replace(" ", "-").lower()
            #if we don't already have the label we will add it
            if not label in label_ids:
                label_ids[label] = current_id
                current_id += 1

            id_ = label_ids[label]
            

            pil_image = Image.open(path).convert("L") # grayscale
            size =(550,550)
            final_image = pil_image.resize(size, Image.ANTIALIAS)
            image_array = np.array(final_image, "uint8")
            
            #detecting the face
            faces = face_cascade.detectMultiScale(image_array)

            #add the faces with the label to separate lists 
            for (x, y, w, h) in faces:
                roi = image_array[y:y+h, x:x+w]
                x_train.append(roi)
                y_labels.append(id_)




with open("labels.pickle", 'wb') as f:
    pickle.dump(label_ids, f)

#Export the training model to be used in Facial Recognition
recognizer.train(x_train, np.array(y_labels))
recognizer.save("trainer.yml")



