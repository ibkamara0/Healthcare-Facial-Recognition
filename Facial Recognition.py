import numpy as np
import cv2
import pickle as p
import pandas as pd
import mysql.connector
from tkinter import *
from tkinter import font
from tkinter import scrolledtext

#Facial recognition
def facial_recognizer():
    face_cascade = cv2.CascadeClassifier('Cascades/data/haarcascade_frontalface_alt2.xml')
    recognizer = recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("trainer.yml")

    labels = {"person_name": 1}
    with open("labels.pickle", 'rb') as f:
        og_labels = p.load(f)
        labels = {v: k for k, v in og_labels.items()}

    # Capture using default web cam
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    # Web cam will stay on
    while True:
        # Reading frame by frame
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1, 1)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:

            roi_gray = gray[y:y + h, x:x + h]
            roi_color = frame[y:y + h, x:x + h]

            id_, conf = recognizer.predict(roi_gray)
            if 45 <= conf <= 85:
                font = cv2.FONT_HERSHEY_SIMPLEX
                name = labels[id_]
                color = (255, 255, 255)
                stroke = 2
                cv2.putText(frame, name, (x, y), font, 1, color, stroke, cv2.LINE_AA)

            img_item = "my-image.png"
            cv2.imwrite(img_item, roi_gray)

            color = (255, 0, 0)
            stroke = 2
            width = x + w
            height = y + h
            cv2.rectangle(frame, (x, y), (width, height), color, stroke)

        # Display frames
        cv2.imshow('Web Cam', frame)

        # Be able to kill the process
        if cv2.waitKey(20) & 0xFF == ord("q"):
            break
    # Release the capture after everything is done

    cap.release()
    cv2.destroyAllWindows()
    return id_

'''
input the face id returned from the face recognizer
and use it to query the database
'''

def query_db(face_id):
    db = mysql.connector.connect(
        host="localhost",
        user="root",
        passwd="asdf;lkj", #your database password
        database="Healthcare_Identification"
    )

    mycursor = db.cursor()

    # Query of specific information from the database
    sql_query = "SELECT fname, " \
                "lname, " \
                "provider, " \
                "record_num, " \
                "sex, dob, " \
                "complaint " \
                "FROM PATIENT_INFO WHERE face_id = {}".format(face_id)

    # Execute the stamement
    mycursor.execute(sql_query)
    information = mycursor.fetchall()[0]
    mycursor.close()
    

    return information

#The creation of the GUI with specific information
def create_gui(firstname, lastname, healthcare_provider, record_num, sex, dob, chief_complaint):
    name = '{}, {}'.format(lastname, firstname)

    root = Tk()

    root.title("Healthcare Information")

    heading_font = font.Font(family='Calibri', size=12, weight='bold')
    info_font = font.Font(family='Calibri', size=12)

    photo = PhotoImage(file="my-image.png")
    img = Label(root, image=photo).grid(sticky=E, rowspan=6, row=0, column=2, padx=10, pady=10)

    Label(root, text="Name:", font=heading_font).grid(sticky=E, row=0)
    Label(root, text=name, font=info_font).grid(sticky=W, row=0, column=1, padx=5, pady=5)

    Label(root, text="Healthcare Provider:", font=heading_font).grid(sticky=E, row=1)
    Label(root, text=healthcare_provider, font=info_font).grid(sticky=W, row=1, column=1, padx=5, pady=5)

    Label(root, text="Medical Record #:", font=heading_font).grid(sticky=E, row=2)
    Label(root, text=record_num, font=info_font).grid(sticky=W, row=2, column=1, padx=5, pady=5)

    Label(root, text="Sex:", font=heading_font).grid(sticky=E, row=3)
    Label(root, text=sex, font=info_font).grid(sticky=W, row=3, column=1, padx=5, pady=5)

    Label(root, text="DOB:", font=heading_font).grid(sticky=E, row=4)
    Label(root, text=dob, font=info_font).grid(sticky=W, row=4, column=1, padx=5, pady=5)

    Label(root, text="Chief Complaint:", font=heading_font).grid(sticky=E, row=5)
    Label(root, text=chief_complaint, font=info_font).grid(sticky=W, row=5, column=1, padx=5, pady=5)

    Label(root, text="Notes", font=heading_font).grid(sticky=S, row=7, column=0, columnspan=2, padx=5, pady=5)
    scrolledtext.ScrolledText(root, width=40, height=10).grid(row=8, column=0, columnspan=2, padx=5, pady=5)

    Button(root, text="SUBMIT", font=heading_font, bg='blue').grid(row=8, column=2, padx=15, pady=15,
                                                                   sticky=N + S + W + E)

    root.mainloop()


if __name__ == "__main__":

    # print(face_id)
    info = query_db(facial_recognizer())

    create_gui(info[0], info[1], info[2], info[3], info[4], info[5], info[6])
