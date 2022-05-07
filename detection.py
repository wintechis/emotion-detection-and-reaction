import cv2
from deepface import DeepFace
import numpy as np

face_cascade_name = cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml'  # haarcascade xml file
face_cascade = cv2.CascadeClassifier()  # processing it for our project
if not face_cascade.load(cv2.samples.findFile(face_cascade_name)):  # fallback event
    print("Error loading xml file")

video = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # requesting webcam

while video.isOpened():  # check videofeed
    ret, frame = video.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # convert to greyscale
    face = face_cascade.detectMultiScale(gray)

    for x, y, w, h in face:
        img = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)  # rectangle to show up and detect the face
        try:
            analyze = DeepFace.analyze(frame, actions=['emotion'])  # analyze class from deepface and ‘frame’ as input
            print(analyze['dominant_emotion'], "  ", analyze['emotion'])
        except:
            pass
            #print("no face")

        # display output
        cv2.imshow('video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # specifying exit key
            break
video.release()
cv2.destroyAllWindows()
