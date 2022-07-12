import numpy as np
import cv2
import streamlit as st
from tensorflow import keras
from keras.models import model_from_json
from keras.preprocessing.image import img_to_array
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, VideoProcessorBase, WebRtcMode

from flask import Flask, render_template, Response
import cv2
import numpy as np
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import image

face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

camera = cv2.VideoCapture(0)


def gen_frames():  # generate frame by frame from camera
    while True:
        # Capture frame by frame
        success, frame = camera.read()
        if not success:
            break
        else:
            gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)

            for (x, y, w, h) in faces_detected:
                print('WORKING')
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), thickness=7)
                roi_gray = gray_img[y:y + w, x:x + h]  # cropping region of interest i.e. face area from  image
                roi_gray = cv2.resize(roi_gray, (48, 48))
                img_pixels = image.img_to_array(roi_gray)
                img_pixels = np.expand_dims(img_pixels, axis=0)
                img_pixels /= 255

                print(img_pixels.shape)

                predictions = model.predict(img_pixels)

                # find max indexed array

                max_index = np.argmax(predictions[0])

                emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
                predicted_emotion = emotions[max_index]
                print(predicted_emotion)
                cv2.putText(frame, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            resized_img = cv2.resize(frame, (1000, 700))

            ret, buffer = cv2.imencode('.jpg', frame)

            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result


@app.route('/video_feed')
def video_feed():
    # Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# from flask import Flask, render_template, Response
# import cv2
# import numpy as np
# from tensorflow.keras.models import model_from_json
# from tensorflow.keras.preprocessing import image
# from turbo_flask import Turbo
# import threading
# import audio_recognizer
# import time
# import concurrent.futures
#
# model = keras.models.load_model('models/model_8_50epoch80_CK48dataset.h5')
# face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# camera = cv2.VideoCapture(0)
#
# def gen_frames():
#     while True:
#         # Capture frame by frame
#         success, frame = camera.read()
#         if not success:
#             break
#         else:
#             gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#             faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)
#             for (x, y, w, h) in faces_detected:
#                 cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), thickness=7)
#                 roi_gray = gray_img[y:y + w, x:x + h]  # cropping region of interest i.e. face area from  image
#                 roi_gray = cv2.resize(roi_gray, (48, 48))
#                 img_pixels = image.img_to_array(roi_gray)
#                 img_pixels = np.expand_dims(img_pixels, axis=0)
#                 img_pixels /= 255
#
#                 predictions = model.predict(img_pixels)
#
#                 max_index = np.argmax(predictions[0])  # find max indexed array
#
#                 emotions = ['angry', 'fear', 'happy', 'sad']
#                 predicted_emotion = emotions[max_index]
#
#                 cv2.putText(frame, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#
#             resized_img = cv2.resize(frame, (1000, 700))
#
#             ret, buffer = cv2.imencode('.jpg', frame)
#
#             frame = buffer.tobytes()
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result
#
# @app.route('/video_feed')
# def video_feed():
#     return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
#
#
#
# if __name__ == '__main__':
#     app.run(debug=True)
#
