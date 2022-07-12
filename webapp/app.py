from flask import Flask, render_template, Response
import cv2
import numpy as np
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import image
from turbo_flask import Turbo
import threading
import audio_recognizer
import time
import concurrent.futures
import keras

#import video_recognizer

_pool = concurrent.futures.ThreadPoolExecutor()
app = Flask(__name__)
turbo = Turbo(app)

#load model
# model = model_from_json(open("models/fer.json", "r").read())
#
# #load weights
# model.load_weights('models/fer.h5')

model = keras.models.load_model('models/model_8_50epoch80_CK48dataset.h5')


try:
    face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
except Exception:
    st.write("Error loading cascade classifiers")
camera = cv2.VideoCapture(0)

def gen_frames():  # generate frame by frame from camera
    while True:
        # Capture frame by frame
        success, frame = camera.read()
        if not success:
            break
        else:
            gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.1, 4)

            for (x, y, w, h) in faces_detected:
                print('WORKING')
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), thickness=7)
                roi_gray = gray_img[y:y + w, x:x + h]  # cropping region of interest i.e. face area from  image
                roi_color = frame[y:y + h, x:x + w]
                roi_color = cv2.resize(roi_color, (80, 80))
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                facess = face_haar_cascade.detectMultiScale(roi_gray)
                if len(facess) == 0:
                    print("Face not detected")
                else:
                    for (ex, ey, ew, eh) in facess:
                        face_roi = roi_color[ey: ey + eh, ex:ex + ew]  ## cropping the face
                # img_pixels = image.img_to_array(roi_gray)
                # img_pixels = np.expand_dims(img_pixels, axis=0)
                # img_pixels /= 255
                        final_image = cv2.resize(face_roi, (80, 80))
                        final_image = np.expand_dims(final_image, axis=0)  ## need fourth dimension
                        final_image = final_image / 255.0

                print(final_image.shape)

                predictions = model.predict(final_image)

                # find max indexed array

                max_index = np.argmax(predictions[0])

                emotions = ['angry', 'happy', 'fear', 'sad']
                predicted_emotion = emotions[max_index]
                print(predicted_emotion)
                cv2.putText(frame, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            resized_img = cv2.resize(frame, (1000, 700))

            ret, buffer = cv2.imencode('.jpg', frame)

            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

@app.before_first_request
def before_first_request():
    threading.Thread(target=update_load).start()


@app.route('/')
def index():
    return render_template('index.html')


def update_load():
    with app.app_context():
        while True:
            time.sleep(1)
            turbo.push(turbo.update(render_template('loadavg.html'), 'load'))

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/audio')
def audio():
    return render_template('audio.html')

@app.route('/Video')
def Video():
    return render_template('Video.html')



def analyze_video():
    return "Video update " + str(time.strftime("%H:%M:%S"))


def analyze_audio():
    return audio_recognizer.analyze_audio()


@app.context_processor
def inject_load():
    # return emotions and post them to jinja
    items = {"video": "", "audio": ""}


    #p1 = _pool.submit(analyze_audio)
    #p2 = _pool.submit(analyze_video)
    #p2 = _pool.submit(video_recognizer.analyze_video())

    #items["audio"] = p1.result()[0]
    #items["video"] = p2.result()

    return items


if __name__ == '__main__':
    app.run(debug=True)
