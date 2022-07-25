#   Python file containing the processing logic for FER analysis tasks
#   outputs a videostream with informational layers

import cv2
import numpy as np
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import image
import keras

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

