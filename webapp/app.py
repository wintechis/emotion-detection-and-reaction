from flask import Flask, render_template, Response
import threading
from audio_recognizer import *

app = Flask(__name__)
video = cv2.VideoCapture(0)


def gen(video):
    while True:
        success, image = video.read()
        frame_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.equalizeHist(frame_gray)

        ret, jpeg = cv2.imencode('.jpg', image)

        frame = jpeg.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/')
def index():
    return render_template('home.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/video_feed')
def video_feed():
    return Response(gen(video), mimetype='multipart/x-mixed-replace; boundary=frame')   # gets called from recognizer.html


@app.route('/recognizer')
def recognizer():
    # return emotions and post them to html
    global video
    '''
    start process1
    start process2
    
    
    while True:
        for time <= 3 sekunden
            await get video predictions

        await get audio predicitons
    '''


    return render_template('recognizer.html')


if __name__ == '__main__':
    app.run(debug=True)
