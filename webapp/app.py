from flask import Flask, render_template, Response, make_response
#import cv2
#import numpy as np
#from tensorflow.keras.models import model_from_json
#from tensorflow.keras.preprocessing import image
from turbo_flask import Turbo
import threading
import audio_recognizer
from time import time
import concurrent.futures
import json
from random import random
import keras

import video_recognizer

app = Flask(__name__)


#load model
# model = model_from_json(open("models/fer.json", "r").read())
#
# #load weights
# model.load_weights('models/fer.h5')




@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(video_recognizer.gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/audio')
def audio():
    return render_template('audio.html', data='test')

@app.route('/Video')
def Video():
    return render_template('Video.html')



@app.route('/live-data')
def live_data():
    # Create a PHP array and echo it as JSON
    data = audio_recognizer.analyze_audio()
    response = make_response(json.dumps(data))
    response.content_type = 'application/json'
    return response




if __name__ == '__main__':
    app.run(debug=True)
