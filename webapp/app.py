#
#   Main driver code for the Flask based web app on a localhost
#   created by Ronja Rehm and Jan Kühlborn
#
###################################################################################

from flask import Flask, render_template, Response, make_response, send_file, send_from_directory
import audio_recognizer
import audio_recognizer8
import json
import video_recognizer
import os
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

app = Flask(__name__, static_folder='static')


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


@app.route('/audio8')
def audio8():
    return render_template('audio_8emotions.html', data='test')


@app.route('/Video')
def Video():
    return render_template('Video.html')


@app.route('/multimodal')
def multimodal():
    with open('highestValue.txt', 'r') as f:
        return render_template('multimodal.html', text=f.read())


@app.route('/live-data')
def live_data():
    # echo audio predictions as JSON
    data = audio_recognizer.analyze_audio()
    response = make_response(json.dumps(data.tolist()))
    response.content_type = 'application/json'
    return response

@app.route('/live-data8')
def live_data8():
    # echo audio predictions as JSON
    data = audio_recognizer8.analyze_audio()
    response = make_response(json.dumps(data.tolist()))
    response.content_type = 'application/json'
    return response


@app.route('/live-data_video')
def live_data_video():
    # Create a PHP array and echo it as JSON
    f = open('video_prediction.json')
    data = json.load(f)
    response = make_response(json.dumps(data))
    response.content_type = 'application/json'
    return response


@app.route('/live-data_multi')
def live_data_multi():
    # Create a PHP array and echo it as JSON
    f = open('video_prediction.json')
    data = json.load(f)
    data2 = audio_recognizer.analyze_audio()
    newdict = [{"name": "Video", "data": [{"name": "Angry", "value": round(data[0])}, {"name": "Fear", "value": round(data[1])}, {"name": "Happy", "value": round(data[2])}, {"name": "Sad", "value": round(data[3])}]}, {"name": "Audio", "data": [{"name": "Angry", "value": round(data2[0])}, {"name": "Fear", "value": round(data2[1])}, {"name": "Happy", "value": round(data2[2])}, {"name": "Sad", "value": round(data2[3])}]}]

    response = make_response(json.dumps(newdict))
    #response.content_type = 'application/json'
    newdict = [{"name": "Video",
                "data": [{"name": "Angry", "value": round(data[0] * 100, 2)},
                         {"name": "Fear", "value": round(data[1] * 100, 2)},
                         {"name": "Happy", "value": round(data[2] * 100, 2)},
                         {"name": "Sad", "value": round(data[3] * 100, 2)}]},
               {"name": "Audio",
                "data": [{"name": "Angry", "value": round(data2[0] * 100, 2)},
                         {"name": "Fear", "value": round(data2[1] * 100, 2)},
                         {"name": "Happy", "value": round(data2[2] * 100, 2)},
                         {"name": "Sad", "value": round(data2[3] * 100, 2)}]}]

    print(newdict)

    response = make_response(json.dumps(newdict))
    response.content_type = 'application/json'
    return response


@app.route('/spectrogram')
def spectrogram():
    return send_file('diagrams\\MelSpec.png', mimetype='image/png')


@app.route('/waveplot')
def waveplot():
    return send_file('diagrams\\Waveplot.png', mimetype='image/png')

@app.route('/emotion')
def emotion():
    data = audio_recognizer.analyze_audio()
    print(np.argmax(data))
    if np.argmax(data) == 0:
        return send_file('static\\images\\angry.jpg', mimetype='image/jpg')
    elif np.argmax(data) == 1:
        return send_file('static\\images\\fear.jpg', mimetype='image/jpg')
    elif np.argmax(data) == 2:
        return send_file('static\\images\\happy.jpg', mimetype='image/jpg')
    elif np.argmax(data) == 3:
        return send_file('static\\images\\sad.png', mimetype='image/png')
    else:
        return send_file('static\\images\\blank.png', mimetype='image/jpg')


@app.route("/happy")
def h():
    return send_file('static/images/happy.jpg')


@app.route("/fear")
def f():
    return send_file('static/images/fear.jpg')


@app.route("/angry")
def a():
    return send_file('static/images/angry.jpg')


@app.route("/sad")
def s():
    return send_file('static/images/sad.jpg')


@app.route("/highest")
def highest():
    with open('highestValue.txt', 'r') as f:
        return render_template('content.html', text=f.read())


if __name__ == '__main__':
    app.run(debug=False)
