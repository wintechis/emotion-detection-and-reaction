import threading

from flask import Flask, render_template, Response, make_response, send_file
import audio_recognizer
import json
import video_recognizer
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

app = Flask(__name__)


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


@app.route('/multimodal')
def multimodal():
    return render_template('multimodal.html')


@app.route('/live-data')
def live_data():
    # echo audio predictions as JSON
    data = audio_recognizer.analyze_audio()
    response = make_response(json.dumps(data))
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


# @app.route('/live-data_multi')
# def live_data_multi():
#     # Create a PHP array and echo it as JSON
#     f = open('multi_prediction.json')
#     data = json.load(f)
#     response = make_response(json.dumps(data))
#     response.content_type = 'application/json'
#     return response

@app.route('/live-data_multi')
def live_data_multi():
    # Create a PHP array and echo it as JSON
    f = open('video_prediction.json')
    data = json.load(f)
    data2 = audio_recognizer.analyze_audio()
    newdict = [{"name": "Video",
                "data": [{"name": "Angry", "value": round(data[0]*100, 2)}, {"name": "Fear", "value": round(data[1]*100, 2)},
                         {"name": "Happy", "value": round(data[2]*100, 2)}, {"name": "Sad", "value": round(data[3]*100, 2)}]},
               {"name": "Audio",
                "data": [{"name": "Angry", "value": round(data2[0]*100, 2)}, {"name": "Fear", "value": round(data2[1]*100, 2)},
                         {"name": "Happy", "value": round(data2[2]*100, 2)}, {"name": "Sad", "value": round(data2[3]*100, 2)}]}]

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




if __name__ == '__main__':
    app.run(debug=True)
    t = threading.Thread(target=your_func)
    t.setDaemon(True)
    t.start()
