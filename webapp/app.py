from flask import Flask, render_template, Response, make_response
import audio_recognizer
import json
import video_recognizer

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

@app.route('/dynamics')
def dynamics():
    return render_template('dynamics.html')


@app.route('/2015.json')
def json1():
    f = open('2015.json')
    jsonfile = json.load(f)
    res = make_response(json.dumps(jsonfile))
    res.content_type = 'application/json'
    return res

@app.route('/2016.json')
def json2():
    return Response('2016.json')



@app.route('/live-data')
def live_data():
    # Create a PHP array and echo it as JSON
    data = audio_recognizer.analyze_audio()

    response = make_response(json.dumps(data))
    response.content_type = 'application/json'
    return response



if __name__ == '__main__':
    app.run(debug=True)
