from flask import Flask, render_template, Response
from turbo_flask import Turbo
import threading
import audio_recognizer
import time
import concurrent.futures
#import video_recognizer

_pool = concurrent.futures.ThreadPoolExecutor()
app = Flask(__name__)
turbo = Turbo(app)


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
            turbo.push(turbo.update(render_template('audio.html'), 'load'))


@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/audio')
def audio():
    return render_template('audio.html')

@app.route('/Video')
def Video():
    return render_template('Video.html')


@app.route('/video_feed')
def video_feed():
    return


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

    items["audio"] = audio_recognizer.analyze_audio()[0]
    #items["video"] = p2.result()

    return items


if __name__ == '__main__':
    app.run(debug=True)
