from flask import Flask, render_template, Response
from turbo_flask import Turbo
import threading
import math
import audio_recognizer
import video_recognizer
import time
import concurrent.futures

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
            turbo.push(turbo.update(render_template('loadavg.html'), 'load'))


@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/recognizer')
def recognizer():
    return render_template('recognizer.html')


@app.route('/video_feed')
def video_feed():
    return


def analyze_video():
    return "Video update " + str(time.strftime("%H:%M:%S"))


def analyze_audio():
    return "Audio update " + str(time.strftime("%H:%M:%S"))


@app.context_processor
def inject_load():
    # return emotions and post them to jinja
    items = {"video": "", "audio": ""}
    # webcam starten

    p1 = _pool.submit(analyze_audio)
    p2 = _pool.submit(analyze_video)

    items["audio"] = p1.result()
    items["video"] = p2.result()

    return items


if __name__ == '__main__':
    app.run(debug=True)
