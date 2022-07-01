from flask import Flask, render_template, Response
from turbo_flask import Turbo
import multiprocessing
import threading
import math
import audio_recognizer
import video_recognizer
import time

queue = multiprocessing.Queue()
app = Flask(__name__)
turbo=Turbo(app)

@app.before_first_request
def before_first_request():
    threading.Thread(target=update_load).start()


@app.route('/')
def index():
    return render_template('index.html')


def update_load():
    with app.app_context():
        while True:
            time.sleep(5)
            turbo.push(turbo.update(render_template('loadavg.html'), 'load'))


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/video_feed')
def video_feed():
    return


def analyze_video():
    queue.put({"video": str(time.strftime("%H:%M:%S"))})


def analyze_audio():
    queue.put({"audio": str(time.strftime("%H:%M:%S"))})


@app.context_processor
def inject_load():
    # return emotions and post them to jinja

    proc1 = multiprocessing.Process(target=analyze_audio())
    proc2 = multiprocessing.Process(target=analyze_video())
    proc1.start()
    proc2.start()

    return {'Test': queue.get()}


if __name__ == '__main__':
    app.run(debug=True)
