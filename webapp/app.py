from flask import Flask, render_template
import threading

app = Flask(__name__)

def analyze_webcam():
    pass

def analyze_audio():
    pass


@app.route('/')
def index():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/recognizer')
def recognizer():
    return render_template('recognizer.html')

if __name__ == '__main__':
    app.run(debug=True)
