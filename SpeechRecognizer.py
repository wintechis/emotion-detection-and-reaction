import librosa as lb
import soundfile as sf
import numpy as np
import os, glob, pickle
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

emotion_labels = {
  '01':'neutral',
  '02':'calm',
  '03':'happy',
  '04':'sad',
  '05':'angry',
  '06':'fearful',
  '07':'disgust',
  '08':'surprised'
}
focused_emotion_labels = ['happy', 'sad', 'angry', 'fear']


def audio_features(file_title, mfcc, chroma, mel):
    with sf.SoundFile(file_title) as audio_recording:
        audio = audio_recording.read(dtype="float32")
        sample_rate = audio_recording.samplerate

        if chroma:
            stft = np.abs(lb.stft(y=audio))
        result = np.array([])
        if mfcc:
            mfccs = np.mean(lb.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))
        if chroma:
            chroma = np.mean(lb.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, chroma))
        if mel:
            mel = np.mean(lb.feature.melspectrogram(y=audio, sr=sample_rate).T, axis=0)
            result = np.hstack((result, mel))
            #fig, ax = plt.subplots()
            #S_dB = lb.power_to_db(mel, ref=np.max)
            #img = lb.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sample_rate, fmax=8000, ax=ax)
            #fig.colorbar(img, ax=ax, format='%+2.0f dB')
            #ax.set(title='Mel-frequency spectrogram')
            #plt.show()
        return result


def loading_audio_data():
    x = []
    y = []

    for file in glob.glob("data//Actor_*//*.wav"):
        file_path = os.path.basename(file)
        emotion = emotion_labels[file_path.split("-")[2]]
        #if emotion not in focused_emotion_labels:
        #    continue
        feature = audio_features(file, mfcc=True, chroma=True, mel=True)
        x.append(feature)
        y.append(emotion)
    final_dataset = train_test_split(np.array(x), y, test_size=0.2, random_state=9)
    return final_dataset


print("Loading data...")
X_train, X_test, y_train, y_test = loading_audio_data()
print("Create Model...")
model = MLPClassifier(hidden_layer_sizes=(200,), learning_rate='adaptive', max_iter=400)
model.fit(X_train, y_train)
print("Predict...")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
print("Accuracy of the Recognizer is: {:.1f}%".format(accuracy*100))
