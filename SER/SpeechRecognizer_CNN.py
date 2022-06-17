import librosa as lb
import soundfile as sf
import numpy as np
import os, glob, pickle
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from SpeechRecognizer_CNN_preprocess import *
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, LSTM
from keras.utils.np_utils import to_categorical
import wandb
from wandb.keras import WandbCallback


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
        gender = int(file_path.split("-")[6].replace(".wav", ""))
        if (gender % 2) == 0:
            gender = "female"
            #continue
        else:
            gender = "male"
            continue
        if emotion not in focused_emotion_labels:
            continue
        feature = audio_features(file, mfcc=True, chroma=True, mel=True)
        x.append(feature)
        y.append(emotion)
    final_dataset = train_test_split(np.array(x), y, test_size=0.2, random_state=9)
    return final_dataset


if __name__ == "__main__":

    max_len = 11
    buckets = 20

    # Save data to array file first
    save_data_to_array(max_len=max_len, n_mfcc=buckets)
    emotion_labels = {'03': 'happy', '04': 'sad', '05': 'angry', '06': 'fearful'}
    labels = ['happy', 'sad', 'angry', 'fearful']
    focused_emotion_labels = ['happy', 'sad', 'angry', 'fearful']

    # # Loading train set and test set
    X_train, X_test, y_train, y_test = get_train_test()

    # # Feature dimension
    channels = 1
    epochs = 50
    batch_size = 100
    num_classes = 4

    X_train = X_train.reshape(X_train.shape[0], buckets, max_len, channels)
    X_test = X_test.reshape(X_test.shape[0], buckets, max_len, channels)
    plt.imshow(X_train[12, :, :, 0])
    plt.show()
    print(y_train[12])

    y_train_hot = to_categorical(y_train)
    y_test_hot = to_categorical(y_test)

    #X_train = X_train.reshape(X_train.shape[0], buckets, max_len)
    #X_test = X_test.reshape(X_test.shape[0], buckets, max_len)

    #model = Sequential()
    #model.add(Flatten(input_shape=(buckets, max_len)))
    #model.add(Dense(num_classes, activation='softmax'))
    #model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])

    #model.fit(X_train, y_train_hot, epochs=epochs, validation_data=(X_test, y_test_hot),
    #          callbacks=[WandbCallback(data_type="image", labels=labels)])

    # build model
    #model = Sequential()
    #model.add(LSTM(16, input_shape=(buckets, max_len, channels), activation="sigmoid"))
    #model.add(Dense(1, activation='sigmoid'))
    #model.add(Dense(num_classes, activation='softmax'))

    #model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])

    #model.fit(X_train, y_train_hot, epochs=epochs, validation_data=(X_test, y_test_hot),
    #          callbacks=[WandbCallback(data_type="image", labels=labels)])


    #CNN
    model = Sequential()
    model.add(Conv2D(32,
                     (4, 4),
                     input_shape=(buckets, max_len, channels),
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
    model.fit(X_train, y_train_hot, epochs=epochs, validation_data=(X_test, y_test_hot),
              callbacks=[WandbCallback(data_type="image", labels=labels)])
