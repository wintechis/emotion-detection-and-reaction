from SpeechRecognizer_CNN_preprocess import *
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, LSTM
from keras.utils.np_utils import to_categorical

max_len = 11
buckets = 20

# Save data to array file first
save_data_to_array(max_len=max_len, n_mfcc=buckets)
emotion_labels = {'01': 'neutral',
        '02': 'calm',
        '03': 'happy',
        '04': 'sad',
        '05': 'angry',
        '06': 'fearful',
        '07': 'disgust',
        '08': 'surprised'}
#labels = ['happy', 'sad', 'angry', 'fearful']
labels = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
focused_emotion_labels = ['happy', 'sad', 'angry', 'fearful']

# # Loading train set and test set
X_train, X_test, y_train, y_test = get_train_test()

# # Feature dimension
channels = 1
epochs = 50
num_classes = 8

X_train = X_train.reshape(X_train.shape[0], buckets, max_len, channels)
X_test = X_test.reshape(X_test.shape[0], buckets, max_len, channels)

y_train_hot = to_categorical(y_train)
y_test_hot = to_categorical(y_test)

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
model.fit(X_train, y_train_hot, epochs=epochs, validation_data=(X_test, y_test_hot))
model.summary()
model.save("D:\Schmiede\Coding\IIS_Seminar\emotion-detection-and-reaction\SER\SER_model.h5")
