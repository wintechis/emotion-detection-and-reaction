from SpeechRecognizer_CNN_preprocess import *
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, LSTM
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sns

max_len = 11
buckets = 20

# Save data to array file first
save_data_to_array(max_len=max_len, n_mfcc=buckets)
emotion_labels = {'01': 'happy', '02': 'sad', '03': 'angry', '04': 'fearful'}
labels = ['happy', 'sad', 'angry', 'fearful']
#labels = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
#focused_labels = ['happy', 'sad', 'angry', 'fearful']

# # Loading train set and test set
X_train, X_test, y_train, y_test = get_train_test()

# # Feature dimension
channels = 1
epochs = 100
num_classes = 4

X_train = X_train.reshape(X_train.shape[0], buckets, max_len, channels)
X_test = X_test.reshape(X_test.shape[0], buckets, max_len, channels)

y_train_hot = to_categorical(y_train)
y_test_hot = to_categorical(y_test)

#CNN
model = Sequential()
model.add(Conv2D(32,
                 (5, 5),
                 input_shape=(buckets, max_len, channels),
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

checkpoint = ModelCheckpoint('SER_model.h5', monitor='val_loss', mode='min', save_best_only=True, verbose=1)
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
model_history = model.fit(X_train, y_train_hot, epochs=epochs, validation_data=(X_test, y_test_hot), callbacks=checkpoint)
model.summary()
model.save("D:\\Schmiede\\Coding\\IIS_Seminar\\emotion-detection-and-reaction\\SER\\best_SER_model.h5")



# PRINT MODEL HISTORY OF ACCURACY AND LOSS OVER EPOCHS
plt.plot(model_history.history['accuracy'])
plt.plot(model_history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('Model_Accuracy.png')
plt.show()
# summarize history for loss
plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('Model_Loss.png')
plt.show()



# PREDICTION LABELS
predictions = model.predict(X_test, batch_size=32)
predictions = predictions.argmax(axis=1)
predictions = predictions.astype(int).flatten()
predictions2 = []
for pred in predictions:
    predictions2.append(emotion_labels["0"+str(pred+1)])
predictions = pd.DataFrame({'Predicted Values': predictions2})


# ACTUAL LABELS
actual2 = []
for i in y_test_hot:
    actual = i.argmax()
    actual2.append(emotion_labels["0"+str(actual+1)])

actual = pd.DataFrame({'Actual Values': actual2})

# COMBINE PREDICTION AND ACTUAL LABELS
finaldf = actual.join(predictions)
finaldf[140:150]



# CREATE CONFUSION MATRIX OF ACTUAL VS. PREDICTION
cm = confusion_matrix(actual, predictions)
plt.figure(figsize=(12, 10))
cm = pd.DataFrame(cm, index=[i for i in labels], columns=[i for i in labels])
ax = sns.heatmap(cm, linecolor='white', cmap='Blues', linewidth=1, annot=True, fmt='')
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
plt.title('Confusion Matrix', size=20)
plt.xlabel('Predicted Labels', size=14)
plt.ylabel('Actual Labels', size=14)
plt.savefig('Model_Confusion_Matrix.png')
plt.show()

