import librosa
import os, glob
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import numpy as np
from tqdm import tqdm

DATA_PATH = "./data_old/"


# Input: Folder Path
# Output: Tuple (Label, Indices of the labels, one-hot encoded labels)
def get_labels(path=DATA_PATH):
    labels = ['happy', 'sad', 'angry', 'fearful']
    label_indices = np.arange(0, len(labels))
    return labels, label_indices, to_categorical(label_indices)


# convert file to wav2mfcc
# Mel-frequency cepstral coefficients
def wav2mfcc(file_path, n_mfcc=20, max_len=11):
    wave, sr = librosa.load(path=file_path, mono=True, sr=None)
    wave = np.asfortranarray(wave[::3])
    mfcc = librosa.feature.mfcc(y=wave, sr=16000, n_mfcc=n_mfcc)

    # If maximum length exceeds mfcc lengths then pad the remaining ones
    if (max_len > mfcc.shape[1]):
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')

    # Else cutoff the remaining parts
    else:
        mfcc = mfcc[:, :max_len]

    return mfcc


def save_data_to_array(path=DATA_PATH, max_len=11, n_mfcc=20):
    labels, _, _ = get_labels(path)
    emotion_labels = {'01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad', '05': 'angry', '06': 'fearful',
                      '07': 'disgust', '08': 'surprised'}

    for label in emotion_labels:
        # Init mfcc vectors
        mfcc_vectors = []
        for file in glob.glob("data//Actor_*//*.wav"):
            file_path = os.path.basename(file)
            if file_path.split("-")[2] == label:
                file_path = os.path.basename(file)
                emotion = emotion_labels[file_path.split("-")[2]]
                mfcc = wav2mfcc(file, max_len=max_len, n_mfcc=n_mfcc)
                mfcc_vectors.append(mfcc)
        np.save(emotion + '.npy', mfcc_vectors)


def get_train_test(split_ratio=0.6, random_state=42):
    # Get available labels
    labels, indices, _ = get_labels(DATA_PATH)

    # Getting first arrays
    X = np.load(labels[0] + '.npy')
    y = np.zeros(X.shape[0])

    # Append all of the dataset into one single array, same goes for y
    for i, label in enumerate(labels[1:]):
        x = np.load(label + '.npy')
        X = np.vstack((X, x))
        y = np.append(y, np.full(x.shape[0], fill_value=(i + 1)))

    assert X.shape[0] == len(y)

    return train_test_split(X, y, test_size=(1 - split_ratio), random_state=random_state, shuffle=True)


def prepare_dataset(path=DATA_PATH):
    labels, _, _ = get_labels(path)
    data = {}
    for label in labels:
        data[label] = {}
        data[label]['path'] = [path + label + '/' + wavfile for wavfile in os.listdir(path + '/' + label)]

        vectors = []

        for wavfile in data[label]['path']:
            wave, sr = librosa.load(wavfile, mono=True, sr=None)
            # Downsampling
            wave = wave[::3]
            mfcc = librosa.feature.mfcc(wave, sr=16000)
            vectors.append(mfcc)

        data[label]['mfcc'] = vectors

    return data


def load_dataset(path=DATA_PATH):
    data = prepare_dataset(path)

    dataset = []

    for key in data:
        for mfcc in data[key]['mfcc']:
            dataset.append((key, mfcc))

    return dataset[:100]
