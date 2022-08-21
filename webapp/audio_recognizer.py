#   Python file containing the processing logic for SER analysis tasks
#   returns an array of class probabilities

from tensorflow import keras
import numpy as np
import librosa
import librosa.display
import pyaudio
import wave
import matplotlib.pyplot as plt
from joblib import load

scaler = load('models/std_scaler.bin')  # load pretrained SciKit StandardScaler
model_audio = keras.models.load_model('models/SER_model_without_CREMA.h5')


# ToDo include following small functions in extract_audio

def noise(data):
    noise_amp = 0.035*np.random.uniform()*np.amax(data)
    data = data + noise_amp*np.random.normal(size=data.shape[0])
    return data


def stretch(data):
    return librosa.effects.time_stretch(data, rate=0.8)


def shift(data):
    shift_range = int(np.random.uniform(low=-5, high=5)*1000)
    return np.roll(data, shift_range)


def pitch(data, sampling_rate):
    return librosa.effects.pitch_shift(data, sampling_rate, n_steps=0.7)


# main functions
def extract_audio_features(data, sample_rate):
    # ZCR
    result = np.array([])
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result = np.hstack((result, zcr))  # stacking horizontally

    # Chroma_stft
    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    result = np.hstack((result, chroma_stft))  # stacking horizontally

    # MFCC
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mfcc))  # stacking horizontally

    # Root Mean Square Value
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    result = np.hstack((result, rms))  # stacking horizontally

    # MelSpectogram
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mel))  # stacking horizontally
    spec = np.abs(librosa.stft(data, hop_length=512))
    spec = librosa.amplitude_to_db(spec, ref=np.max)
    librosa.display.specshow(spec, sr=sample_rate, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.clim(-80, 0)
    plt.title('Spectrogram')
    plt.tight_layout()
    plt.savefig('diagrams\\MelSpec.png')
    plt.clf()
    #plt.show()
    plt.figure(figsize=(8, 4))
    librosa.display.waveshow(data, sr=sample_rate)
    plt.title('Waveplot')
    plt.savefig('diagrams\\Waveplot.png')
    plt.clf()
    return result


def get_audio_features(path):
    data, sample_rate = librosa.load(path)

    # without augmentation
    res1 = extract_audio_features(data, sample_rate)
    result = np.array(res1)

    # data with noise
    noise_data = noise(data)
    res2 = extract_audio_features(noise_data, sample_rate)
    result = np.vstack((result, res2))  # stacking vertically

    # data with stretching and pitching
    new_data = stretch(data)
    data_stretch_pitch = pitch(new_data, sample_rate)
    res3 = extract_audio_features(data_stretch_pitch, sample_rate)
    result = np.vstack((result, res3))  # stacking vertically

    return result


# driver function: returns prediction
def analyze_audio():

#while True:
    samp_rate = 44100  # 44.1kHz sampling rate
    chunk = 4096  # 2^12 samples for buffer
    record_secs = 3  # seconds to record
    dev_index = 1  # device index found by p.get_device_info_by_index(ii)
    wav_output_filename = 'temp_audio.wav'  # name of .wav file

    # check input devices
    audio = pyaudio.PyAudio()
    #info = audio.get_host_api_info_by_index(0)
    #numdevices = info.get('deviceCount')
    #for i in range(0, numdevices):
    #    if (audio.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
    #        print("Input Device id ", i, " - ", audio.get_device_info_by_host_api_device_index(0, i).get('name'))

    ###################
    # GET AUDIO STREAM
    stream = audio.open(format=pyaudio.paInt16, channels=1, rate=samp_rate, input=True, frames_per_buffer=chunk,
                        input_device_index=dev_index)
    frames = []
    for ii in range(0, int((samp_rate / chunk) * record_secs)):
        data = stream.read(chunk)
        frames.append(data)
    stream.stop_stream()
    stream.close()
    audio.terminate()

    wavefile = wave.open(wav_output_filename, 'wb')
    wavefile.setnchannels(1)
    wavefile.setsampwidth(audio.get_sample_size(pyaudio.paInt16))  # 16-bit resolution
    wavefile.setframerate(samp_rate)
    wavefile.writeframes(b''.join(frames))
    wavefile.close()

    x_audio = get_audio_features(wav_output_filename)
    x_audio = scaler.transform(x_audio)
    x_audio = np.expand_dims(x_audio, axis=2)
    pred = model_audio.predict(x_audio)
    j=[]
    for i in range(3):
        j.append(np.argmax(pred[i]))
    counts = np.bincount(j)     #return most frequent value from stacked array (because we predict on different versions)
    return np.argmax(counts)
