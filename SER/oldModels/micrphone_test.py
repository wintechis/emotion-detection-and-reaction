import pyaudio
import wave

FORMAT = pyaudio.paInt16        # 16-bit resolution
chans = 1                       # 1 channel
samp_rate = 44100               # 44.1kHz sampling rate
chunk = 4096                    # 2^12 samples for buffer
record_secs = 3                 # seconds to record
dev_index = 2                   # device index found by p.get_device_info_by_index(ii)
wav_output_filename = 'test1.wav' # name of .wav file


# check input devices
audio = pyaudio.PyAudio()
info = audio.get_host_api_info_by_index(0)
numdevices = info.get('deviceCount')
for i in range(0, numdevices):
    if (audio.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
        print("Input Device id ", i, " - ", audio.get_device_info_by_host_api_device_index(0, i).get('name'))

# GET AUDIO STREAM
stream = audio.open(format=FORMAT, channels=chans, rate=samp_rate, input=True, frames_per_buffer=chunk, input_device_index=dev_index)
frames = []
for ii in range(0, int((samp_rate/chunk)*record_secs)):
    data = stream.read(chunk)
    frames.append(data)
stream.stop_stream()
stream.close()
audio.terminate()

#wavefile = wave.open(wav_output_filename, 'wb')
#wavefile.setnchannels(chans)
#wavefile.setsampwidth(audio.get_sample_size(FORMAT))
#wavefile.setframerate(samp_rate)
#wavefile.writeframes(b''.join(frames))
#wavefile.close()
print("file closed")
