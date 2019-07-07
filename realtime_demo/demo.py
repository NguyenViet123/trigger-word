import numpy as np
import matplotlib.mlab as mlab
from scipy.io.wavfile import write
from queue import Queue
import sys
import time
from keras.models import load_model
import keras.backend as K
import pyaudio
import keras_metrics as km
from preprocess.td_utils import get_spectrogram

PRETRAIN_PATH = '../pretrain/best_f1.h5'

import pyaudio


def loss(y_true, y_pred):
    print(y_true.shape, y_pred.shape)
    weight = np.ones_like(y_true)
    weight = weight * 10
    return K.mean(K.mean(- y_true * K.log(y_pred + 0.000001) - (1 - y_true + 0.000001) * K.log(1 - y_pred), axis=1))


model = load_model(PRETRAIN_PATH, custom_objects={'loss': loss,
                                                  'binary_precision': km.binary_precision(),
                                                  'binary_recall': km.binary_recall(),
                                                  'binary_f1_score': km.binary_f1_score()})


# model.summary()


def predict(x):
    x = x.swapaxes(0, 1)
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x)
    return preds.reshape(-1)


def check_keyword(preds, chunk_duration, feed_duration, threshold=0.5):
    preds = preds > threshold
    chunk_sample = int(len(preds) * chunk_duration / feed_duration)
    chunk_preds = preds[-chunk_sample:]
    level = chunk_preds[0]
    for pred in chunk_preds:
        if pred > level:
            return True
        else:
            level = preds
    return False


chunk_duration = 0.5
fs = 44100
chunk_samples = int(fs * chunk_duration)


def audio_input(callback):
    stream = pyaudio.PyAudio().open(
        format=pyaudio.paInt16,
        channels=1,
        rate=fs,
        input=True,
        # output=True,
        frames_per_buffer=chunk_samples,
        # input_device_index=13,
        stream_callback=callback
    )

    return stream


q = Queue()
silence_threshold = 100
data = np.zeros(fs * 10)
feed_sample = fs * 10


def callback(data_in, frame_count, time_info, status):
    global data
    data0 = np.frombuffer(data_in, dtype='int16')
    if np.abs(data).mean() < silence_threshold:
        print('-', end='')
        return data_in, pyaudio.paContinue
    else:
        print('.', end='')
    data = np.append(data, data0)
    if len(data) > feed_sample:
        data = data[-feed_sample:]
        q.put(data)

    return data_in, pyaudio.paContinue


stream = audio_input(callback)
stream.start_stream()
t = time.time()

while time.time() - t < 10:
    data = q.get()
    spectrum = get_spectrogram(data)
    preds = predict(spectrum)
    if check_keyword(preds, chunk_duration, feed_duration=10):
        print('1', end='')

stream.stop_stream()
stream.close()
