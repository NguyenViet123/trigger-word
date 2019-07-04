import matplotlib.pyplot as plt
import numpy as np
from pydub import AudioSegment
import glob
import td_utils
import time
import IPython

PREFIX = '../'
PREFIX_DATA = '../data/train/'
BACKGROUND_PATH = 'data/_background_noise_'
KEY_PATH = 'data/happy'
NEGATIVE_PATH = 'data/negative'
DATA_LEN = 10000
Ty = 1375
Tx = 5511
n_freq = 101


def prepare_data(idx, n_sample):
    X = []
    Y = []
    for i in range(n_sample):
        n_kw = np.random.randint(1, 5)
        n_ne = np.random.randint(0, 3)
        bg = random_choice_background()
        kw = random_choice_keywords(n_kw)
        ne = random_choice_negatives(n_ne)
        x, y = create_single_sample(bg, kw, ne)
        X.append(x)
        Y.append(y)

    np.save('X' + idx + '.npy', X)
    np.save('Y' + idx + '.npy', Y)


def preprocess_audio(file_name):
    padding = AudioSegment.silent(duration=DATA_LEN)
    segment = AudioSegment.from_wav(file_name)[:DATA_LEN]
    segment = padding.overlay(segment)
    segment = segment.set_frame_rate(44100)
    segment.export(file_name, format='wav')


def random_choice_background():
    paths = glob.glob(PREFIX + BACKGROUND_PATH + '/*')
    sel = np.random.randint(0, len(paths))
    bg = AudioSegment.from_wav(paths[sel])
    rand_start_time = np.random.randint(0, len(bg) - DATA_LEN - 1)
    return bg[rand_start_time: rand_start_time + DATA_LEN + 1]


def random_choice_keywords(n_keyword):
    paths = glob.glob(PREFIX + KEY_PATH + '/*')
    sel = np.random.randint(0, len(paths), size=n_keyword)
    kw = [AudioSegment.from_wav(paths[i]) for i in sel]
    return kw


def random_choice_negatives(n_negative):
    paths = glob.glob(PREFIX + NEGATIVE_PATH + '/*')
    sel = np.random.randint(0, len(paths), size=n_negative)
    ne = [AudioSegment.from_wav(paths[i]) for i in sel]
    return ne


def is_overlapping(segment_time, previous_segments):
    segment_start, segment_end = segment_time
    for previous_s, previous_e in previous_segments:
        if segment_start <= previous_e and segment_end >= previous_s:
            return True
    return False


def get_random_time_segment(segment_ms):
    segment_start = np.random.randint(low=0, high=DATA_LEN - segment_ms)
    segment_end = segment_start + segment_ms - 1

    return segment_start, segment_end


def insert_audio_clip(background, audio_clip, previous_segments):
    segment_ms = len(audio_clip)
    segment_time = get_random_time_segment(segment_ms)
    while is_overlapping(segment_time, previous_segments):
        segment_time = get_random_time_segment(segment_ms)

    previous_segments.append(segment_time)

    new_background = background.overlay(audio_clip, position=segment_time[0])

    return new_background, segment_time


def insert_ones(y, segment_end_ms):
    segment_end_y = int(segment_end_ms * Ty / 10000)
    for i in range(segment_end_y + 1, segment_end_y + 51):
        if i < Ty:
            y[i] = 1

    return y


def create_single_sample(background, keywords, negatives):
    background -= 20
    y = np.zeros(Ty)
    previous_segments = []
    # insert activate
    for k in keywords:
        background, segment_time = insert_audio_clip(background, k, previous_segments)
        y = insert_ones(y, segment_end_ms=segment_time[1])

    # insert negatives
    for n in negatives:
        background, segment_time = insert_audio_clip(background, n, previous_segments)

    background = td_utils.match_target_amplitude(background, -20.)
    f_name = PREFIX_DATA + str(time.time()) + '.wav'
    background = background.set_frame_rate(44100)
    background.export(f_name, format='wav')
    x = td_utils.graph_spectrogram(f_name)
    return x, y


start_time = time.time()
[prepare_data(str(i), 30) for i in range(100)]
# prepare_data(str(29), 50)
# prepare_data(str(30), 50)
# for i in range(200):
#     prepare_data(str(i), 30)
print('Done create data in ', time.time() - start_time)


# def load_data(path):
#     X = np.load('X.npy')
#     Y = np.load('Y.npy')
#     print(X.shape)
#     print(Y.shape)
#
#
# load_data('./')
