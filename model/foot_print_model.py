from scipy import signal
from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np
from preprocess import Augment
import glob
import librosa
import librosa.display
from keras.layers import Input, Conv2D, GRU, Dense, Dropout, BatchNormalization, Bidirectional, Activation, Reshape
from keras.models import Model, load_model
import keras.backend as K


class CRNN:
    def __init__(self, ep, bs):
        self.ep = ep
        self.bs = bs
        self.kw_path = glob.glob('../data_model2/happy/*')
        self.ne_path = glob.glob('../data_model2/negative/*')
        self.full_path = np.array(self.kw_path + self.ne_path)
        self.n_sample = len(self.full_path)
        print('sample: ', self.n_sample)
        self.labels = np.array([1] * len(self.kw_path) + [0] * len(self.ne_path))
        self.idx_bs = 0
        self.aug = Augment.AudioAugmentation()

        perm = list(range(self.n_sample))
        np.random.shuffle(perm)
        self.full_path = self.full_path[perm]
        self.labels = self.labels[perm]

    def model(self, shape=(101, 40, 1)):
        input = Input(shape=shape)
        x = Conv2D(filters=32, kernel_size=(20, 5), strides=(8, 2))(input)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        print(x.shape)
        x = Reshape(target_shape=(11, 18 * 32))(x)
        x = Bidirectional(GRU(32, activation='relu', return_sequences=True))(x)
        print(x.shape)
        x = Bidirectional(GRU(32, activation='relu'))(x)
        x = Dense(units=64)(x)
        x = Dense(units=2, activation='softmax')(x)

        return Model(input, x)

    def training(self):
        model = self.model()
        model.compile('adam', loss='sparse_categorical_crossentropy', metrics=['acc'])
        for e in range(self.ep):
            n_batch = self.n_sample // self.bs
            loss_ = 0.
            for b in range(n_batch):
                X, Y = self.get_data_for_batch()
                his = model.fit(X, Y, batch_size=self.bs, epochs=1)
                loss_ += his.history['loss'][0]

            print('Train loss: ', loss_ / n_batch)

        model.save('test.h5')

    def get_data_for_batch(self):
        if self.idx_bs >= self.n_sample // self.bs:
            self.idx_bs = 0
            perm = list(range(self.n_sample))
            np.random.shuffle(perm)
            self.full_path = self.full_path[perm]
            self.labels = self.labels[perm]

        start = self.idx_bs * self.bs
        self.idx_bs += 1
        X = self.get_mel(self.full_path[start: start + self.bs])
        Y = self.labels[start: start + self.bs]
        return X, Y

    def get_mel(self, paths):
        X = []
        for path in paths:
            data = self.aug.read_audio_file(path)
            S = librosa.feature.melspectrogram(data, sr=16000, n_mels=40, n_fft=400, hop_length=int(0.010 * 16000),
                                               power=1)
            x = librosa.power_to_db(S ** 2, ref=np.median)
            x = x.swapaxes(0, 1)
            x = np.expand_dims(x, axis=2)
            X.append(x)

        return np.array(X)


#
# model = CRNN(ep=1, bs=32)
# model.training()

def get_mel_single(path):
    aug = Augment.AudioAugmentation()
    data = aug.read_audio_file(path)
    S = librosa.feature.melspectrogram(data, sr=16000, n_mels=40, n_fft=400, hop_length=int(0.010 * 16000),
                                       power=1)
    x = librosa.power_to_db(S ** 2, ref=np.median)
    x = x.swapaxes(0, 1)
    x = np.expand_dims(x, axis=2)

    return x

data = get_mel_single('/home/viet-pc/Desktop/rikkeisoft/trigger-word/data_model2/happy/0bde966a_nohash_0.wav')
print(data.shape)
data = np.expand_dims(data, axis=0)
print(data.shape)
m = load_model('test.h5')
print(m.predict(data))
