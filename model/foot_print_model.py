from scipy import signal
from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np
from preprocess import Augment
import time
import glob
import librosa
import librosa.display
from keras.layers import Input, Conv2D, GRU, Dense, Dropout, BatchNormalization, Bidirectional, Activation, Reshape
from keras.models import Model, load_model
import keras.backend as K
import warnings

warnings.filterwarnings("ignore")


class CRNN:
    def __init__(self, ep, bs):
        self.ep = ep
        self.bs = bs
        self.kw_path_train = glob.glob('../data_model2/train/happy/*')
        self.ne_path_train = glob.glob('../data_model2/train/negative/*')
        self.bg_path_train = glob.glob('../data_model2/train/bg/*')[:200]
        self.non_happy_from_mic = glob.glob('../create_data/non_happy/*')
        self.error_pred = glob.glob('../data_model2/train/error_pred/*')
        self.bg_path_train = self.bg_path_train + self.non_happy_from_mic + self.error_pred

        self.kw_path_dev = glob.glob('../data_model2/dev/happy/*')
        self.ne_path_dev = glob.glob('../data_model2/dev/negative/*')
        self.bg_path_dev = glob.glob('../data_model2/dev/bg/*')

        # self.similar_sound_train = glob.glob('../data_model2/train/error_pred/*')
        # self.similar_sound_dev = glob.glob('../data_model2/similar_sound/*')

        # Train set
        self.labels_train = np.array(
            [1] * len(self.kw_path_train) + [0] * len(self.ne_path_train) + [0] * len(self.bg_path_train))
        self.full_path_train = np.array(
            self.kw_path_train + self.ne_path_train + self.bg_path_train)

        # Dev set
        self.labels_dev = np.array(
            [1] * len(self.kw_path_dev) + [0] * len(self.ne_path_dev) + [0] * len(self.bg_path_dev))
        self.full_path_dev = np.array(self.kw_path_dev + self.ne_path_dev + self.bg_path_dev)

        self.n_sample = len(self.full_path_train)
        self.n_dev = len(self.full_path_dev)
        print('Sample: ', self.n_sample, 'Dev: ', self.n_dev)
        print('Training:')
        print('Happy: ', len(self.kw_path_train), 'Not Happy: ', len(self.ne_path_train) + len(self.bg_path_train))
        print('Dev:')
        print('Happy: ', len(self.kw_path_dev), 'Not Happy: ', len(self.ne_path_dev) + len(self.bg_path_dev))

        self.idx_bs = 0
        self.aug = Augment.AudioAugmentation()
        self.aug_list = [self.aug.add_noise, self.aug.shift, self.aug.stretch]

        # shuffle data
        perm = list(range(self.n_sample))
        np.random.shuffle(perm)
        self.full_path_train = self.full_path_train[perm]
        self.labels_train = self.labels_train[perm]

    def model(self, shape=(101, 40, 1)):
        input = Input(shape=shape)
        x = Conv2D(filters=32, kernel_size=(20, 5), strides=(8, 2))(input)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Reshape(target_shape=(11, 18 * 32))(x)
        x = Bidirectional(GRU(32, activation='relu', return_sequences=True))(x)
        x = Bidirectional(GRU(32, activation='relu'))(x)
        x = Dense(units=64)(x)
        x = Dense(units=2, activation='softmax')(x)

        return Model(input, x)

    def training(self):
        # model = self.model()
        model = load_model('/home/vietnv/PycharmProjects/trigger-word/model/model_6_0.9571788413098237.h5')
        model.compile('adam', loss='sparse_categorical_crossentropy', metrics=['acc'])
        X_dev, Y_dev = self.get_dev_set()
        for e in range(self.ep):
            s_time = time.time()
            n_batch = self.n_sample // self.bs
            loss_ = 0.
            acc_ = 0.
            for b in range(n_batch):
                X, Y = self.get_data_for_batch()
                his = model.fit(X, Y, batch_size=self.bs, epochs=1, verbose=0)
                loss_ += his.history['loss'][0]
                acc_ += his.history['acc'][0]

            print('Train loss: ', loss_ / n_batch, 'Train acc: ', acc_ / n_batch, 'Time:', time.time() - s_time)

            his_val = model.evaluate(X_dev, Y_dev, verbose=0)
            # if his_val[1] > max_acc:
            max_acc = his_val[1]
            model.save('model_mic2_1_' + str(max_acc) + '.h5')

    def get_dev_set(self):
        X_dev = self.get_mel(self.full_path_dev)
        Y_dev = self.labels_dev
        return X_dev, Y_dev

    def get_data_for_batch(self):
        if self.idx_bs >= self.n_sample // self.bs:
            self.idx_bs = 0
            perm = list(range(self.n_sample))
            np.random.shuffle(perm)
            self.full_path_train = self.full_path_train[perm]
            self.labels_train = self.labels_train[perm]

        start = self.idx_bs * self.bs
        self.idx_bs += 1
        X = self.get_mel(self.full_path_train[start: start + self.bs])
        Y = self.labels_train[start: start + self.bs]
        return X, Y

    def get_mel(self, paths):
        X = []
        for path in paths:
            aug_idx = set(np.random.choice(3, np.random.randint(4)))
            data = self.aug.read_audio_file(path)
            for i in aug_idx:
                data = self.aug_list[i](data)
            S = librosa.feature.melspectrogram(data, sr=16000, n_mels=40, n_fft=400, hop_length=int(0.010 * 16000),
                                               power=1)
            x = librosa.power_to_db(S ** 2, ref=np.median)
            x = librosa.feature.mfcc(S=x, n_mfcc=40)
            x = x.swapaxes(0, 1)
            x = np.expand_dims(x, axis=2)
            X.append(x)

        return np.array(X)


model = CRNN(ep=20, bs=32)
model.training()

# def get_mel_single(path):
#     aug = Augment.AudioAugmentation()
#     data = aug.read_audio_file(path)
#     S = librosa.feature.melspectrogram(data, sr=16000, n_mels=40, n_fft=400, hop_length=int(0.010 * 16000),
#                                        power=1)
#     x = librosa.power_to_db(S ** 2, ref=np.median)
#
#     x = x.swapaxes(0, 1)
#
#     x = np.expand_dims(x, axis=2)
#
#     return x

#
# s = time.time()
# data = get_mel_single('/home/vietnv/PycharmProjects/trigger-word/data_model2/train/happy/4d9e07cf_nohash_0.wav')
# print(time.time() - s)
# data = np.expand_dims(data, axis=0)
# print(data.shape)
# m = load_model('test.h5')
# print(m.predict(data))
