from scipy import signal
from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np
from preprocess import Augment

# rate, audio = wavfile.read('../data_model2/happy/1a6eca98_nohash_0.wav')
# frequencies, times, spectrogram = signal.spectrogram(audio, fs=16000, nfft=100, noverlap=50, nperseg=100)
#
# plt.pcolormesh(times, frequencies, spectrogram)
# plt.imshow(spectrogram)
# plt.ylabel('Frequency [Hz]')
# plt.xlabel('Time [sec]')
# plt.show()
#
# print(audio)
# audio = audio / 32767 + 0.005 * np.random.randn(audio.shape[0])
# print(audio)
# wavfile.write('test.wav', rate=rate, data=audio)
# print(np.max(audio))
# print(np.min(audio))
# print(rate)
# print(frequencies)
# print(times)
# print(spectrogram.shape)


aug = Augment.AudioAugmentation()
data = aug.read_audio_file('../data_model2/happy/1a6eca98_nohash_0.wav')

print(data.shape)
