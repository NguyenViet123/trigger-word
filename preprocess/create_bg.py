from pydub import AudioSegment
import glob
import numpy as np


def random_choice_background():
    paths = glob.glob('../data_model2/background/*')
    sel = np.random.randint(0, len(paths))
    bg = AudioSegment.from_wav(paths[sel])
    rand_start_time = np.random.randint(0, len(bg) - 1000 - 1)
    return bg[rand_start_time: rand_start_time + 1000]

num_bg = 300
while num_bg > 0:
    bg = random_choice_background()
    print(len(bg))
    bg.export('../data_model2/bg/'+'bg_' + str(num_bg) + '.wav', format='wav')
    num_bg -= 1
