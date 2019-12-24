# coding = utf-8
import numpy as np
# from scipy.io import wavfile
from preprocessing.gfcc.feature_extractor import cochleagram_extractor
# from matplotlib import pyplot as plt
import librosa
# from speech_utils import read_sphere_wav
import matplotlib.pyplot as plt
from preprocessing.gfcc.mfcc_extractor import get_mfcc


def gfcc_extractor(cochleagram, gf_channel, cc_channels):
    dctcoef = np.zeros((cc_channels, gf_channel))
    for i in range(cc_channels):
        n = np.linspace(0, gf_channel-1, gf_channel)
        dctcoef[i, :] = np.cos((2 * n + 1) * i * np.pi / (2 * gf_channel))
    # plt.figure()
    # plt.imshow(dctcoef)
    # plt.show()
    return np.matmul(dctcoef, cochleagram)


if __name__ == '__main__':
    # wav_data, wav_header = read_sphere_wav(u"../130_2b3_Tc_mc_AKGC417L.wav")
    # plt.subplot(211)
    wav_data, sr = librosa.core.load("../crackles_c.wav", sr=None)
    # plt.plot(wav_data)
    # plt.title("Tín hiệu thô")
    # mel_spec = librosa.feature.melspectrogram(wav_data, sr, hop_length=int(0.025*sr), win_length=int(0.01*sr))
    # mel_spec = librosa.power_to_db(mel_spec)
    # # mel_spec = get_mfcc(wav_data, sr, frame_size=0.025, frame_stride=0.01, num_ceps=40, nfilt=40, is_plotting=False)
    # plt.pcolormesh(mel_spec)
    # plt.title("Quang phổ mel")
    cochlea = cochleagram_extractor(wav_data, sr, int(0.025*sr), int(0.01*sr), 128, 'hanning')
    # cochlea = librosa.power_to_db(cochlea)
    # cochlea = 10 * np.log(cochlea)
    # gfcc = dct(z, type=2, norm='ortho')
    gfcc = gfcc_extractor(cochlea, 128, 40)
    # plt.figure(figsize=(12, 6))
    # plt.subplot(211)
    # plt.subplot(212)
    plt.pcolormesh(gfcc)
    # plt.title("Cochleagram")
    # plt.imshow(np.flipud(cochlea))
    # plt.subplot(212)
    # plt.pcolormesh(gfcc)
    # plt.imshow(np.flipud(gfcc))
    plt.show()
