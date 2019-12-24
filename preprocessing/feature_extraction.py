import gammatone.fftweight as fftweight
import numpy as np
from scipy.fftpack import dct
from scipy.signal import butter, lfilter
from preprocessing.gfcc.feature_extractor import cochleagram_extractor
from preprocessing.gfcc.gfcc_extractor import gfcc_extractor
from utils import *
import librosa


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def get_gfccc_features(data, sr, npersegment, nperhop, gf_channels=60, cc_channels=31, window_type='hanning'):
    cochlea = cochleagram_extractor(data, sr, npersegment, nperhop, gf_channels, window_type)
    gfcc = gfcc_extractor(cochlea, gf_channels, cc_channels)
    return gfcc


# variables
def get_mfcc(signal, sr, pre_emphasis=0.97, frame_size=0.092, frame_stride=0.046, NFFT=512, nfilt=40, num_ceps=25):
    sample_rate = sr
    emphasized_signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])
    frame_length, frame_step = frame_size * sample_rate, frame_stride * sample_rate  # Convert from seconds to samples
    signal_length = len(emphasized_signal)
    frame_length = int(round(frame_length))
    frame_step = int(round(frame_step))

    # Make sure that we have at least 1 frame
    num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))

    pad_signal_length = num_frames * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))

    # Pad Signal to make sure that all frames have equal number of samples
    # without truncating any samples from the original signal
    pad_signal = np.append(emphasized_signal, z)

    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]

    frames *= np.hamming(frame_length)

    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))  # Magnitude of the FFT
    pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))  # Power Spectrum

    low_freq_mel = 0
    high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))  # Convert Hz to Mel
    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
    hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz
    bin = np.floor((NFFT + 1) * hz_points / sample_rate)

    fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m - 1])   # left
        f_m = int(bin[m])             # center
        f_m_plus = int(bin[m + 1])    # right

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
    filter_banks = 20 * np.log10(filter_banks)  # dB

    # mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1 : (num_ceps + 1)] # Keep 2-13
    mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')
    mfcc = mfcc.T
    return mfcc


def get_extracted_feature(x, feature_type='gfcc'):
    assert feature_type in feature_types
    if feature_type == 'mfcc':
        return  get_mfcc(x, sampling_rate, frame_size=win_length, frame_stride=stride_length, NFFT=512, nfilt=n_filters, num_ceps=24)
    elif feature_type == 'gfcc':
        return get_gfccc_features(x, sampling_rate, win_size, stride_size, n_filters, n_filters)
    elif feature_type == 'cochleagram':
        return cochleagram_extractor(x, sampling_rate, win_size, stride_size, n_filters, 'hanning')
    elif feature_type == 'melspec':
        return librosa.feature.melspectrogram(x, sr=sampling_rate, n_fft=win_size, hop_length=stride_size)