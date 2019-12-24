import numpy
from scipy.fftpack import dct
import matplotlib.pyplot as plt
import librosa
import matplotlib.patches as patches


def get_mfcc(signal, sr, pre_emphasis=0.97, frame_size=0.092, frame_stride=0.046, NFFT=512, nfilt=40, num_ceps=12, is_plotting=True):
    # print(signal.shape)
    sample_rate = sr
    # plt.subplot(121)
    # plt.plot(signal)
    # plt.title("Dữ liệu chưa chỉnh tăng (a)")
    emphasized_signal = numpy.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])
    # plt.subplot(122)
    if is_plotting:
        plt.plot(emphasized_signal[24000:int(0.1*sr) + 24000])
        # plt.title("Dữ liệu đã chỉnh tăng (b)")
        plt.show()
    frame_length, frame_step = frame_size * sample_rate, frame_stride * sample_rate  # Convert from seconds to samples
    signal_length = len(emphasized_signal)
    frame_length = int(round(frame_length))
    frame_step = int(round(frame_step))
    # frame_length = NFFT
    # frame_step = int(NFFT / 2)
    num_frames = int(numpy.ceil(float(numpy.abs(signal_length - frame_length)) / frame_step))  # Make sure that we have at least 1 frame

    pad_signal_length = num_frames * frame_step + frame_length
    z = numpy.zeros((pad_signal_length - signal_length))
    pad_signal = numpy.append(emphasized_signal, z) # Pad Signal to make sure that all frames have equal number of samples without truncating any samples from the original signal

    indices = numpy.tile(numpy.arange(0, frame_length), (num_frames, 1)) + numpy.tile(numpy.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[indices.astype(numpy.int32, copy=False)]
    if is_plotting:
        window_1 = frames[4]
        plt.subplot(311)
        plt.plot(window_1)
        plt.title('Cửa sổ thông thường')
        plt.subplot(312)
        plt.plot(window_1*numpy.hamming(frame_length))
        plt.title('Cửa sổ áp dụng Hamming')
        plt.subplot(313)
        plt.plot(window_1*numpy.hanning(frame_length))
        plt.title('Cửa sổ áp dụng Hanning')
        plt.show()
    frames *= numpy.hamming(frame_length)
    # print(frames.shape)
    # frames *= 0.54 - 0.46 * numpy.cos((2 * numpy.pi * n) / (frame_length - 1))  # Explicit Implementation **

    mag_frames = numpy.absolute(numpy.fft.rfft(frames, NFFT))  # Magnitude of the FFT
    pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))  # Power Spectrum

    low_freq_mel = 0
    high_freq_mel = (2595 * numpy.log10(1 + (sample_rate / 2) / 700))  # Convert Hz to Mel
    mel_points = numpy.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
    hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz
    bin = numpy.floor((NFFT + 1) * hz_points / sample_rate)

    fbank = numpy.zeros((nfilt, int(numpy.floor(NFFT / 2 + 1))))
    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m - 1])   # left
        f_m = int(bin[m])             # center
        f_m_plus = int(bin[m + 1])    # right

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
    if is_plotting:
        plt.plot(fbank.T)
        plt.show()
    filter_banks = numpy.dot(pow_frames, fbank.T)
    filter_banks = numpy.where(filter_banks == 0, numpy.finfo(float).eps, filter_banks)  # Numerical Stability
    filter_banks = 10 * numpy.log10(filter_banks)  # dB
    return filter_banks.T
    # if is_plotting:
    #     plt.pcolormesh(filter_banks.T)
    #     plt.show()
    # # filter_banks = numpy.cbrt(filter_banks)
    #
    # # mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1 : (num_ceps + 1)] # Keep 2-13
    # mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1:]
    # # using cep lifter to restore high frequency
    # # (nframes, ncoeff) = mfcc.shape
    # # n = numpy.arange(ncoeff)
    # # cep_lifter = 23
    # # lift = 1 + (cep_lifter / 2) * numpy.sin(numpy.pi * n / cep_lifter)
    # # mfcc *= lift
    # mfcc = mfcc.T
    # return mfcc


if __name__ == '__main__':
    sig, fs = librosa.load('../crackles_a.wav', sr=None)
    mfcc = get_mfcc(sig, sr=fs, frame_size=0.025, frame_stride=0.01, nfilt=40, is_plotting=False)

    sig, fs = librosa.load('tracheal.wav', sr=None)
    mfcc_2 = get_mfcc(sig, sr=fs, frame_size=0.025, frame_stride=0.01, nfilt=40, is_plotting=False)


    # mel_filters = librosa.filters.mel(44100, n_fft=2048, n_mels=10)
    plt.subplot(121)
    plt.pcolormesh(mfcc_2)
    plt.title("Tiếng thở thông thường")
    plt.subplot(122)
    plt.pcolormesh(mfcc)
    plt.title("Tiếng thở có chứa tiếng rì rào")
    plt.show()

    # spec = librosa.feature.melspectrogram(sig, sr=44100, n_fft=2048, hop_length=512)
    # print(spec.shape)
    # plt.pcolormesh(spec)
    # plt.show()