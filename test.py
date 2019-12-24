import librosa
import matplotlib.pyplot as plt


wav, sr = librosa.load('wheezing_a.wav', sr=None)
plt.plot(wav)
plt.show()
