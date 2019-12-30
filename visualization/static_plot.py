import matplotlib.pyplot as plt
import librosa
from utils import *
from resampy import resample
from preprocessing import feature_extraction
import numpy as np
from model.classification_model import PredictionModel
import matplotlib.patches as mpatches
import matplotlib.colors as c


class StaticPlot():
    def __init__(self, feature_type='cochleagram'):
        self.fig = plt.figure(figsize=(18, 10))
        self.cMap = c.ListedColormap(['r', 'g', 'b', 'm', 'k'])
        self.feature_type = feature_type
        self.model = PredictionModel()

        self.wav = None
        self.filtered_wav = None
        self.features = None
        self.frame_lvl_preds = None
        self.video_lvl_preds = None
        self.ground_truth = None

    def analyze_and_visualize(self, file_path):
        # raw data
        ax1 = plt.subplot(321)
        self.load_wav(file_path)
        plt.plot(self.wav)
        plt.title('Raw sound')

        # filtered data
        # ax2 = plt.subplot(323, sharex=ax1)
        self.filter_wav()
        # plt.plot(self.filtered_wav)
        # plt.title('Heart sound filtered sound')

        # plot legend
        ax3 = plt.subplot(323)
        plt.legend([mpatches.Patch(color=b) for b in self.cMap.colors], existing_labels, loc='center', prop={'size': 20})
        plt.axis('off')
        plt.title('Description')

        # extract feature
        ax4 = plt.subplot(322)
        self.extract_feature()
        plt.pcolormesh(self.features)
        plt.title(self.feature_type)

        # predict labels for each frame
        ax5 = plt.subplot(324, sharex=ax4)
        self.predict_frame_level()
        plt.pcolormesh(np.flipud(self.frame_lvl_preds), cmap=self.cMap)
        plt.clim(0, 4)
        plt.title('Frame lvl prediction')

        # column voting for each timestamp
        plt.subplot(326, sharex=ax4)
        self.predict_video_level()
        plt.pcolormesh(self.video_lvl_preds, cmap=self.cMap)
        plt.clim(0, 4)
        plt.title('Video lvl prediction')

        if 'audio_and_txt_files' in file_path or 'test_files' in file_path:
            file_path = file_path[:-3] + 'txt'
            self.get_ground_truth(file_path)
            ax2 = plt.subplot(325)
            # self.filter_wav()
            plt.pcolormesh(self.ground_truth, cmap=self.cMap)
            plt.clim(0, 4)
            plt.title('Ground truth')
        plt.show()

    def get_ground_truth(self, file_path):
        assert self.wav is not None
        assert self.features is not None
        file_labels = read_labels(file_path)
        start_indices = []
        end_indces = []
        labels = []
        for label in file_labels:
            start_indices.append(label['start'])
            end_indces.append(label['end'])
            labels.append(label['label'])
        labels_np = np.full(self.wav.shape, 4)
        for i in range(len(start_indices)):
            labels_np[start_indices[i]:end_indces[i]] = labels[i]
        print(np.unique(labels_np, return_counts=True))
        self.ground_truth = np.full(self.features.shape[1], 4)
        idx = 0
        for i in range(0, labels_np.shape[0], int(sampling_rate * 0.01)):
            label_temp = labels_np[i: i+int((0.025*sampling_rate))]
            if label_temp.shape[0] != int(0.025*sampling_rate):
                break
            real_label = np.argmax(np.bincount(label_temp))
            self.ground_truth[idx] = real_label
            idx += 1
        self.ground_truth = np.expand_dims(self.ground_truth, axis=0)

    def load_wav(self, file_path):
        self.wav, sr = librosa.load(file_path, sr=None)
        if sr != sampling_rate:
            # print('wrong sample rate')
            self.wav = resample(self.wav, sr, sampling_rate)

    def filter_wav(self):
        assert self.wav is not None
        self.filtered_wav = feature_extraction.butter_bandpass_filter(self.wav, 50, 2500, sampling_rate, order=6)

    def extract_feature(self):
        assert self.filtered_wav is not None
        self.features = feature_extraction.get_extracted_feature(self.filtered_wav, feature_type=self.feature_type)

    def predict_frame_level(self):
        assert self.features is not None
        skip = 32
        X_val = []
        for i in range(0, self.features.shape[1], skip):
            if self.features[:, i:i + 128].shape[1] == 128:
                X_val.append(self.features[:, i:i + 128].T)

        X_val = np.array(X_val)
        # X_val = np.expand_dims(X_val, axis=-1)
        # print(X_val.shape)
        preds = self.model.predict(X_val)
        # print(preds.shape, features.shape[1])

        self.frame_lvl_preds = np.full((preds.shape[0], self.features.shape[1]), 4)
        current_pos = 0
        for idx, pred in enumerate(preds):
            real_prediction = np.zeros(128)
            for i in range(8):
                real_prediction[int(i * 16):int((i + 1) * 16)] = pred[i]
            # print(np.unique(real_prediction, return_counts=True))
            # real_prediction = np.expand_dims(real_prediction, axis=0)
            self.frame_lvl_preds[idx, current_pos:current_pos + 128] = real_prediction
            current_pos += skip

    def predict_video_level(self):
        assert self.frame_lvl_preds is not None
        self.video_lvl_preds = np.zeros(self.features.shape[1])
        for i in range(self.features.shape[1]):
            labels = self.frame_lvl_preds[:, i]
            timestamp_labels = labels[labels != 4]
            counts = np.bincount(timestamp_labels)
            if len(counts) == 0:
                most_prominent_label = 4
            else:
                most_prominent_label = np.argmax(counts)
            self.video_lvl_preds[i] = most_prominent_label
        self.video_lvl_preds = np.expand_dims(self.video_lvl_preds, axis=0)


if __name__ == '__main__':
    plot = StaticPlot()
