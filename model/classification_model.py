from model.cnnlstm import CNNLSTMModel
from utils import weights_path
import numpy as np


class Model:
    def __init__(self):
        self.model = CNNLSTMModel()
        self.model.load_weights(weights_path)
        self.threshold = 0.7

    def predict(self, x):
        x = np.expand_dims(np.expand_dims(x, axis=-1), axis=0)
        predictions = self.model.predict(x)
        labels = np.array([3]*predictions.shape[0], dtype=np.int64)
        for i in range(predictions.shape[0]):
            label = np.argmax(predictions[i])
            labels[i] = label if predictions[i][label] > self.threshold else 3
        return labels

    def predict_one_label(self, x):
        predictions = self.predict(x)
        return np.argmax(predictions)


if __name__ == '__main__':
    model = Model()