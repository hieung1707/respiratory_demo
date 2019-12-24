from keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Input, Reshape, LSTM, BatchNormalization
from keras import regularizers
from keras.models import Model
from utils import weights_path
import numpy as np
from utils import class_mapping


class PredictionModel:
    def __init__(self):
        _NUM_CLASSES = 4
        input_shape = (128, 128, 1,)
        input_layer = Input(shape=input_shape)
        bn_axis = 2

        # Block 1
        x = BatchNormalization(axis=bn_axis)(input_layer)
        x = Conv2D(64, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv1')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool1')(x)

        # Block 2
        x = BatchNormalization(axis=bn_axis)(x)
        x = Conv2D(128, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool2')(x)

        # Block 3
        x = BatchNormalization(axis=bn_axis)(x)
        x = Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv3/conv3_1')(x)
        x = Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv3/conv3_2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool3')(x)

        # Block 4
        x = BatchNormalization(axis=bn_axis)(x)
        x = Conv2D(512, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv4/conv4_1')(x)
        x = Conv2D(512, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv4/conv4_2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool4')(x)

        # x = GlobalAveragePooling2D()(x)
        x = Reshape((8, -1))(x)
        # # Previously LSTM layer was 64 nodes
        x = LSTM(128, return_sequences=True, kernel_regularizer=regularizers.l2(0.01), stateful=False)(x)
        x = LSTM(128, return_sequences=True, kernel_regularizer=regularizers.l2(0.01), stateful=False)(x)
        x = Dropout(0.5)(x)
        # x = Dense(64, activation='relu')(x)
        x = Dense(_NUM_CLASSES, activation='softmax')(x)

        self.model = Model(inputs=input_layer, outputs=x)
        self.model.load_weights(weights_path)
        self.model._make_predict_function()
        self.threshold = 0.5

    def predict(self, x):
        x = np.expand_dims(x, axis=-1)
        predictions = self.model.predict(x)
        labels = np.full((predictions.shape[0], predictions.shape[1]), 3)
        # labels = np.array([3]*predictions.shape[1], dtype=np.int64)
        for j in range(predictions.shape[0]):
            for i in range(predictions.shape[1]):
                label = np.argmax(predictions[j, i])
                labels[j, i] = label if predictions[j, i, label] > self.threshold else 3
        return labels

    def predict_one_label(self, x):
        labels = self.predict(x)
        labels = np.squeeze(labels, axis=0)
        return class_mapping[np.argmax(np.bincount(labels))]


if __name__ == '__main__':
    model = PredictionModel()
    a = np.zeros((128, 128))
    prediction = model.predict(a)
    print(prediction)

