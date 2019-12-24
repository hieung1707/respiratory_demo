from keras.layers import Conv2D, MaxPooling2D, Dropout, Input, Reshape, LSTM, BatchNormalization, Dense, TimeDistributed
from keras.models import Model
from keras import regularizers


class CNNLSTMModel(Model):
    def __init__(self):
        super(CNNLSTMModel, self).__init__()
        _NUM_CLASSES = 4
        input_shape = (128, 128, 1,)
        self.input_layer = Input(shape=input_shape)
        bn_axis = 2
        # Block 1
        self.bn1 = BatchNormalization(axis=bn_axis)
        self.conv1 = Conv2D(64, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv1')
        self.pool1 = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool1')

        # Block 2
        self.bn2 = BatchNormalization(axis=bn_axis)
        self.conv2 = Conv2D(128, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv2')
        self.pool2 = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool2')

        # Block 3
        self.bn3 = BatchNormalization(axis=bn_axis)
        self.conv3_1 = Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv3/conv3_1')
        self.conv3_2 = Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv3/conv3_2')
        self.pool3 = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool3')

        # Block 4
        self.bn4 = BatchNormalization(axis=bn_axis)
        self.conv4_1 = Conv2D(512, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv4/conv4_1')
        self.conv4_2 = Conv2D(512, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv4/conv4_2')
        self.pool4 = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool4')

        self.reshape1 = Reshape((8, -1))
        self.lstm1 = LSTM(128, return_sequences=True, return_state=False, kernel_regularizer=regularizers.l2(0.01),
                          stateful=False)
        self.lstm2 = LSTM(128, return_sequences=True, return_state=False, kernel_regularizer=regularizers.l2(0.01),
                          stateful=False)
        self.dropout1 = Dropout(0.5)
        self.dense1 = Dense(_NUM_CLASSES, activation='softmax')
        outputs = self.create_model()
        self.inputs = self.input_layer
        self.outputs = outputs
        self.build(input_shape)

    def create_model(self):
        x = self.bn1(self.input_layer)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.bn2(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.bn3(x)
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.pool3(x)
        x = self.bn4(x)
        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.pool4(x)
        x = self.reshape1(x)
        x = self.lstm1(x)
        x = self.lstm2(x)
        x = self.dropout1(x)
        x = self.dense1(x)
        return x