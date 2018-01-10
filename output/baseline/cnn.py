from keras import Sequential
from keras.layers import Activation, BatchNormalization, Conv2D, Dense, Dropout, \
    Flatten, \
    MaxPooling2D, PReLU
from keras.optimizers import Adam
from keras.regularizers import l2

import config


class CNN:
    @classmethod
    def model(cls):
        model = Sequential()

        model.add(cls._init_conv2D())
        model.add(Flatten())
        model.add(cls._dense(config.NUM_PAINTERS))
        model.add(Activation("softmax"))

        model.compile(
            loss='categorical_crossentropy',
            optimizer=Adam(lr=0.000074),
            metrics=['accuracy']
        )

        return model

    @staticmethod
    def _conv2D(filters):
        return Conv2D(filters, config.CONV_KERNEL, padding='same',
                      kernel_regularizer=l2(config.L2_REG))

    @staticmethod
    def _batchNorm():
        return BatchNormalization(axis=1)

    @staticmethod
    def _maxPooling2D():
        return MaxPooling2D(pool_size=(2, 2))

    @staticmethod
    def _dense(size):
        return Dense(size, kernel_regularizer=l2(config.L2_REG))

    @staticmethod
    def _init_conv2D():
        return Conv2D(16, config.CONV_KERNEL, input_shape=(3, *config.IMAGE_DIM),
                      kernel_regularizer=l2(config.L2_REG), padding='same')
