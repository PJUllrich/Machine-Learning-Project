from keras import Sequential
from keras.layers import Activation, BatchNormalization, Conv2D, Dense, Dropout, \
    Flatten, \
    MaxPooling2D, PReLU
from keras.optimizers import Adam

import config


class CNN:
    @classmethod
    def model(cls):
        model = Sequential()

        model.add(Conv2D(16, config.CONV_KERNEL, input_shape=(3, *config.IMAGE_DIM)))

        model.add(cls._batchNorm())
        model.add(PReLU())
        model.add(cls._conv2D(16))
        model.add(cls._batchNorm())
        model.add(PReLU())
        model.add(cls._maxPooling2D())

        model.add(cls._conv2D(32))
        model.add(cls._batchNorm())
        model.add(PReLU())
        model.add(cls._conv2D(32))
        model.add(cls._batchNorm())
        model.add(PReLU())
        model.add(cls._conv2D(32))
        model.add(cls._batchNorm())
        model.add(PReLU())
        model.add(cls._maxPooling2D())

        model.add(cls._conv2D(64))
        model.add(cls._batchNorm())
        model.add(PReLU())
        model.add(cls._conv2D(64))
        model.add(cls._batchNorm())
        model.add(PReLU())
        model.add(cls._conv2D(64))
        model.add(cls._batchNorm())
        model.add(PReLU())
        model.add(cls._maxPooling2D())

        model.add(cls._conv2D(128))
        model.add(cls._batchNorm())
        model.add(PReLU())
        model.add(cls._conv2D(128))
        model.add(cls._batchNorm())
        model.add(PReLU())
        model.add(cls._conv2D(128))
        model.add(cls._batchNorm())
        model.add(PReLU())
        model.add(cls._maxPooling2D())

        model.add(cls._conv2D(256))
        model.add(cls._batchNorm())
        model.add(PReLU())
        model.add(cls._conv2D(256))
        model.add(cls._batchNorm())
        model.add(PReLU())
        model.add(cls._conv2D(256))
        model.add(cls._batchNorm())
        model.add(PReLU())
        model.add(cls._maxPooling2D())
        model.add(Dropout(0.5))

        model.add(Flatten())
        model.add(Dense(config.PENULTIMATE_SIZE))
        model.add(cls._batchNorm())
        model.add(PReLU())

        model.add(Dropout(0.5))
        model.add(Dense(config.SOFTMAX_SIZE))
        model.add(cls._batchNorm())
        model.add(Activation("softmax"))

        model.compile(
            loss='categorical_crossentropy',
            optimizer=Adam(lr=0.000074),
            metrics=['accuracy']
        )

        return model

    @staticmethod
    def _conv2D(filters):
        return Conv2D(filters, config.CONV_KERNEL)

    @staticmethod
    def _batchNorm():
        return BatchNormalization(axis=1)

    @staticmethod
    def _maxPooling2D():
        return MaxPooling2D(pool_size=(2, 2))
