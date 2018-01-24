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

        model.add(cls.init_conv2D())
        model.add(cls.batchNorm())
        model.add(PReLU())
        model.add(cls.conv2D(16))
        model.add(cls.batchNorm())
        model.add(PReLU())
        model.add(cls.conv2D(16))
        model.add(cls.batchNorm())
        model.add(PReLU())
        model.add(cls.maxPooling2D())

        model.add(cls.conv2D(32))
        model.add(cls.batchNorm())
        model.add(PReLU())
        model.add(cls.conv2D(32))
        model.add(cls.batchNorm())
        model.add(PReLU())
        model.add(cls.conv2D(32))
        model.add(cls.batchNorm())
        model.add(PReLU())
        model.add(cls.maxPooling2D())

        model.add(cls.conv2D(64))
        model.add(cls.batchNorm())
        model.add(PReLU())
        model.add(cls.conv2D(64))
        model.add(cls.batchNorm())
        model.add(PReLU())
        model.add(cls.conv2D(64))
        model.add(cls.batchNorm())
        model.add(PReLU())
        model.add(cls.maxPooling2D())

        model.add(cls.conv2D(128))
        model.add(cls.batchNorm())
        model.add(PReLU())
        model.add(cls.conv2D(128))
        model.add(cls.batchNorm())
        model.add(PReLU())
        model.add(cls.conv2D(128))
        model.add(cls.batchNorm())
        model.add(PReLU())
        model.add(cls.maxPooling2D())

        model.add(Flatten())
        model.add(cls.dense(config.PENULTIMATE_SIZE))
        model.add(cls.batchNorm())
        model.add(PReLU())

        model.add(Dropout(0.5))
        model.add(cls.dense(config.SOFTMAX_SIZE))
        model.add(cls.batchNorm())
        model.add(cls.dense(config.NUM_PAINTERS))
        model.add(Activation("softmax"))

        model_compiled = cls.compile_model(model)

        return model_compiled

    @staticmethod
    def conv2D(filters):
        return Conv2D(filters, config.CONV_KERNEL, padding='same',
                      kernel_regularizer=l2(config.L2_REG), activation='relu')

    @staticmethod
    def batchNorm():
        return BatchNormalization(axis=1)

    @staticmethod
    def maxPooling2D():
        return MaxPooling2D(pool_size=(2, 2))

    @staticmethod
    def dense(size):
        return Dense(size,  kernel_regularizer=l2(config.L2_REG), activation='relu')

    @staticmethod
    def init_conv2D():
        return Conv2D(16, config.CONV_KERNEL, input_shape=(3, *config.IMAGE_DIM),
                      kernel_regularizer=l2(config.L2_REG), padding='same')

    @classmethod
    def compile_model(cls, model):
        model.compile(
            loss='categorical_crossentropy',
            optimizer=Adam(lr=0.0001),
            metrics=['accuracy']
        )

        return model

    @classmethod
    def input_shape(cls):
        return (3, *config.IMAGE_DIM)
