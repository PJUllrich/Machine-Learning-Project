import os

from keras.preprocessing.image import ImageDataGenerator

import config
from data_selection import DataSelector


class DataProvider:
    @classmethod
    def get_generators(cls):
        DataSelector.select_data()

        train_gen = cls._get_train_gen()
        val_gen = cls._get_val_gen()

        return train_gen, val_gen

    @classmethod
    def _get_train_gen(cls):
        generator = cls.data_train_generator()
        return cls._get_gen_flow(generator, config.FOLDER_NAME_TRAINING)

    @classmethod
    def _get_val_gen(cls):
        generator = cls.data_val_generator()
        return cls._get_gen_flow(generator, config.FOLDER_NAME_VALIDATION)

    @classmethod
    def _get_gen_flow(cls, gen, folder):
        path = os.path.join(config.URL_DATA, folder)
        return gen.flow_from_directory(
            path,
            class_mode=config.CLASS_MODE,
            target_size=config.IMAGE_DIM,
            batch_size=config.BATCH_SIZE
        )

    @classmethod
    def data_train_generator(cls):
        return ImageDataGenerator(
            rotation_range=90,
            zoom_range=0.2,
            # width_shift_range=0.2,
            # height_shift_range=0.2,
            # rescale=1./255,
            # shear_range=0.3,
            # horizontal_flip=True,
            # vertical_flip=True
        )

    @classmethod
    def data_val_generator(cls):
        return ImageDataGenerator(
            rotation_range=90,
            zoom_range=0.2
        )
