from keras.preprocessing.image import ImageDataGenerator

import config


class DataProvider:
    @classmethod
    def get_generators(cls):
        train_gen = cls._get_train_gen()
        test_gen = cls._get_test_gen()

        return train_gen, test_gen

    @classmethod
    def _get_train_gen(cls):
        generator = cls.data_train_generator()
        return generator.flow_from_directory(
            config.URL_PAINTINGS,
            target_size=config.IMAGE_DIM,
            batch_size=config.NUM_PAINTINGS
        )

    @classmethod
    def _get_test_gen(cls):
        generator = cls.data_test_generator()
        return generator.flow_from_directory(
            config.URL_PAINTINGS,
            target_size=config.IMAGE_DIM,
            batch_size=config.NUM_PAINTINGS
        )

    @classmethod
    def data_train_generator(cls):
        return ImageDataGenerator(
            featurewise_center=True,
            featurewise_std_normalization=True,
            rotation_range=180,
            zoom_range=0.2,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.3,
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode='reflect')

    @classmethod
    def data_test_generator(cls):
        return ImageDataGenerator(
            rotation_range=90,
            zoom_range=0.2
        )
