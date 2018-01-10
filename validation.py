from keras.applications import VGG16
from keras.models import load_model
from keras.optimizers import Adam

import config
from data_provider import DataProvider


class Validator:
    @classmethod
    def validate(cls):
        print('Loading model...')
        model = load_model(config.URL_CNN_MODEL)
        cls.run_validation(model)

    @classmethod
    def validate_VGG16(cls):
        model = VGG16(weights=None, classes=config.NUM_PAINTERS)
        model.compile(
            loss='categorical_crossentropy',
            optimizer=Adam(lr=0.000074),
            metrics=['accuracy']
        )
        cls.run_validation(model)

    @classmethod
    def run_validation(cls, model):
        _, gen_val = DataProvider.get_generators()

        print('Evaluating model...')
        result = model.evaluate_generator(gen_val, use_multiprocessing=True, workers=4)

        print('Results:')
        for idx, metric in enumerate(model.metrics_names):
            print(f'\t{metric}: {result[idx]}')


if __name__ == '__main__':
    Validator.validate_VGG16()
