from keras.models import load_model

import config
from data_provider import DataProvider


class Validator:
    @classmethod
    def validate(cls):
        _, gen_val = DataProvider.get_generators()

        print('Loading model...')
        model = load_model(config.URL_CNN_MODEL)

        print('Evaluating model...')
        result = model.evaluate_generator(gen_val, use_multiprocessing=True, workers=4)

        print('Results:')
        for idx, metric in enumerate(model.metrics_names):
            print(f'\t{metric}: {result[idx]}')


if __name__ == '__main__':
    Validator.validate()
