from keras.models import load_model

import config
from cnn import CNN
from data_provider import DataProvider


class Trainer:
    @classmethod
    def train(cls, load=False):
        if load:
            print('Loading model...')
            model = load_model(config.URL_CNN_MODEL)
        else:
            print('Creating new model...')
            model = CNN.model()

        cls.fit_model(model)

    @classmethod
    def train_top_layers(cls):
        top_layers = CNN.get_top_layers(input_shape=CNN.input_shape())
        top_layers_compiled = CNN.compile_model(top_layers)
        cls.fit_model(top_layers_compiled, save_full=False)

    @classmethod
    def fit_model(cls, model, save_full=True):
        gen_train, gen_val = DataProvider.get_generators()

        model.summary()
        model.fit_generator(
            gen_train,
            steps_per_epoch=config.NUM_STEPS_PER_EPOCH,
            epochs=config.NUM_EPOCH,
            validation_data=gen_val,
            validation_steps=config.NUM_VALIDATION_STEPS,
            use_multiprocessing=True,
            workers=4,
            shuffle=True
        )
        if save_full:
            model.save(config.URL_CNN_MODEL + '-new')
        else:
            model.save_weights(config.URL_CNN_MODEL + '-new')


if __name__ == '__main__':
    Trainer.train(load=True)
