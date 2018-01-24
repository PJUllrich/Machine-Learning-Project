from keras.applications import VGG16
from keras.models import load_model, Sequential
from keras.optimizers import Adam

import config
from cnn import CNN
from data_provider import DataProvider


class Validator:
    @classmethod
    def validate(cls):
        print('Loading model...')
        model = load_model(config.URL_CNN_MODEL)
        cls.run_validation(model)

    @classmethod
    def validate_VGG16(cls):
        # Create a VGG16 Model without top layers
        model_base = VGG16(include_top=False, input_shape=CNN.input_shape())

        # Create top layers from own specifications
        model_top = CNN.get_top_layers(input_shape=model_base.output_shape[1:])
        model_top.load_weights(config.URL_TOP_MODEL)

        # Combine the 2 models into a full model
        model_full = Sequential()
        for layer in model_base.layers:
            model_full.add(layer)
        model_full.add(model_top)

        # Lock the pre-trained VGG16 layers
        for layer in model_full.layers[:25]:
            layer.trainable = False

        # Compile the full model
        model_full.compile(
            loss='categorical_crossentropy',
            optimizer=Adam(),
            metrics=['accuracy']
        )

        # Run the validation
        cls.run_validation(model_full)

    @classmethod
    def run_validation(cls, model):
        _, gen_val = DataProvider.get_generators()

        print('Evaluating model...')
        result = model.evaluate_generator(gen_val, use_multiprocessing=True, workers=4)

        print('Results:')
        for idx, metric in enumerate(model.metrics_names):
            print(f'\t{metric}: {result[idx]}')


if __name__ == '__main__':
    Validator.validate()
