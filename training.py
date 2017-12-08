import config
from cnn import CNN
from data_provider import DataProvider


class Trainer:
    @classmethod
    def train(cls):
        model = CNN.model()
        gen_train, gen_val = DataProvider.get_generators()

        model.summary()

        model.fit_generator(
            gen_train,
            steps_per_epoch=config.NUM_STEPS_PER_EPOCH,
            epochs=50,
            validation_data=gen_val,
            validation_steps=config.NUM_VALIDATION_STEPS
        )
        model.save_weights('first_try.h5')


if __name__ == '__main__':
    Trainer.train()
