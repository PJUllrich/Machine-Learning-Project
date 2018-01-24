# Machine Learning Project

## Setting up
### Data
1. Copy the `paintings.zip` from `Google Drive` into the `Project folder`
1. Extract the `paintings.zip` file.
1. Delete the `paintings.zip` file.

### Dev Environment
1. Install `Python 3.6.3`
1. Open a Terminal in your `Project Folder`
1. Create a Virtual Environment with `python3 -m venv ./venv`
1. Activate the Virtual Environment with `source venv/bin/activate`
1. Install the requirements with `pip install -r requirements.txt`

### Keras
1. Run `training.py` once. At first it won't work since Keras is not yet configured, but this will create a `.keras` folder in `$HOME`
1. Open the file `.keras/keras.json` in your Home directory.
1. Change the `backend` entry to 'theano'.
1. Change the `image_data_format` entry to `channels_first`
1. Save and exit the file.
1. Now, you should be able to run `training.py` without errors!

## Configuring the CNN
* All configurations can be made in `config.py`. I pushed the ones that are most interesting to the top of the file.
* The layers of the CNN can be changed in `cnn.py`. Make sure that the Flatten() layer is always the last one.

## Running the CNN
* Run the `training.py` script. This script trains and validates the CNN.

## Resources
https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
https://keras.io/
https://github.com/inejc/painters/blob/master/painters/train_cnn.py
https://www.wga.hu/index1.html