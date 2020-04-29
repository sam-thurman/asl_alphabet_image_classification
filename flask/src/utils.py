'''
Code for making predictions with model(s) stored in flask/src/models
'''
import numpy as np

import skimage
from skimage.color import rgb2gray
from skimage import io
from skimage import transform

import keras
import tensorflow
from keras.models import load_model


def load_saved_model(model_name):
    '''
    This function takes in a saved model name as a string and returns the loaded model
    (copy_model3.h5)
    '''
    print('running')
    model_path = f'../models/{model_name}'
    return keras.models.load_model(model_path)


def preprocess_for_predict(file_name):
    data_path = f'src/test_images/{file_name}'
    Xi = io.imread(data_path)
    Xi = rgb2gray(Xi)
    Xi = transform.resize(Xi, (128, 128))
    Xi = (Xi - 0.5)*2
    Xi = np.expand_dims(Xi, axis=2)
    Xi = np.expand_dims(Xi, axis=0)
    return Xi


def predict_on(model, image):
    soft = model.predict(image)
    pred = np.argmax(soft)
    return pred
