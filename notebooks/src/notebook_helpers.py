import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from imageio import imread, imsave
from skimage import data
from skimage.transform import resize
from skimage.color import rgb2gray, gray2rgb

import keras
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.python.keras.backend import set_session

import sys
import os

# don't print annoying INFO and WARNING messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

unet = load_model('../../models/edge_detect/unet2.keras')


def load_saved_model(model_path):
    '''
    Takes in a saved model name as a string and returns the loaded model
    '''
    print('running')
    model = load_model(model_path)
    print(model.summary())

    return model


# def blurr_canny(im, sigma=0.2):
#     blur = cv2.GaussianBlur(im, (5, 5), 0)
#     return auto_canny(blur)


def float_image_to_uint8(im):

    return (im * 255).round().astype('uint8')


def predict_custom_image(image):

    model = unet
    target_size = model.input.__dict__['_keras_shape'][1:-1]
    im_resize = resize(image, target_size)
    gray = gray2rgb(im_resize[:, :, 0])
    im = np.expand_dims(gray, axis=0)
    preds = model.predict(im)
    pred = np.float_(preds[:, :, :, 0][0])
    pred = resize(pred, (200, 200))
    pred = np.expand_dims(pred, axis=2)

    return pred


def color_to_gray(img):

    image = predict_custom_image(img)
    gray_image = rgb2gray(img)

    return gray_image


def load_val_generator(path_to_val_data):

    image_size = 200
    batch_size = 32

    valgen = keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255, preprocessing_function=predict_custom_image)
    val_generator = valgen.flow_from_directory(
        path_to_val_data,  # directory for validation images
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode='categorical',
        color_mode='grayscale',
        shuffle=False)

    return val_generator


def evaluate_model_on_gen(model, generator):

    evaluation = dict(zip(model.metrics_names,
                          model.evaluate_generator(generator)))
    num_samps = len(generator[0][0])

    print(f'Accuracy on {num_samps} samples in given generator: ',
          evaluation['accuracy'])
    print(f'Loss on {num_samps} samples in given generator: ',
          evaluation['loss'])

    return evaluation


key_dict = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4,
            'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9,
            'K': 10, 'L': 11, 'M': 12, 'N': 13, 'O': 14,
            'P': 15, 'Q': 16, 'R': 17, 'S': 18, 'T': 19,
            'U': 20, 'V': 21, 'W': 22, 'X': 23, 'Y': 24,
            'Z': 25, 'delete': 26, 'nothing': 27, 'space': 28}


def predict_on_gen_and_report(model, generator, key=key_dict):

    true_labels = generator.labels
    true_letters = []
    for label in true_labels:
        true_letters.append(list(key.keys())[list(key.values()).index(label)])

    preds = model.predict_generator(generator)
    pred_maxs = []
    for pred in preds:
        pred_maxs.append(np.argmax(pred))
    pred_letters = []
    for index in pred_maxs:
        pred_letters.append(list(key.keys())[list(key.values()).index(index)])

    results_df = pd.concat([pd.DataFrame(np.array(true_letters).reshape(-1, 1)).rename(columns={0: "True Labels"}),
                            pd.DataFrame(np.array(pred_letters).reshape(-1, 1)).rename(columns={0: "Predicted Labels"})], axis=1)
    results_df.index.name = 'Sample Number'

    return results_df
