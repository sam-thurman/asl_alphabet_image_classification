import warnings
import pandas as pd
import numpy as np
import scipy as math
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from PIL import Image

from sklearn.model_selection import train_test_split
import skimage
from skimage.color import rgb2gray, gray2rgb
from skimage import data
from skimage.transform import resize
from imageio import imread, imsave


import keras
from keras import regularizers
from keras.models import load_model
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.layers import Dense, Input, Dropout, GlobalAveragePooling2D, Flatten, Conv2D, Conv1D, BatchNormalization, Activation, MaxPooling2D, MaxPooling1D
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

import sys
import os
from load_model import load_saved_model
unet = load_saved_model('models/unet2.keras')


def to_rgb1(im):
    # I think this will be slow
    w, h = im.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = im
    ret[:, :, 1] = im
    ret[:, :, 2] = im
    return ret

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
    # pred = np.expand_dims(pred, axis=2)
    pred = resize(pred, (200, 200))
    pred = np.expand_dims(pred, axis=2)

    return pred


def color_to_gray(img):
    image = predict_custom_image(img)
    gray_image = rgb2gray(img)
    return gray_image


def create_train_test_generator(folder_path):
    image_size = 200
    batch_size = 32
    datagen = keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        preprocessing_function=predict_custom_image,
        validation_split=0.3)
    train_generator = datagen.flow_from_directory(
        folder_path,  # directory for training images
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode='categorical',
        color_mode='grayscale',
        shuffle=False,
        subset='training')
    test_generator = datagen.flow_from_directory(
        folder_path,  # directory for training images
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode='categorical',
        color_mode='grayscale',
        shuffle=False,
        subset='validation')
    return train_generator, test_generator


def create_master_train_generator(folder_path):
    image_size = 200
    batch_size = 32
    datagen = keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        preprocessing_function=predict_custom_image)
    train_generator = datagen.flow_from_directory(
        folder_path,  # directory for training images
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode='categorical',
        color_mode='grayscale',
        shuffle=False)
    return train_generator


def create_val_generator(folder_path):
    image_size = 200
    batch_size = 32
    valgen = keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        preprocessing_function=predict_custom_image)
    val_generator = valgen.flow_from_directory(
        folder_path,  # directory for validation images
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode='categorical',
        color_mode='grayscale',
        shuffle=False)
    return val_generator


def generator_mask_viz_check(generator):
    plt.imshow(array_to_img(generator[0][0][0]))


def compile_model():
    model = Sequential()
    model.add(Conv2D(128, (3, 3), padding='same', input_shape=(200, 200, 1)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(29, activation='softmax',
                    activity_regularizer=regularizers.l1(0.02)))
    optimizer = Adam(lr=0.01)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())
    return model


def model_fit_gen_and_save(model=None, train_generator=None, val_generator=None, num_epochs=None, model_name=None):
    model.fit_generator(
        train_generator, validation_data=val_generator, epochs=num_epochs)
    model.save(f'{model_name}.h5')


def compile_train_save_model(train_generator=None, val_generator=None, num_epochs=None, model_name=None):
    model = compile_model()
    model_fit_gen_and_save(model=model, train_generator=train_generator,
                           val_generator=val_generator, num_epochs=num_epochs, model_name=model_name)
