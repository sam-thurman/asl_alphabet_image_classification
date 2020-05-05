'''
Code for making predictions with model(s) stored in flask/src/models
'''
import warnings
import sys
from skimage.color import rgb2gray, gray2rgb
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage import data
from imageio import imread, imsave
import numpy as np
import skimage
from skimage.color import rgb2gray
from skimage import io
from skimage import transform
import keras
import tensorflow as tf
from keras.models import load_model


def load_saved_model(model_name):
    '''
    Takes in a saved model name as a string and returns the loaded model
    (model19.h5)
    '''
    print('running')
    return keras.models.load_model(f'src/models/{model_name}')


def preprocess_for_predict(file_name):
    '''
    Takes in a saved image name (inside src/test_images) as a string, and returns 
    the preprocessed image compatible with model input
    '''

    data_path = f'src/test_images/{file_name}'
    # load file as image
    X = io.imread(data_path)
    # convert to grayscale
    X = rgb2gray(X)

    # resize for model expected shape
    X = transform.resize(X, (200, 200))
    X = (X - 0.5)*2
    X = np.expand_dims(X, axis=2)
    X = np.expand_dims(X, axis=0)
    return X


def predict_on(model, image):
    '''
    Predict instance using given model
    '''

    soft = model.predict(image)
    pred = np.argmax(soft)
    return pred


# import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
coins = data.coins()


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


def predict_custom_image(model, image):

    # im = gray2rgb(image)
    # if isinstance(image, str):
    #     im = imread(image)
    # else:
    #     im = image

    # if len(im.shape) == 2:
    #     im = to_rgb1(im)

    target_size = model.input.__dict__['_keras_shape'][1:-1]
    im_resize = resize(image, target_size)
    gray = gray2rgb(im_resize[:, :, 0])
    im = np.expand_dims(gray, axis=0)
    preds = model.predict(im)
    pred = np.float_(preds[:, :, :, 0][0])
    pred = resize(pred, (200, 200))
    pred = np.expand_dims(pred, axis=2)
    pred = np.expand_dims(pred, axis=0)

    #     im_resize=cv2.cvtColor(im_resize, cv2.COLOR_RGB2GRAY)

    # canny_pred = blurr_canny(float_image_to_uint8(im_resize))

    return pred
    # return to_rgb1(pred)


def color_to_gray(img):
    image = predict_custom_image(img)
    gray_image = rgb2gray(img)
    return gray_image
