# Import required packages
import warnings
import numpy as np
from imageio import imread, imsave
from skimage import data
from skimage.transform import resize
from keras.models import load_model
import sys

warnings.filterwarnings('ignore')
coins = data.coins()


def to_rgb1(im):
    '''
    Add rgb dimensions
    Input: image (w, h)
    Output: image (w, h, c)
    '''
    w, h = im.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = im
    ret[:, :, 1] = im
    ret[:, :, 2] = im
    return ret


def blurr_canny(im, sigma=0.2):
    '''
    Iffy OpenCV edger
    '''
    blur = cv2.GaussianBlur(im, (5, 5), 0)
    return auto_canny(blur)


def float_image_to_uint8(im):
    return (im * 255).round().astype('uint8')


def predict_custom_image(image=None, model=None):
    '''
    Implement edger on an image and return masked frame
    Inputs: Image, frame to be masked
            Model, edger model with predict method that 
            takes in and returns an image
    Output: Masked image
    '''
    # Ensure proper format for input
    if isinstance(image, str):
        im = imread(image)
    else:
        im = image
    if len(im.shape) == 2:
        im = to_rgb1(im)
    target_size = model.input.__dict__['_keras_shape'][1:-1]
    im_resize = resize(im, target_size)
    im = np.expand_dims(im_resize, 0)
    # Predict on image - generate mask
    preds = model.predict(im)
    # Acces prediction image
    pred = preds[:, :, :, 0][0]

    return pred
