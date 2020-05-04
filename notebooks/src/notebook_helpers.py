import warnings
import numpy as np
from imageio import imread, imsave
from skimage import data
from skimage.transform import resize
from keras.models import load_model
import matplotlib.pyplot as plt
from skimage.color import rgb2gray, gray2rgb
# import matplotlib.pyplot as plt
import sys
import keras
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.python.keras.backend import set_session
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
tf.keras.backend.clear_session()

warnings.filterwarnings('ignore')
coins = data.coins()

unet = load_model('../../models/edge_detect/unet2.keras')
graph = tf.get_default_graph()
sess = keras.backend.get_session()
init = tf.global_variables_initializer()
sess.run(init)

def load_saved_model(model_name):
    '''
    Takes in a saved model name as a string and returns the loaded model
    (model19.h5)
    '''
    print('running')
    return keras.models.load_model(f'../../models/{model_name}')

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
    global graph
    global sess
    with graph.as_default():
        set_session(sess)
        preds = model.predict(im)
        pred = np.float_(preds[:, :, :, 0][0])
        # pred = np.expand_dims(pred, axis=2)
        pred = resize(pred, (200, 200))
        pred = np.expand_dims(pred, axis=2)
        #     im_resize=cv2.cvtColor(im_resize, cv2.COLOR_RGB2GRAY)

        # canny_pred = blurr_canny(float_image_to_uint8(im_resize))

        return pred
    # return to_rgb1(pred)

def color_to_gray(img):
    image = predict_custom_image(img)
    gray_image = rgb2gray(img)
    return gray_image


def load_val_generator():
    validation_path = '../../data/asl_alphabet_validation'
    image_size = 200
    batch_size = 32
    valgen = keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255, preprocessing_function=predict_custom_image) 
    val_generator = valgen.flow_from_directory(
                validation_path,  # directory for validation images
                target_size=(image_size, image_size),
                batch_size=batch_size,
                class_mode='categorical',
                color_mode='grayscale',
                shuffle=False)
    return val_generator