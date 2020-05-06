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


# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# tf.keras.backend.clear_session()

# warnings.filterwarnings('ignore')
# # coins = data.coins()

# def load_tensorflow_shared_session(self):
#     """ Load a Tensorflow/Keras shared session """
#     # LP: create a config by gpu cpu backend
#     if os.getenv('HAS_GPU', '0') == '1':
#         config = tf.ConfigProto(
#             device_count={'GPU': 1},
#             intra_op_parallelism_threads=1,
#             allow_soft_placement=True)
#         config.gpu_options.allow_growth = True
#         config.gpu_options.per_process_gpu_memory_fraction = 0.6
#     else:
#         config = tf.ConfigProto(
#             intra_op_parallelism_threads=1,
#             allow_soft_placement=True)
#      # LP: create session by config
#     self.tf_session = tf.Session(config=config)

#     return self.tf_session

config = tf.ConfigProto(
            device_count={'GPU': 1},
            intra_op_parallelism_threads=1,
            allow_soft_placement=True)
sess = tf.Session(config=config)
graph = tf.get_default_graph()

def load_saved_model(model_path):
    '''
    Takes in a saved model name as a string and returns the loaded model
    '''
    print('running')
    global sess
    global graph
    with graph.as_default():
        set_session(sess)
        model = load_model(model_path)
        return model

unet = load_saved_model('../../models/edge_detect/unet2.keras')

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
    # im = gray2rgb(image)
    # if isinstance(image, str):
    #     im = imread(image)
    # else:
    #     im = image

    # if len(im.shape) == 2:
    #     im = to_rgb1(im)
    
    global graph
    global sess
    with graph.as_default():
        set_session(sess)
        
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
            #     im_resize=cv2.cvtColor(im_resize, cv2.COLOR_RGB2GRAY)

            # canny_pred = blurr_canny(float_image_to_uint8(im_resize))

        return pred


def color_to_gray(img):
    image = predict_custom_image(img)
    gray_image = rgb2gray(img)
    return gray_image


def load_val_generator():
    global sess
    global graph
    with graph.as_default():
        set_session(sess)
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

def evaluate_generator(model, generator):
    global sess
    global graph
    with graph.as_default():
        set_session(sess)
        return dict(zip(model.metrics_names,model.evaluate_generator(generator)))
