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


def load_saved_model(model_path):
    '''
    Takes in a saved model name as a string and returns the loaded model
    '''
    print('running')
    model = load_model(model_path)
    return model
