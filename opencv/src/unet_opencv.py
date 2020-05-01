# Import required packages
import detect_edges
import numpy as np
import cv2
import imutils
import os
from os.path import join
import tensorflow as tf
import keras
from keras import Model, Sequential
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage.transform import resize

# Key dictionary from validation generator
key_dict = {'A': 0, 'B': 1, 'C': 2,
            'D': 3, 'E': 4, 'F': 5,
            'G': 6, 'H': 7, 'I': 8,
            'J': 9, 'K': 10, 'L': 11,
            'M': 12, 'N': 13, 'O': 14,
            'P': 15, 'Q': 16, 'R': 17,
            'S': 18, 'T': 19, 'U': 20,
            'V': 21, 'W': 22, 'X': 23,
            'Y': 24, 'Z': 25, 'del': 26,
            'nothing': 27, 'space': 28}

# Init video capture
cap = cv2.VideoCapture(0)
# Load edger and classifier
model_path = '../../models/'
edger = load_model(join(model_path, 'edge_detect/unet2.keras'))
classifier = load_model(join(model_path, 'model3.h5'))

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Crop edges of webcam view to make square
    (h, w, c) = frame.shape
    margin = (int(w)-int(h))/2
    square_feed = [0, int(h), int(0 + margin), int(int(w) - margin)]
    #                 y1            y2                x1            x2
    square_roi = frame[square_feed[0]:square_feed[1],
                       square_feed[2]:square_feed[3]]
    # Flip horizontally for easier user interpretability
    flip = cv2.flip(square_roi, 1)
    # Generate edge mask of frame
    mask = detect_edges.predict_custom_image(image=flip, model=edger)
    # Copy frame for model input
    model_in = mask.copy()

    # Frame from gray --> RGB
    model_in = detect_edges.to_rgb1(model_in)
    # Add batch dimension
    model_in = np.expand_dims(model_in, axis=0)
    # Resize model input
    input_size = 200
    resized_model_in = resize(model_in, (1, input_size, input_size, 3))
    # Classify and print class to original (shown) frame
    output = np.argmax(classifier.predict(resized_model_in))
    letter_predict = list(key_dict.keys())[
        list(key_dict.values()).index(output)]
    cv2.putText(mask, letter_predict, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    print(letter_predict)

    # Display the resulting frame
    cv2.imshow("Masked/predicted", mask)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
print(output.shape)
print('width={}, height={}'.format(w, h))

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
del classifier
del unet
