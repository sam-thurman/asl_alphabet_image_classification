import detect_edges
import numpy as np
import cv2
import imutils
import os
import tensorflow as tf
import keras
from keras import Model, Sequential
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage.transform import resize
# key dictionary from validation generator
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

cap = cv2.VideoCapture(0)
unet = load_model('../unet2.keras')
classifier = load_model('../model8.h5')
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here

    # turn to grayscale
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Crop edges of webcam view to make square
    (h, w, c) = frame.shape
    margin = (int(w)-int(h))/2
    square_feed = [0, int(h), int(0 + margin), int(int(w) - margin)]
    #                 y1            y2                x1            x2
    square_roi = frame[square_feed[0]:square_feed[1],
                       square_feed[2]:square_feed[3]]

    # Resize for model input

    # flip horizontally for easier user interpretability
    flip = cv2.flip(square_roi, 1)
    # display model output on screen
    model_in = flip.copy()
    mask_in = detect_edges.predict_custom_image(image=model_in, model=unet)
    input_size = 200
    # resized = cv2.resize(model_in, (input_size, input_size))
    # model_in = np.expand_dims(model_in, axis=2)
    model_in = detect_edges.to_rgb1(mask_in)
    model_in = np.expand_dims(model_in, axis=0)
    resized_model_in = resize(model_in, (1, input_size, input_size, 3))
    output = np.argmax(classifier.predict(resized_model_in))
    letter_predict = list(key_dict.keys())[
        list(key_dict.values()).index(output)]
    cv2.putText(flip, letter_predict, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    print(letter_predict)
    # Display the resulting frame
    cv2.imshow("Masked/predicted", model_in)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
del model
print(output.shape)
print('width={}, height={}'.format(w, h))

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
# del model
