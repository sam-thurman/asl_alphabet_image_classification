# SignBot
### Image classification for letters of ASL alphabet

This project serves as proof of concept for an algorithm that can detect and interpret ASL from images and live video.  The model used in this project was a Keras Sequential() model running on Tensorflow 1.10.  CNN was trained on 87000 jpg images of hands, 3000 images per letter of the American Sign Language alphabet, as well as 3000 images for 'delete', 'space', and 3000 images of blank rooms categorized as 'nothing'.  Image data can be found and downloaded [here](https://www.kaggle.com/grassknoted/asl-alphabet).  

The goal of this project was to gain a deeper understanding of Convolutional Neural Network architecture, as well as to explore real time classification using computer vision.  Although work on this project is purely exploratory, a possible application for this algorithm would be automating ASL recognition in video conferencing, removing need for human translators.

To access this repository and it's contents, clone down the repo and use the `env.yml` file to install the required packages.  For more information on how to do that, go [here](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file) for a conda cheat sheet.

Most current model structure is as follows:
```
# 1st conv layer
model.add(Conv2D(32,(3,3), padding='same', input_shape=(image_size, image_size,1)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# 2nd conv layer
model.add(Conv2D(64,(5,5), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# 3rd conv layer
model.add(Conv2D(128,(3,3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# 4th conv layer
model.add(Conv2D(128,(3,3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# flattening
model.add(Flatten())

# 1st fully connected layer
model.add(Dense(128))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.4))

# 2nd fully connected layer
model.add(Dense(64))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.3))

model.add(Dense(nb_classes, activation='softmax'))
```
## Data Preparation
There were two portions of data preparation that need to happen in order to make this project work.  First, I had to generate usable data to train my model.  Second, I had to produce a groomed video feed, meaning a live video capture whose frames were formatted in an interpretable way for the model.
#### For modeling
For the modeling phase, data preparation was pretty minimal in terms of hands-on effort.  I chose Keras's ImageDataGenerator module because it has an easy interface to read in images directly from your local database, and has some great pre-processing functionality.  In terms of pre-processing, the image pixel d
#### For predictions via OpenCV 