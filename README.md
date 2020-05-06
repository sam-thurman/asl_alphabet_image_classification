# SignBot
## Repository Structure
`usage_and_setup.md`--> *An important note on compatibility and running notebooks*
`environment.yml`--> Variables to recreate conda environment
`notebooks/final/`--> Final model evaluation and quick view of processed data <br>
`notebooks/exploratory/`--> Model construction, data cleaning, model explanations and evaluations <br>
`opencv/src/`--> Scripts for OpenCV live video capture with edge detection <br>
`flask`--> Code for flask deployment.  Flask app contains simple model input-->prediction demo <br>

## Image classification for letters of ASL alphabet
This project serves as proof of concept for an algorithm that can detect and interpret ASL from images and live video.  The neural net used in this project is a Keras Sequential() model running on the Tensorflow 2.1 backend.  Model was trained on 87000 jpg images of hands, 3000 images per letter of the American Sign Language alphabet, as well as 3000 images for 'delete', 'space', and 3000 images of blank rooms categorized as 'nothing'.  Image data can be found and downloaded [here](https://www.kaggle.com/grassknoted/asl-alphabet).  
<br>
The goal of this project was to gain a deeper understanding of Convolutional Neural Network architecture, as well as to explore real time classification using computer vision.  Although work on this project is purely exploratory, a possible application for this algorithm would be automating ASL recognition in video conferencing, removing need for human translators.
<br>
To access this repository and it's contents, clone down the repo and use the `env.yml` file to install the required packages.  For more information on how to do that, go [here](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file) for a conda cheat sheet.

## Model Architecture and Performance
The final model used for predicting on this dataset was trained on all 87000 images provided in the training set, and achieved 93.333% accuracy on the validation data.  
<br>
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
## Data Understanding
As mentioned above, the data used in this project was 87000 training and 29 validation images from a single source on Kaggle.  Given the initial time constraints attached to this project, outside data was not able to be brought in. The images in the dataset were all taken by the source and all have, for the most part, the same style and format.  The backgrounds behind the hands in the images were very similar if not the same in every frame, and the arm was always bare(i.e. no long clothing or other accessories).  The hand was also often underexposed, making it difficult to distinguish between fingers in the foreground at times.  Given the various issues with this dataset, a masking model was used in the data cleaning stage, producing more usable and generalizable data for the classifier (for more info on this, see Data Preparation).  
## Data Preparation
There were two portions of data preparation that need to happen in order to make this project work.  First, I had to generate usable data to train my model.  Second, I had to produce a groomed video feed, meaning a live video capture whose frames were formatted in an interpretable way for the model.
#### For modeling
There are many libraries and methods out there to choose from when it comes to loading in image data. In the end, I chose Keras's ImageDataGenerator module because it has an easy interface to read in images directly from your local database, and has some great pre-processing functionality.  In terms of pre-processing inside ImageDataGenerator, the image pixel dimensions were adjusted (scaled, converted to grayscale), and the image was resized.  This was all done to reduce the size of the data coming into the model.

At this point, the model would be receiving a grayscale, resized/scaled copy of the actual train image.  The problem with this is that when the model is asked to classify this image, the algorithm is actually picking up on the pixel composition around the hand in the frame, and using that as the determining factor in the classification prediction as opposed to the pixils within or outlining the hand.  This is an issue because when the model is asked to predict on images outside of the particular dataset that the model was trained on (i.e. a hand in front of a different styled background), the model is unsure of how to interpret the background image it is seeing, and therefor is unable to make an accurate prediction.  The solution I arrived at was to implement a second neural network before images were fed to the classifier.  This model uses a combination of convolutional and pooling layers, as well as some upsampling techniques to act as an edge-detector.  The edge-detector takes in the re-sized, grayscale image and returns a mostly black image, with light-colored lines and shading indicating edges of objects in the image.  At a high level, the algorithm detects places in the image where pixel values differ greatly, and draws a mask over these pixel difference points.  This is the image fed to the classifier now, as all low level textures, colors and identifiers of the background have been augmented to produce this very distinct collection of lines.  During initial model construction and comparison, a typical 70/30 train test split was used.
#### For predictions via OpenCV 
OpenCV works the same way as any basic webcam, by recording frames every 20th or 30th (or whatever you specify) of a second and displaying them. These frames can be fed to the model to predict on, so all that needs to happen is the frames need to be formated in a way that is compatible with the model input.  This involves a simple resizing and minor formatting of the frame, after which it is run through the edge detector to produce a mask.  It is then fed to the model for predictions and after some slightly ugly steps to transform probability output into letter labels, the final letter prediction is returned and displayed on the frame being shown to the user.

## Future Iterations and Next Steps
- Although masking of input data for their edges removes a lot of contextual info (color, textures, unimportant background shapes), outside data still needs to be brought in for the model to be truly generalizable for new data.  As it stands, there are still distinguishing features exclusive to the training data that makes it difficult for the classify on certain outside data.  An important next step is going to be to incorporate some more outside data, or if possible generate it myself.
- Implementing OpenCV functionality in web-app deployment will allow for real-time predictions as opposed to call-->response.  This type of interaction with the model is much more useful than uploading a single frame and having the model predict on it.  For this to work *well*, a more robust object detection network will need to be added to the OpenCV script in order to better track hands and groom backgrounds for clean model input.
- Currently the project is novel, and serves no real-world business value as it is only able to interpret ASL letters shown to it. Next steps after further algorithm optimization would include training the model on a catalogue of basic words.  The goal of this would be to later attach some sentiment analysis to the system and enable a sort of automated response from the algorithm (think low-level chat bot).

## Sources
Final edge detection architecture trained on BSD500 contour dataset - [article here](http://www.kubacieslik.com/developing-deep-learning-edge-detector-solve-toy-problem/)
Keras modeling, [info here](https://keras.io/why-use-keras/)