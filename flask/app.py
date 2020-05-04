import tensorflow as tf
import keras
from keras import Sequential, Model
from tensorflow.python.keras.backend import set_session
from imageio import imread, imsave
from waitress import serve
from src.utils import *
from flask import Flask, send_from_directory, render_template, request, abort
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
app = Flask(__name__, static_url_path="/static")
tf.keras.backend.clear_session()
'''
to run flask app in development mode from terminal:
export FLASK_ENV=development
to init deploy:
env FLASK_APP=app.py flask run
'''

# letter labels from val_generator
key_dict = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4,
            'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9,
            'K': 10, 'L': 11, 'M': 12, 'N': 13, 'O': 14,
            'P': 15, 'Q': 16, 'R': 17, 'S': 18, 'T': 19,
            'U': 20, 'V': 21, 'W': 22, 'X': 23, 'Y': 24,
            'Z': 25}

unet = load_saved_model('unet1.keras')
model = load_saved_model('model19.h5')
graph = tf.get_default_graph()
sess = keras.backend.get_session()
init = tf.global_variables_initializer()
sess.run(init)


@app.route("/")
def index():
    """Return the main page."""
    return send_from_directory("static", "index.html")


@app.route("/get_results", methods=["POST"])
def get_results():
    """ Predict the letter signed in input image """
    # access form submission
    data = request.form
    # load model
    # grab image and preprocess for model
    filename = data['image_name']
    true_label = data['image_true_label']
    global graph
    global sess
    with graph.as_default():
        set_session(sess)
        image = predict_custom_image(
            unet, imread(f'static/test_images/{filename}'))
        # classify image and attach letter label
        prediction_index = predict_on(model, image)
        letter_predict = list(key_dict.keys())[list(
            key_dict.values()).index(prediction_index)]
        # print to results.html
        return render_template("results.html", predicted_class=letter_predict, file_name=filename, true_label=true_label)


if __name__ == "__main__":
    serve(app, host='0.0.0.0', port=5000)
