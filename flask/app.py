from waitress import serve
from src.utils import *
from flask import Flask, send_from_directory, render_template, request, abort
app = Flask(__name__, static_url_path="/static")
# from src.models.wine_predictor import predict_wine

'''
to run flask app in development mode from terminal
export FLASK_ENV=development
env FLASK_APP=app.py flask run
'''

key_dict = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4,
            'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9,
            'K': 10, 'L': 11, 'M': 12, 'N': 13, 'O': 14,
            'P': 15, 'Q': 16, 'R': 17, 'S': 18, 'T': 19,
            'U': 20, 'V': 21, 'W': 22, 'X': 23, 'Y': 24,
            'Z': 25}


@app.route("/")
def index():
    """Return the main page."""
    return send_from_directory("static", "index.html")


@app.route("/get_results", methods=["POST"])
def get_results():
    """ Predict the class of wine based on the inputs. """
    data = request.form
    # print(data)
    model = load_saved_model('copy_model3.h5')
    filename = data['image_name']
    true_label = data['image_true_label']
    image = preprocess_for_predict(filename)
    prediction_index = predict_on(model, image)
    letter_predict = list(key_dict.keys())[list(
        key_dict.values()).index(prediction_index)]
    return render_template("results.html", predicted_class=letter_predict, file_name=filename, true_label=true_label)


if __name__ == "__main__":
    serve(app, host='0.0.0.0', port=5000)
