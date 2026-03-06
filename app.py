from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Load trained model
model = load_model("car.h5")

classes = ['01-minor', '02-moderate', '03-severe']


def predict_damage(img_path):

    img = image.load_img(img_path, target_size=(224,224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    pred = model.predict(x)
    predicted_class = classes[np.argmax(pred)]

    return predicted_class


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():

    if 'file' not in request.files:
        return "No file uploaded"

    file = request.files['file']
    filepath = os.path.join("static", file.filename)
    file.save(filepath)

    result = predict_damage(filepath)

    return render_template("index.html", prediction=result, img_path=filepath)


if __name__ == "__main__":
    app.run(debug=True)