from flask_cors import CORS
from flask import Flask, request, render_template, json, jsonify
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import layers
from tensorflow.keras import Model
import cv2
import numpy as np
import io

app = Flask(__name__)
CORS(app)

pre_trained_model = InceptionV3(
    input_shape=(150, 150, 3), include_top=False, weights=None
)

last_layer = pre_trained_model.get_layer("mixed7")
last_output = last_layer.output
# Flatten the output layer to 1 dimension
x = layers.Flatten()(last_output)
# Add a fully connected layer with 2048 hidden units and ReLU activation
x = layers.Dense(256, activation="relu")(x)
# Add a dropout rate of 0.5
x = layers.Dropout(0.5)(x)
# Add a final sigmoid layer for classification
x = layers.Dense(5, activation="softmax")(x)

model = Model(pre_trained_model.input, x)

model.compile(
    optimizer=RMSprop(lr=0.0001), loss="categorical_crossentropy", metrics=["accuracy"]
)
local_weights_file = "model/model.h5"
model.load_weights(local_weights_file)

labels = ["Drawings", "Engraving", "Iconography", "Paintings", "Sculpture"]


@app.route("/", methods=["GET"])
def main():
    return render_template("index.html")


@app.route("/api/prepare", methods=["POST"])
def prepare():
    file = request.files["file"]
    file = preprocessing(file)
    # print(model())
    res = model.predict(file)
    label = labels[np.argmax(res)]
    return json.dumps({"labels": label})


def preprocessing(file):
    in_memory_file = io.BytesIO()
    file.save(in_memory_file)
    data = np.fromstring(in_memory_file.getvalue(), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    norm_image = cv2.normalize(
        img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
    )
    res = cv2.resize(norm_image, dsize=(150, 150), interpolation=cv2.INTER_CUBIC)
    res = np.reshape(res, (1, 150, 150, 3))
    return res


if __name__ == "__main__":
    app.run()
