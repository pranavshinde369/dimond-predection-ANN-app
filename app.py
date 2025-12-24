

from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib

app = Flask(__name__)

# Load model & preprocessor
model = tf.keras.models.load_model("diamond_price_model.keras")
preprocessor = joblib.load("dl_preprocessor.joblib")

DEFAULTS = {
    'depth_percent': 61.5,
    'table_percent': 57.0,
    'meas_length': 6.7,
    'meas_width': 6.7,
    'meas_depth': 4.1,
    'cut_quality': 'Very Good',
    'symmetry': 'Very Good',
    'polish': 'Very Good',
    'lab': 'GIA',
    'culet_size': 'N',
    'girdle_min': 'M',
    'girdle_max': 'M'
}

@app.route("/", methods=["GET", "POST"])
def index():
    price = None

    if request.method == "POST":
        data = {
            'carat_weight': float(request.form['carat_weight']),
            'cut': request.form['cut'],
            'color': request.form['color'],
            'clarity': request.form['clarity'],
            **DEFAULTS
        }

        df = pd.DataFrame([data])
        X = preprocessor.transform(df)
        log_price = model.predict(X)[0][0]
        price = round(np.expm1(log_price), 2)

    return render_template("index.html", price=price)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)

