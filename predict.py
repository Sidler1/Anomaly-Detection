import tensorflow as tf
import datetime
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)
ad_model = tf.keras.models.load_model("models/ai.ckpt/")


def predict(predict_data, model) -> pd.DataFrame():
    pred = model.predict(np.array(predict_data))
    pred = pd.DataFrame(pred, columns=predict_data.columns)
    pred.index = predict_data.index
    scored = pd.DataFrame(index=predict_data.index)
    scored['Error_Rate'] = np.mean(np.abs(pred - predict_data), axis=1)
    scored['Threshold'] = 1
    scored['Anomaly'] = scored['Threshold'] < scored['Error_Rate']
    return scored


@app.route('/', methods=['POST'])
def index():
    pd.read_csv(request.files['csv']).to_csv("data/" + str(datetime.date.today()) + ".csv", index=False)
    resp = predict(pd.read_csv("data/" + str(datetime.date.today()) + ".csv"), ad_model)
    resp = resp.drop(['Threshold'], 1)
    return jsonify(resp.to_dict(orient="index"))


if __name__ == '__main__':
    app.run()
