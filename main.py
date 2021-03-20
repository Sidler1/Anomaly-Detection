import os

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import Input
from tensorflow.python.keras.layers import Dense, GaussianNoise
from tensorflow.python.keras.models import Model

DATA_DIR = os.path.join("E:/PycharmProjects/VMT-AI/data/")
MODELS_CKPT_DIR = os.path.join("E:/PycharmProjects/VMT-AI/models/vmt.ckpt/")


def build_model(train_set: pd.DataFrame) -> Model:
    act_func = 'elu'
    encoding_dim = 20
    features = train_set.shape[1]
    inp = Input(shape=(features,))
    x = GaussianNoise(stddev=0.3)(inp)
    x = Dense(encoding_dim * 2, activation=act_func)(x)
    out = Dense(encoding_dim * 2, activation=act_func)(x)
    out = Dense(features, activation=act_func)(x)
    model = Model(inputs=[inp], outputs=[out])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss="mae")
    model.summary()
    return model


def getdata(filename: str, columns):
    merged_data = pd.DataFrame()
    msg = pd.read_csv(DATA_DIR + filename)
    merged_data = merged_data.append(msg)
    merged_data.columns = columns
    dataset_train = merged_data[3:]
    dataset_test = merged_data[:2]
    x_train = pd.DataFrame(dataset_train,
                           columns=dataset_train.columns,
                           index=dataset_train.index)
    x_test = pd.DataFrame(dataset_test,
                          columns=dataset_test.columns,
                          index=dataset_test.index)
    return x_train, x_test


def handle_model_callback(model: Model):
    model.save('./models/ad-model.h5')
    model.save_weights('./models/ad-model_weights.h5')


if __name__ == '__main__':
    tf_callback = tf.keras.callbacks.ModelCheckpoint(MODELS_CKPT_DIR, save_freq=500, monitor="loss",
                                                     save_best_only=True, verbose=1)
    # tb_callback = tf.keras.callbacks.TensorBoard(log_dir="models/logs", histogram_freq=10, write_graph=True,
    #                                             write_images=True, update_freq="epoch", embeddings_freq=10)
    X_train, X_test = getdata("test.csv", ["110", "115", "120"])
    vmt_model = build_model(X_train)
    if os.path.exists(MODELS_CKPT_DIR):
        vmt_model = tf.keras.models.load_model(MODELS_CKPT_DIR)
    vmt_model.fit(np.array(X_train), np.array(X_train), batch_size=32, verbose=1, epochs=2000, validation_split=0.1,
                  callbacks=[tf_callback])
    handle_model_callback(vmt_model)
