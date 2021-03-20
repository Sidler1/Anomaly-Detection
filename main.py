import os
import argparse

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import Input
from tensorflow.python.keras.layers import Dense, GaussianNoise
from tensorflow.python.keras.models import Model

MODELS_CKPT_DIR = os.path.join("models/ai.ckpt/")


def build_model(train_set: pd.DataFrame, lr: float) -> Model:
    act_func = 'elu'
    encoding_dim = 20
    features = train_set.shape[1]
    inp = Input(shape=(features,))
    x = GaussianNoise(stddev=0.3)(inp)
    x = Dense(encoding_dim * 2, activation=act_func)(x)
    out = Dense(encoding_dim * 2, activation=act_func)(x)
    out = Dense(features, activation=act_func)(x)
    model = Model(inputs=[inp], outputs=[out])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss="mae")
    model.summary()
    return model


def getdata(filename: str, columns):
    merged_data = pd.DataFrame()
    msg = pd.read_csv(filename)
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


def parse_arguments():
    parser = argparse.ArgumentParser(description='A Small Anomaly Detection Program')
    parser.add_argument("--save_freq", help="Define a Save Interval", default=500, type=int)
    parser.add_argument("--epochs", help="Define the Epochs to run", default=5000, type=int)
    parser.add_argument("--batch_size", help="Define the Batch-Size", default=32, type=int)
    parser.add_argument("--train_file", help="Define the train file location", default="data/train.csv", type=str)
    parser.add_argument("--verb_train", help="Verbosity for Training", default=0, type=int)
    parser.add_argument("--verb_save", help="Verbosity for Saving", default=0, type=int)
    parser.add_argument("--lr", help="Learning Rate", default=0.001, type=float)
    args = parser.parse_args()
    return args.save_freq, args.epochs, args.batch_size, args.train_file, args.verb_train, args.verb_save, args.lr


if __name__ == '__main__':
    save_freq, epochs, batch_size, train_file, verb_train, verb_save, lr = parse_arguments()
    tf_callback = tf.keras.callbacks.ModelCheckpoint(MODELS_CKPT_DIR, save_freq=save_freq, monitor="loss",
                                                     save_best_only=True, verbose=verb_save)
    # tb_callback = tf.keras.callbacks.TensorBoard(log_dir="models/logs", histogram_freq=10, write_graph=True,
    #                                             write_images=True, update_freq="epoch", embeddings_freq=10)
    X_train, X_test = getdata(train_file, ["110", "115", "120"])
    ad_model = build_model(X_train, lr)
    if os.path.exists(MODELS_CKPT_DIR):
        ad_model = tf.keras.models.load_model(MODELS_CKPT_DIR)
    ad_model.fit(np.array(X_train), np.array(X_train), batch_size=batch_size, verbose=verb_train, epochs=epochs,
                 validation_split=0.1,
                 callbacks=[tf_callback])
    handle_model_callback(ad_model)
