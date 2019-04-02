import tensorflow as tf
import pandas as pd
import numpy as np
import os
import keras
import sys

from keras.preprocessing.image import ImageDataGenerator
from keras.applications.mobilenet import MobileNet
from keras.layers import Dense, Flatten, Embedding, Input, Dropout, Concatenate, BatchNormalization, CuDNNGRU
from keras.models import Model
from keras.optimizers import Adam, SGD, RMSprop
from keras import backend as K
from keras.callbacks import EarlyStopping

from scipy import sparse

def keras_rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

data_folder = os.path.abspath(os.path.join(os.pardir, "data"))

train_X = sparse.load_npz(os.path.join(data_folder, "processed", "train_X.npz")).toarray()
val_X = sparse.load_npz(os.path.join(data_folder, "processed", "val_X.npz")).toarray()

train_df = pd.read_csv(os.path.join(data_folder, "processed", "train_image_df.csv"))
val_df = pd.read_csv(os.path.join(data_folder, "processed", "val_image_df.csv"))

train_y = train_df.deal_probability.values
val_y = val_df.deal_probability.values

inputs = Input(shape=(train_X.shape[1], ))
outputs = Dense(64, activation="relu")(inputs)
outputs = BatchNormalization()(outputs)
outputs = Dense(32, activation="relu")(outputs)
outputs = Dropout(0.3)(outputs)
outputs = BatchNormalization()(outputs)
outputs = Dense(32, activation="relu")(outputs)
outputs = BatchNormalization()(outputs)
outputs = Dense(16, activation="relu")(outputs)
outputs = BatchNormalization()(outputs)
outputs = Dense(8, activation="relu")(outputs)
outputs = BatchNormalization()(outputs)
outputs = Dense(1, activation="sigmoid")(outputs)

model = Model(inputs, outputs)
model.compile(optimizer="Adam", loss=keras_rmse, metrics=[keras_rmse])

model.fit(x=train_X,
 y=train_y,
 validation_data=(val_X, val_y),
 epochs=10, 
 batch_size=512)
