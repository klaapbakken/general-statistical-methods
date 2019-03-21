import tensorflow as tf
import pandas as pd
import numpy as np
import os
import keras
import sys

from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Dense, Flatten, Embedding, Input, Dropout, Concatenate, BatchNormalization, CuDNNGRU
from keras.models import Model
from keras.optimizers import Adam, SGD, RMSprop
from keras import backend as K
from keras.callbacks import EarlyStopping

from scipy import sparse

def data_generator(image_generator, X, X_title, X_desc):
    index_generator = image_generator.index_generator
    for images, targets in image_generator:
        indices = next(index_generator)
        yield [X[indices, :].toarray(), X_title[indices, :], X_desc[indices, :], images], targets

def keras_rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

with open(os.path.join(os.pardir, "train_image_path.txt"), 'r') as f:
    image_folder = f.readline().rstrip("\n")
    f.close()

data_folder = os.path.abspath(os.path.join(os.pardir, "data"))

train_X = sparse.load_npz(os.path.join(data_folder, "processed", "train_X.npz"))
val_X = sparse.load_npz(os.path.join(data_folder, "processed", "val_X.npz"))

train_title = np.load(os.path.join(data_folder, "processed", "train_title.npy"))
val_title = np.load(os.path.join(data_folder, "processed", "val_title.npy"))

train_desc = np.load(os.path.join(data_folder, "processed", "train_desc.npy"))
val_desc = np.load(os.path.join(data_folder, "processed", "val_desc.npy"))

train_df = pd.read_csv(os.path.join(data_folder, "processed", "train_image_df.csv"))
val_df = pd.read_csv(os.path.join(data_folder, "processed", "val_image_df.csv"))

image_generator = ImageDataGenerator()

train_image_generator = image_generator.flow_from_dataframe(
    dataframe=train_df,
    directory=image_folder,
    x_col="image_path",
    y_col="deal_probability",
    class_mode='other',
    target_size=(299, 299),
    color_mode='rgb',
    batch_size=256)

val_image_generator = image_generator.flow_from_dataframe(
    dataframe=val_df,
    directory=image_folder,
    x_col="image_path",
    y_col="deal_probability",
    class_mode='other',
    target_size=(299, 299),
    color_mode='rgb',
    batch_size=256)

train_generator = data_generator(train_image_generator, train_X, train_title, train_desc)
val_generator = data_generator(val_image_generator, val_X, val_title, val_desc)

#Neural network

dense_input = Input(shape=(train_X.shape[1], ))
dense_output = Dropout(0.2)(dense_input)
dense_output = Dense(512, activation="relu")(dense_output)
dense_output = BatchNormalization()(dense_output)
dense_output = Dropout(0.3)(dense_output)
dense_output = Dense(256, activation="relu")(dense_output)
dense_output = Dropout(0.3)(dense_output)
dense_output = BatchNormalization()(dense_output)
dense_output = Dense(128, activation="relu")(dense_output)
dense_output = BatchNormalization()(dense_output)

title_input = Input(shape=(train_title.shape[1], ))
title_embedding_layer = Embedding(1000, 32, input_length=train_title.shape[1])(title_input)
title_rnn_output = CuDNNGRU(64)(title_embedding_layer)
title_rnn_output = BatchNormalization()(title_rnn_output)

desc_input = Input(shape=(train_desc.shape[1], ))
desc_embedding_layer = Embedding(1000, 32, input_length=train_desc.shape[1])(desc_input)
desc_rnn_output = CuDNNGRU(64)(desc_embedding_layer)
desc_rnn_output = BatchNormalization()(desc_rnn_output)

inceptionv3_model = InceptionV3(input_shape=(299, 299, 3), include_top=False)
for layer in inceptionv3_model.layers:
    layer.trainable = False

image_input = inceptionv3_model.input
image_output = Flatten()(inceptionv3_model.output)
image_output = Dense(512, activation="relu")(image_output)

output = Concatenate()([dense_output, title_rnn_output, desc_rnn_output, image_output])
output = Dense(512, activation="relu")(output)
output = BatchNormalization()(output)
output = Dense(256, activation="relu")(output)
output = BatchNormalization()(output)
output = Dense(128, activation="relu")(output)
output = BatchNormalization()(output)
output = Dense(64, activation="relu")(output)
output = BatchNormalization()(output)
output = Dense(32, activation="relu")(output)
output = BatchNormalization()(output)
output = Dense(16, activation="relu")(output)
output = BatchNormalization()(output)
output = Dense(1, activation="sigmoid")(output)

model = Model([dense_input, title_input, desc_input, image_input], output)
model.compile(optimizer="Adam", loss=keras_rmse, metrics=[keras_rmse, "mean_squared_error"])
history = model.fit_generator(
    train_generator,
    steps_per_epoch = np.ceil(train_X.shape[0] / 256),
    validation_data = val_generator,
    validation_steps = np.ceil(val_X.shape[0] / 256),
    callbacks = [EarlyStopping()],
    epochs = 10
)
