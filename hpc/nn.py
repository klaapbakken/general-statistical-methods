import tensorflow as tf
import pandas as pd
import numpy as np
import os
import keras

from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16, decode_predictions
from keras.layers import Dense, Flatten, Embedding, Input, Dropout, Concatenate, BatchNormalization, CuDNNGRU
from keras.models import Model
from keras.optimizers import Adam, SGD, RMSprop
from keras import backend as K
from keras.callbacks import EarlyStopping

from scipy import sparse

def data_generator(image_generator, X, X_title, X_desc):
    index_generator = image_generator.index_generator
    for indices, images in zip(index_generator, image_generator):
        yield (images[0], X[indices, :], X_title[indices, :], X_desc[indices, :]), images[1]

def keras_rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


with open(os.path.join(os.pardir, "train_image_path.txt"), 'r') as f:
    image_folder = f.readline().rstrip("\n")
    f.close()

data_folder = os.path.abspath(os.path.join(os.pardir, "data"))

def data_generator(image_generator, X, X_title, X_desc):
    index_generator = image_generator.index_generator
    for indices, images in zip(index_generator, image_generator):
        yield (images[0], X[indices, :], X_title[indices, :], X_desc[indices, :]), images[1]

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
    target_size=(224, 224),
    color_mode='rgb',
    batch_size=32)

val_image_generator = image_generator.flow_from_dataframe(
    dataframe=val_df,
    directory=image_folder,
    x_col="image_path",
    y_col="deal_probability",
    class_mode='other',
    target_size=(224, 224),
    color_mode='rgb',
    batch_size=32)

train_generator = data_generator(train_image_generator, train_X, train_title, train_desc)
val_generator = data_generator(val_image_generator, val_X, val_title, val_desc)

#Neural network

inputs = Input(shape=(train_X.shape[1], ))
outputs = Dropout(0.2)(inputs)
outputs = Dense(1024, activation="relu")(outputs)
outputs = BatchNormalization()(outputs)
outputs = Dropout(0.3)(outputs)
outputs = Dense(256, activation="relu")(outputs)
outputs = Dropout(0.3)(outputs)
outputs = BatchNormalization()(outputs)
outputs = Dense(128, activation="relu")(outputs)
outputs = BatchNormalization()(outputs)
outputs = Dense(128, activation="relu")(outputs)
outputs = BatchNormalization()(outputs)
outputs = Dense(64, activation="relu")(outputs)
outputs = BatchNormalization()(outputs)
outputs = Dense(32, activation="relu")(outputs)
outputs = BatchNormalization()(outputs)
outputs = Dense(16, activation="relu")(outputs)
outputs = BatchNormalization()(outputs)
outputs = Dense(1, activation="sigmoid")(outputs)

model = Model(inputs, outputs)
model.compile(optimizer="Adam", loss=keras_rmse, metrics=[keras_rmse, "mean_squared_error"])
history = model.fit(train_X, train_df.deal_probability, batch_size=1024, epochs=20,
                    validation_data=(val_df.deal_probability, val_y),
                   callbacks=[EarlyStopping()])


