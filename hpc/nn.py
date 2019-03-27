import tensorflow as tf
import pandas as pd
import numpy as np
import os
import keras
import sys

from skimage.io import imread
from skimage.transform import resize

import warnings

from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Dense, Flatten, Embedding, Input, Dropout, Concatenate, BatchNormalization, CuDNNGRU
from keras.models import Model
from keras.optimizers import Adam, SGD, RMSprop
from keras import backend as K
from keras.callbacks import EarlyStopping
from keras.models import load_model
from keras.utils import Sequence

from scipy import sparse

warnings.filterwarnings("ignore")

class DataGenerator(Sequence):
    def __init__(self, X, text, desc, image_df, batch_size, image_folder):
        self.X = X
        self.text = text
        self.desc = desc
        self.image_paths = image_df.image_path.values
        self.y = image_df.deal_probability.values
        self.batch_size = batch_size
        self.image_folder = image_folder
    
    def __len__(self):
        return int(np.ceil(self.X.shape[0] / self.batch_size))
    
    def __getitem__(self, idx):
        batch_X = self.X[idx * self.batch_size:(idx + 1) * self.batch_size, :]
        batch_text = self.text[idx * self.batch_size:(idx + 1) * self.batch_size, :]
        batch_desc = self.desc[idx * self.batch_size:(idx + 1) * self.batch_size, :]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_path = self.image_paths[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        batch_image = np.array([
            resize(
                imread(
                    os.path.join(
                        self.image_folder,
                        file_name)
                ),
                (224, 224)) 
            for file_name in batch_path])
    
        return [batch_X.toarray(), batch_text, batch_desc, batch_image], batch_y

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

train_generator = DataGenerator(train_X, train_title, train_desc, train_df[["image_path", "deal_probability"]], 32, image_folder)
val_generator = DataGenerator(val_X, val_title, val_desc, val_df[["image_path", "deal_probability"]], 32, image_folder)

#Neural network

dense_input = Input(shape=(train_X.shape[1], ))
dense_output = Dense(16, activation="relu")(dense_input)
dense_output = BatchNormalization()(dense_output)
dense_output = Dense(8, activation="relu")(dense_output)
dense_output = BatchNormalization()(dense_output)

title_input = Input(shape=(train_title.shape[1], ))
title_embedding_layer = Embedding(10000, 50, input_length=train_title.shape[1])(title_input)
title_rnn_output = CuDNNGRU(16)(title_embedding_layer)
title_rnn_output = BatchNormalization()(title_rnn_output)
title_rnn_output = Dense(8, activation="relu")(title_rnn_output)

desc_input = Input(shape=(train_desc.shape[1], ))
desc_embedding_layer = Embedding(10000, 50, input_length=train_desc.shape[1])(desc_input)
desc_rnn_output = CuDNNGRU(16)(desc_embedding_layer)
desc_rnn_output = BatchNormalization()(desc_rnn_output)
desc_rnn_output = Dense(8, activation="relu")(desc_rnn_output)

image_model = InceptionV3(input_shape=(224, 224, 3), include_top=False)
for layer in image_model.layers:
    layer.trainable = False

image_input = image_model.input
image_output = Flatten()(image_model.output)
image_output = Dense(16, activation="relu")(image_output)

output = Concatenate()([dense_output, title_rnn_output, desc_rnn_output, image_output])
output = Dropout(0.5)(output)
output = Dense(32, activation="relu")(output)
output = Dropout(0.5)(output)
output = BatchNormalization()(output)
output = Dense(16, activation="relu")(output)
output = Dropout(0.5)(output)
output = BatchNormalization()(output)
output = Dense(8, activation="relu")(output)
output = BatchNormalization()(output)
output = Dense(1, activation="sigmoid")(output)

model = Model([dense_input, title_input, desc_input, image_input], output)
model.compile(optimizer="Adam", loss=keras_rmse, metrics=[keras_rmse, "mean_squared_error"])

model.summary()
history = model.fit_generator(
    train_generator,
    validation_data = val_generator,
    callbacks = [EarlyStopping(), ModelCheckpoint("checkpoint_model.h5", monitor="val_loss", save_best_only=True, mode="min")],
    epochs = 10,
    workers = 8
)

model.save("trained_model.h5")
