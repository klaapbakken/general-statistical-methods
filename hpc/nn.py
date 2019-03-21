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

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import Normalizer
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectFromModel

from scipy import sparse


with open(os.path.join(os.pardir, "train_image_path.txt"), 'r') as f:
    image_folder = f.readline().rstrip("\n")
    f.close()

data_folder = os.path.abspath(os.path.join(os.pardir, "data"))

def data_generator(image_generator, X, X_title, X_desc):
    index_generator = image_generator.index_generator
    for indices, images in zip(index_generator, image_generator):
        yield (images[0], X[indices, :], X_title[indices, :], X_desc[indices, :]), images[1]

train_X = np.load(os.path.join(data_folder, "processed", "train_X.npy"))
val_X = np.load(os.path.join(data_folder, "processed", "val_X.npy"))

train_title = np.load(os.path.join(data_folder, "processed", "train_title.npy"))
val_title = np.load(os.path.join(data_folder, "processed", "val_title.npy"))

train_desc = np.load(os.path.join(data_folder, "processed", "train_desc.npy"))
val_desc = np.load(os.path.join(data_folder, "processed", "val_desc.npy"))

train_df = pd.read_csv(os.path.join(data_folder, "processed", "train_image_df.csv"))
val_df = pd.read_csv(os.path.join(data_folder, "processed", "val_image_df.csv"))

print(train_X.shape, val_X.shape)
print(train_title_array.shape, val_title_array.shape)
print(train_desc_array.shape, val_desc_array.shape)
print(len(train_df), len(val_df))
