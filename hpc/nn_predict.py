import tensorflow as tf
import pandas as pd
import numpy as np
import os
import keras
import sys
import warnings

from skimage.io import imread
from skimage.transform import resize

from PIL import Image

from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Dense, Flatten, Embedding, Input, Dropout, Concatenate, BatchNormalization, CuDNNGRU
from keras.models import Model
from keras.optimizers import Adam, SGD, RMSprop
from keras import backend as K
from keras.callbacks import EarlyStopping
from keras.models import load_model
from keras.utils import Sequence
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split

from scipy import sparse

warnings.filterwarnings("ignore")

def trim_dataframe(df):
    trimmed_df = df.join(pd.DataFrame({"user_active_ads" : df.groupby("user_id").size()}), on="user_id")
    
    activation_dates = pd.to_datetime(trimmed_df.activation_date).sort_values().values
    last_activation_date = activation_dates[-1]
    activation_range_length = activation_dates[-1] - activation_dates[1]
    trimmed_df = trimmed_df.assign(days_since_activation = last_activation_date - pd.to_datetime(trimmed_df.activation_date))
    
    features_to_drop = ["item_id", "city", "param_2", "param_3", "item_seq_number", "image_top_1", "user_id", "activation_date"]
    trimmed_df = trimmed_df.drop(features_to_drop, axis=1)
    return trimmed_df

def change_data_representation(df, feature_to_index_maps, data_folder, image_folder):
    
    empty_img_path = os.path.join(data_folder, "external", "empty_img.jpg")
    
    empty_img = Image.fromarray(np.zeros((224, 224, 3)).astype(np.uint8))
    empty_img_path = os.path.join(data_folder, "external", "empty_img.jpg")
    empty_img.save(empty_img_path)
    
    empty_img_relpath = os.path.relpath(empty_img_path, image_folder)
    
    num_df = df.assign(category_index = df.category_name.map(feature_to_index_maps[0]),
                       region_index = df.region.map(feature_to_index_maps[1]),
                       parent_category_index = df.parent_category_name.map(feature_to_index_maps[2]),
                       param_1_index = df.param_1.map(feature_to_index_maps[3]),
                       user_type_index = df.user_type.map(feature_to_index_maps[4]),
                       image_path = df.image + ".jpg",
                       days_since_activation_num = df.days_since_activation.astype(int)
                      )
    
    num_df.fillna({"param_1_index" : 0,
                   "region_index" : 0,
                   "parent_category_index" : 0,
                   "user_type_index" : 0,
                   "category_index" : 0,
                   "description" : "",
                  "image_path" : empty_img_relpath},
                 inplace=True)
    
    num_df.price.fillna(num_df.groupby("category_name")["price"].transform("mean"),
                        inplace=True)
    
    corrupt_df = pd.read_csv(os.path.join(data_folder, "external", "corrupt_files.csv"))
    corrupt_files = corrupt_df.corrupt_path.values
    
    num_df.replace(to_replace={"image_path" : corrupt_files},
     value={"image_path" : [empty_img_relpath]*len(corrupt_files)},
     inplace=True)



    features_to_keep = ["title", "description", "price", "deal_probability", "user_active_ads",\
                        "image_path", "category_index", "region_index", "parent_category_index",\
                       "param_1_index", "user_type_index", "days_since_activation_num"]

    num_df = num_df[features_to_keep]
    
    return num_df

def get_feature_to_index_maps(df):
    category_index_mapping = {category : int(index + 1) for index, category\
                              in enumerate(df.category_name.unique())}
    region_index_mapping = {region : int(index + 1) for index, region\
                            in enumerate(df.region.unique())}
    parent_category_index_mapping = {par_category : int(index + 1) for index, par_category\
                                     in enumerate(df.parent_category_name.unique())}
    param_1_index_mapping = {param_1 : int(index + 1) for index, param_1\
                             in enumerate(df.param_1.unique())}
    
    user_type_index_mapping = dict(zip(df.user_type.unique(), np.arange(1,4)))
    
    return [category_index_mapping, region_index_mapping, parent_category_index_mapping,\
            param_1_index_mapping, user_type_index_mapping]

def create_one_hot_encoder(df):
    one_hot_enc = OneHotEncoder(handle_unknown="ignore")
    categorical_columns = ["category_index", "region_index", "parent_category_index",\
                          "param_1_index", "user_type_index"]
    one_hot_enc.fit(df[categorical_columns])
    return one_hot_enc

def one_hot_encode(df, enc):
    categorical_columns = ["category_index", "region_index", "parent_category_index",\
                          "param_1_index", "user_type_index"]
    return enc.transform(df[categorical_columns])

def keras_rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


class TestDataGenerator(Sequence):
    def __init__(self, X, text, desc, image_df, batch_size, image_folder, ids):
        self.X = X
        self.text = text
        self.desc = desc
        self.image_paths = image_df.image_path.values
        self.y = image_df.deal_probability.values
        self.batch_size = batch_size
        self.image_folder = image_folder
        self.ids = ids
    
    def __len__(self):
        return int(np.ceil(self.X.shape[0] / self.batch_size))
    
    def __getitem__(self, idx):
        batch_X = self.X[idx * self.batch_size:(idx + 1) * self.batch_size, :]
        batch_text = self.text[idx * self.batch_size:(idx + 1) * self.batch_size, :]
        batch_desc = self.desc[idx * self.batch_size:(idx + 1) * self.batch_size, :]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_path = self.image_paths[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_ids = self.ids[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        batch_image = np.array([
            resize(
                imread(
                    os.path.join(
                        self.image_folder,
                        file_name)
                ),
                (224, 224)) 
            for file_name in batch_path])
    
        return [batch_X.toarray(), batch_text, batch_desc, batch_image], batch_y, batch_ids

with open(os.path.join(os.pardir, "train_path.txt"), 'r') as f:
    train_path = f.readline().rstrip("\n")
    f.close()

with open(os.path.join(os.pardir, "train_image_path.txt"), 'r') as f:
    train_image_folder_path = f.readline().rstrip("\n")
    f.close()

with open(os.path.join(os.pardir, "test_path.txt"), 'r') as f:
    test_path = f.readline().rstrip("\n")
    f.close()

with open(os.path.join(os.pardir, "test_image_path.txt"), 'r') as f:
    test_image_folder_path = f.readline().rstrip("\n")
    f.close()

data_folder = os.path.abspath(os.path.join(os.pardir, "data"))

raw_df = trim_dataframe(pd.read_csv(train_path))

raw_train_df, _ = train_test_split(raw_df, test_size = 0.2, random_state=1337)

test_csv = pd.read_csv(test_path)
test_csv["deal_probability"] = np.random.uniform()
test_ids = test_csv.item_id.values
raw_test_df = trim_dataframe(test_csv)

feature_to_index_maps = get_feature_to_index_maps(raw_train_df)

train_df = change_data_representation(raw_train_df, feature_to_index_maps, data_folder, train_image_folder_path)
test_df = change_data_representation(raw_test_df, feature_to_index_maps, data_folder, test_image_folder_path)

enc = create_one_hot_encoder(train_df)
numerical_features = ["price", "user_active_ads", "days_since_activation_num"]

scaler = Normalizer()

train_cat_X = one_hot_encode(train_df, enc)
train_num_X = train_df[numerical_features].values
scaler.fit(train_num_X)
train_num_X = scaler.transform(train_num_X)

test_cat_X = one_hot_encode(test_df, enc)
test_num_X = test_df[numerical_features]
test_num_X = scaler.transform(test_num_X)

train_text_df = train_df.assign(td = train_df.title + train_df.description)
vocabulary = train_text_df.td.values
tokenizer = Tokenizer(num_words=2500)
tokenizer.fit_on_texts(vocabulary)

train_title = tokenizer.texts_to_sequences(train_df["title"].values)
test_title = tokenizer.texts_to_sequences(test_df["title"].values)

train_desc = tokenizer.texts_to_sequences(train_df["description"].values)
test_desc = tokenizer.texts_to_sequences(test_df["description"].values)

maximum_title = max(list(map(lambda x: len(x), train_title)))
maximum_desc = max(list(map(lambda x: len(x), train_desc)))

test_title_array = pad_sequences(test_title, maxlen=maximum_title)

test_desc_array = pad_sequences(test_title, maxlen=maximum_desc)

test_X = sparse.hstack((test_cat_X, test_num_X)).tocsr()

test_generator = TestDataGenerator(test_X, test_title_array, test_desc_array, test_df[["image_path", "deal_probability"]], 32, test_image_folder_path, test_ids)

model = load_model("trained_model.h5")

prediction_list = []
id_list = []
for index in range(test_generator.__len__()):
    X, y, ids = test_generator.__getitem__(index)
    prediction_list.append(model.predict_on_batch((X)))
    id_list.append(ids)

predictions = np.concatenate(prediction_list)
ids = np.concatenate(id_list)

predictions_df = pd.DataFrame(data={"item_id" : ids, "deal_probability" : predictions})
predictions.to_csv("predictions.csv")






