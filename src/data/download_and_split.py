import os
import zipfile

import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split

raw_data_folder = os.path.join(os.path.abspath(os.curdir), os.pardir, os.pardir, "data", "raw")

os.system("kaggle competitions download -f train.csv -p " + raw_data_folder + " avito-demand-prediction")

with zipfile.ZipFile(os.path.join(raw_data_folder, "train.csv.zip"), "r") as zip_ref:
    zip_ref.extractall(raw_data_folder)
    
os.remove(os.path.join(raw_data_folder, "train.csv.zip"))

data_folder = os.path.join(raw_data_folder, os.pardir)
training_data = os.path.join(data_folder, "raw", "train.csv")

df = pd.read_csv(training_data)

train_df, validation_df = train_test_split(df, test_size = 0.2, random_state=1337)

train_df.to_csv(os.path.join(data_folder, "interim", "train.csv"))
validation_df.to_csv(os.path.join(data_folder, "interim", "validation.csv"))
