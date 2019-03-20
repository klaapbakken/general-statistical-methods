import os

import pandas as pd

def trim_dataframe(df):
    trimmed_df = df.join(pd.DataFrame({"user_active_ads" : df.groupby("user_id").size()}), on="user_id")
    
    activation_dates = pd.to_datetime(trimmed_df.activation_date).sort_values().values
    last_activation_date = activation_dates[-1]
    activation_range_length = activation_dates[-1] - activation_dates[1]
    trimmed_df = trimmed_df.assign(days_since_activation = last_activation_date - pd.to_datetime(trimmed_df.activation_date))
    
    features_to_drop = ["Unnamed: 0", "item_id", "city", "param_2", "param_3", "item_seq_number", "image_top_1", "user_id", "activation_date"]
    trimmed_df = trimmed_df.drop(features_to_drop, axis=1)
    return trimmed_df

with open(os.path.join(os.pardir, "train_path.txt"), 'r') as f:
    train_path = f.readline().rstrip("\n")
    f.close()

with open(os.path.join(os.pardir, "train_image_path.txt"), 'r') as f:
    train_image_folder_path = f.readline().rstrip("\n")
    f.close()

train_df = trim_dataframe(pd.read_csv(train_path))

print(train_df.head())


