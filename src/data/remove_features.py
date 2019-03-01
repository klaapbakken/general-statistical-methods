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

data_folder = os.path.join(os.path.abspath(os.curdir), os.pardir, os.pardir, "data")

training_data = os.path.join(data_folder, "interim", "train_train.csv")
validation_data = os.path.join(data_folder, "interim", "train_validation.csv")

training_df = pd.read_csv(training_data)
validation_df = pd.read_csv(validation_data)

trimmed_training_df = trim_dataframe(training_df)
trimmed_validation_df = trim_dataframe(validation_df)

trimmed_training_df.to_csv(os.path.join(data_folder, "interim", "trimmed_train_train.csv"))
trimmed_validation_df.to_csv(os.path.join(data_folder, "interim", "trimmed_train_validation.csv"))


