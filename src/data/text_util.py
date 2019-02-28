import os
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder


def create_bow_dataset(num_rows, min_word_freq=100):
    """
    Creates a dataset using bag of words and saves it in processed.

    Args:
        min_word_freq: Minimum frequency of a word for it to be included in
                       the bag of words.
    """

    train = pd.read_csv('./data/interim/train.csv', nrows=num_rows)
    val = pd.read_csv('./data/interim/validation.csv', nrows=num_rows)

    train, val = replace_text_columns_with_bow(train, val)
    train, val = remove_useless_columns(train, val)
    train, val = encode_categorical_variables(train, val)

    # train, val = remove_nans_from_parameters(train, val)
    # train, val = remove_nans_from_rows(train, val)

    train.to_csv("./data/processed/train_with_bow.csv", index=False)
    val.to_csv("./data/processed/validation_with_bow.csv", index=False)


def replace_text_columns_with_bow(train, val, text_columns=['title', 'description']):

    for column in text_columns:
        train_text = train[column]
        val_text = train[column]

        train_bow, val_bow = text_column_to_bag_of_words(train_text, val_text)

        train = pd.concat([train, train_bow], axis=1).reset_index()
        val = pd.concat([val, val_bow], axis=1).reset_index()

    train = train.drop(columns=text_columns)
    val = val.drop(columns=text_columns)

    return train, val


def text_column_to_bag_of_words(training_data, val_data, min_word_freq=10):
    """
    Converts a column of data to bag of words

    Given a panda.Series of training and val data this function fits a bag of
    words method on the training column. It then encodes both the training and
    val column.

    Args:
        training_data: A column of text data. Used to fit bag of words.
        val_data: A column of val text data.

    Returns:
        bow_training_data: Encoded training data.
        bow_val_data: Encoded val data.

    """

    # Replace nans with spaces
    training_data.fillna(" ", inplace=True)
    val_data.fillna(" ", inplace=True)

    vec = TfidfVectorizer(ngram_range=(1, 1), min_df=min_word_freq, max_df=0.9,
                          lowercase=True, strip_accents='unicode', sublinear_tf=True)

    bow_training_data = vec.fit_transform(training_data)
    bow_val_data = vec.transform(val_data)

    bow_training_data = pd.DataFrame(
        bow_training_data.toarray(), columns=vec.get_feature_names())
    bow_val_data = pd.DataFrame(
        bow_val_data.toarray(), columns=vec.get_feature_names())

    return pd.DataFrame(bow_training_data), pd.DataFrame(bow_val_data)


def remove_useless_columns(train, val):
    """Removing columns that seem useless."""
    useless_columns = ["Unnamed: 0", "item_id", "item_seq_number"]
    train = train.drop(columns=useless_columns)
    val = val.drop(columns=useless_columns)
    return train, val


def encode_categorical_variables(train, val):
    """Encodes all categorical variables for the model"""
    # Easier to remove continuous var then listing up all categoricals.

    print("Columns: ", val.columns)

    train = pd.get_dummies(train, dummy_na=True)
    print("Duplicated stuff: ", val[val.index.duplicated()])
    val.reindex(columns=train.columns, fill_value=0)
    return train, val


def remove_nans_from_parameters(train, val):
    """Replaces Nans in the param columns with a space char."""
    replacements = {"param_1": " ", "param_2": " ", "param_3": " "}
    train = train.fillna(replacements)
    val = val.fillna(replacements)
    return train, val


def remove_nans_from_rows(train, val):
    """Removes any rows with nans in the two input dataframes."""
    train = train.dropna()
    val = val.dropna()
    return train, val


if __name__ == "__main__":
    """
    To create the bow dataset you should run "python -m src.data.text_util" from the general
    statistical methods folder.
    """
    create_bow_dataset(num_rows=1000)
