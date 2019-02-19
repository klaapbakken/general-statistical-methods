from sklearn.feature_extraction.text import TfidfVectorizer


def text_column_to_bag_of_words(training_data, test_data):
    """Converts a column of data to bag of words

    Given a panda.Series of training and test data this function fits a bag of words method on the training column. It then encodes both the training and test column.

    Args:
        training_data: A column of text data. Used to fit bag of words.
        test_data: A column of test text data.

    Returns:
        bow_training_data: Encoded training data.
        bow_test_data: Encoded test data.

    """

    min_word_freq = 50

    # Replace nans with spaces
    training_data.fillna(" ", inplace=True)
    test_data.fillna(" ", inplace=True)

    vec = TfidfVectorizer(ngram_range=(1, 1), min_df=min_word_freq, max_df=0.9,
                          lowercase=True, strip_accents='unicode', sublinear_tf=True)

    bow_training_data = vec.fit_transform(training_data)
    bow_test_data = vec.transform(test_data)

    return bow_training_data, bow_test_data
