import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils import shuffle
from scoring import score_prediction
from feature_engineering import simple_imputation
from feature_engineering import extract_train_features


def impute_na(train):
    na_ids = train.isna().any().values
    feature_list = train.columns[na_ids.nonzero()[0]]
    for feature in feature_list:
        missing_ids = train.loc[train[feature].isna(), :].index.values
        imp = simple_imputation(train, feature, type="median")
        train.loc[missing_ids, feature] = imp


def split_train_test_(data, split=4):
    """
    use random forest regression to predict propability of being booked for each row
    :param X_train: x_train
    :param y_test: y_train
    :param n_estimators: passed to RandomForestRegressor
    :return: vector with booking probability for each row
    """
    # not necessary anymore
    #data = data.loc[:, data.columns != "date_time"]
    print("length original data", len(data))

    training, valid = split_train_test(data)


    # train_down, number_books, number_clicks, id_list = oversample(train, max_rank=5)
    print("length new data", len(train_down))

    X_train = training.loc[:, training.columns != "target"]
    y_train = training.loc[:, training.columns == "target"]

    X_valid = valid.loc[:, valid.columns != "target"]
    y_valid= valid.loc[:, valid.columns == "target"]

    rf = RandomForestRegressor(n_estimators=100)

    rf.fit(X_train, y_train["target"])
    prediction_rf = rf.predict(X_valid)

    return y_valid, prediction_rf


def prediction_to_submission(prediction, y_test):

    y_test["prediction"] = prediction
    y_test_sorted = y_test.sort_values(["srch_id", "prediction"], ascending=[True, False])
    return y_test_sorted[["srch_id", "prop_id"]]


def split_train_test(data, split=4):
    """

    split data into train (split-1/split) and test (1/split) data
    :param data: pandas df
    :param split: (int) determines proportional size of train/test
    :return: two pandas dfs: train and test
    """

    test = shuffle(data.loc[data["srch_id"] % split == 0])
    train = shuffle(data.loc[data["srch_id"] % split != 0])

    return train, test


def cross_validate(estimator, data, target, k_folds=3, split=4, to_print=False):
    """
    cross-validate over k-folds
    :param estimator: sklearn estimator instance
    :param data: whole dataset
    :param target: df column serving as target
    :param k_folds: number of folds to perform
    :param split: split param for split_train_test funct
    :param to_print: if True scores are printed and not returned
    :return: if to_print=False function returns list with scores
    """

    scores = []
    for i in range(k_folds):
        print(f"Fold {i+1} running...")

        train, test = split_train_test(data, split)

        # downsample training data
        print(f"Downsampling training data...\n")
        extract_train_features(data=train, target="book", max_rank=5)

        X_train = train.drop(columns=["booking_bool", "click_bool", "position"])
        y_train = train[target]
        X_test = test.drop(columns=["booking_bool", "click_bool", "position"])
        y_test = test

        # fit model
        print(f"Fitting model...")
        estimator.fit(X_train, y_train)

        # predict
        print(f"Generating predictions...")
        prediction = estimator.predict(X_test)

        # score
        score = score_prediction(prediction, y_test, to_print=False)
        scores.append(score)
        print(f"Fold {i+1} finished!")

    if to_print:
            print(f"Prediction scores for {k_folds} folds are:\n {scores}")
    else:
        return scores


if __name__ == "__main__":
    target = "booking_bool"
    data = pd.read_csv("C:/Users/Frede/Dropbox/Master/DM/Assignments/2/DM2/final_training_data.csv")
    impute_na(data)
    estimator = RandomForestRegressor(n_estimators=100)
    cross_validate(estimator=estimator, data=data, target=target, k_folds=1, to_print=True)

    """
    # import finalized training data csv
    train = pd.read_csv("final_training_data.csv")
    # add target
    extract_train_features(train, target="book", max_rank=10)

    """
