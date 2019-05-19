import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils import shuffle
from scoring import score_prediction
from feature_engineering import extract_train_features
from feature_engineering import simple_imputation
from sklearn.naive_bayes import GaussianNB
from feature_engineering import *
import random

# import finalized training data csv
train = pd.read_csv("final_training_data.csv")
# add target
extract_train_features(train, target="score_rank", max_rank=10)

def impute_na(train):
    na_ids = train.isna().any().values
    feature_list = train.columns[na_ids.nonzero()[0]]
    for feature in feature_list:
        missing_ids = train.loc[train[feature].isna(), :].index.values
        imp =  simple_imputation(train, feature, type="median")
        train.loc[missing_ids, feature] = imp

    return train

train = impute_na(train)

def split_train_test_simple(data, split=4):
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

    train, test = split_train_test(data)
    X_train = train.drop(columns=["target", "booking_bool", "click_bool", "position"])
    y_train = train["target"]
    X_test = test.drop(columns=["target", "booking_bool", "click_bool", "position"])
    y_test = test.loc[:, ["srch_id", "prop_id", "booking_bool", "click_bool"]]

    # train_down, number_books, number_clicks, id_list = oversample(train, max_rank=5)
    print("length train data", len(train))
    print("length valid data", len(test))

    random.seed(42)
    rf = RandomForestRegressor(n_estimators=100)

    rf.fit(X_train, y_train)
    prediction_rf = rf.predict(X_test)

    return y_test, prediction_rf


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


def random_split_train_test(data, split=0.75):

    srch_ids = shuffle(data["srch_id"].values)
    bound = int(len(srch_ids)*split)

    train_ids = srch_ids[:bound]
    test_ids = srch_ids[bound:]

    train = data.loc[data["srch_id"].isin(train_ids)]
    test = data.loc[data["srch_id"].isin(test_ids)]
    return train,test

def cross_validate(estimator, data, k_folds=3, split=4, to_print=False):
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
        X_train = train.drop(columns=["target", "booking_bool", "click_bool", "position"])
        y_train = train["target"]
        X_test = test.drop(columns=["target", "booking_bool", "click_bool", "position"])
        y_test = test.loc[:, ["srch_id", "prop_id", "booking_bool", "click_bool"]]

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
    targets = ["book", "book_click", "score", "score_rank"]
    n_estimators = [50, 100, 250]
    max_ranks = [5, 10]

    for target in targets:
        for n_estimator in n_estimators:
            for max_rank in max_ranks:

                print(f"\nCURRENT CONFIGURATION")
                print("########################################################################")
                print(f"Target = {target}")
                print(f"N_trees = {n_estimator}")
                print(f"Max_rank = {max_rank}")
                print("########################################################################")

                data = pd.read_csv("C:/Users/Frede/Dropbox/Master/DM/Assignments/2/DM2/final_training_data.csv")
                impute_na(data)
                extract_train_features(data=data, target=target,max_rank=max_rank)
                estimator = RandomForestRegressor(n_estimators=n_estimator)
                cross_validate(estimator=estimator, data=data, k_folds=1, to_print=True)


    """
    # import finalized training data csv
    train = pd.read_csv("final_training_data.csv")
    # add target
    extract_train_features(train, target="book", max_rank=10)

    """
