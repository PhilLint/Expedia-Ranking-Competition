import numpy as np
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils import shuffle
from sklearn import preprocessing
from scoring import score_prediction
from sklearn.naive_bayes import GaussianNB


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
        print(f"Fold {i+1} running...\n")
        # split data

        if target == "booking_bool":
            secondary = "click_bool"
        elif target == "click_bool":
            secondary = "booking_bool"

        train, test = split_train_test(data, split)

        train_cols = [col for col in train.columns if col not in [target, secondary]]

        X_train = train[train_cols]
        y_train = train[target]
        X_test = test[train_cols]
        y_test = test

        # fit model
        estimator.fit(X_train, y_train)

        # predict
        prediction = estimator.predict(X_test)

        # score
        score = score_prediction(prediction, y_test, to_print=False)
        scores.append(score)

    if to_print:
            print(f"Prediction scores for {k_folds} are:\n {scores}")
    else:
        return scores


if __name__ == "__main__":
    target = "booking_bool"
    data = pd.read_csv("C:/Users/Frede/Dropbox/Master/DM/Assignments/2/DM2/final_training_data.csv")
    data = data.sample(n=100_000)
    data = data.dropna()
    estimator = RandomForestRegressor(n_estimators=100)
    cross_validate(estimator=estimator, data=data, target=target, k_folds=3, to_print=True)
