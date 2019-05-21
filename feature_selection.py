import numpy as np
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from model_test import impute_na, get_sample, split_train_test
from feature_engineering import extract_train_features
from scoring import score_prediction


def feature_selection(data, estimator, n_features=None):

    X_train = data.drop(columns=["target", "booking_bool", "click_bool", "position", "random_bool"])
    y_train = data["target"]
    selector = RFE(estimator=estimator, n_features_to_select=n_features)
    selector.fit(X_train, y_train)
    cols = selector.support_
    print(X_train.loc[:, cols].columns)


def decreasing_features_select(data, estimator, target):

        train, test = split_train_test(data, split=4)

        X_train = train.drop(columns=["target", "booking_bool", "click_bool", "position", "random_bool"])
        y_train = train["target"]
        X_test = test.drop(columns=["target", "booking_bool", "click_bool", "position", "random_bool"])
        y_test = test[["target", "srch_id", "prop_id", "booking_bool", "click_bool"]]

        n_features = len(X_train.columns)
        for n in range(n_features,10, -2):
            print("##################################")
            print(f"Number of features used: {n}.")
            selector = RFE(estimator=estimator, n_features_to_select=n)
            selector.fit(X_train, y_train)
            cols = selector.support_
            print("Features used: ")
            print(X_train.loc[:, cols].columns)
            reduced_train = X_train.loc[:, cols]
            estimator.fit(reduced_train, y_train)
            reduced_test = X_test.loc[:, cols]
            pred = clf_to_predict(estimator, reduced_test, target)
            score_prediction(pred, y_test, to_print=True)


def clf_to_predict(estimator, X_test, target):
    if target == "book":
        prediction = estimator.predict_proba(X_test)[:, 1]
    elif target == "score":
        predict_array = estimator.predict_proba(X_test)
        predict_array[:, 2] = predict_array[:, 2] * pred_weight
        prediction = predict_array[:, [1, 2]].sum(axis=1)
    else:
        print("ERROR. no using classification with score_rank!")
        return

    return prediction


if __name__ == "main":
    pd.options.mode.chained_assignment = None
    data = pd.read_csv("C:/Users/Frede/Dropbox/Master/DM/Assignments/2/DM2/final_training_fixed_data.csv")
    impute_na(data)
    sample = get_sample(data=data, size=0.1)
    estimator = RandomForestClassifier(n_estimators=100, n_jobs=-1)
    targets = ["score"]
    max_rank = 10
    pred_weight = 3

    for target in targets:
        print(f"\nCURRENT CONFIGURATION")
        print("########################################################################")
        print(f"Target = {target}")
        print(f"Max_rank = {max_rank}")
        print(f"Pred_weight = {pred_weight}")
        print("########################################################################")

        extract_train_features(data=sample, target=target, max_rank=max_rank)
        #decreasing_features_select(data=sample, estimator=estimator, target=target)
        feature_selection(data=sample, estimator=estimator, n_features=10)









