import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils import shuffle
from joblib import dump, load
from scoring import score_prediction
from feature_engineering import extract_train_features
from feature_engineering import simple_imputation
from feature_engineering import *
import random

def impute_na(train):
    na_ids = train.isna().any().values
    feature_list = train.columns[na_ids.nonzero()[0]]
    for feature in feature_list:
        missing_ids = train.loc[train[feature].isna(), :].index.values
        imp =  simple_imputation(train, feature, imp_type="median")
        train.loc[missing_ids, feature] = imp

    return train


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

    training, valid = split_train_test(data)

    # delete booking bool ,... from training data but keep for test data for evaluation
    not_used_target_info = ["click_bool", "booking_bool", "position", "random_bool"]
    training = training.loc[:, ~training.columns.isin(not_used_target_info)]

    # train_down, number_books, number_clicks, id_list = oversample(train, max_rank=5)
    print("length train data", len(training))
    print("length valid data", len(valid))

    X_train = training.loc[:, training.columns != "target"]
    y_train = training.loc[:, training.columns == "target"]

    X_valid = valid.loc[:, valid.columns != "target"]
    y_valid = valid

    random.seed(42)
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


def random_split_train_test(data, split=0.75):

    srch_ids = shuffle(data["srch_id"].value_counts().index.tolist())
    bound = int(len(srch_ids)*split)

    train_ids = srch_ids[:bound]
    test_ids = srch_ids[bound:]

    train = data.loc[data["srch_id"].isin(train_ids)]
    test = data.loc[data["srch_id"].isin(test_ids)]
    return train,test


def cross_validate(estimator, data, type_est, pred_weight, target, k_folds=3, split=4, to_print=False, save_model=False):
    """
    cross-validate over k-folds
    :param estimator: sklearn estimator instance
    :param data: whole dataset
    :param type_est: type of estimator used (classifier, regression)
    :param pred_weight: weighting of booking_bool and click_bool predictions in classification
    :param k_folds: number of folds to perform
    :param split: split param for split_train_test funct
    :param to_print: if True scores are printed and not returned
    :param save_model: if True save model
    :return: if to_print=False function returns list with scores
    """

    scores = []
    for i in range(k_folds):
        print(f"Fold {i+1} running...")

        train, test = split_train_test(data, split)
        X_train = train.drop(columns=["target", "booking_bool", "click_bool", "position", "random_bool"])
        y_train = train["target"]
        X_test = test.drop(columns=["target", "booking_bool", "click_bool", "position", "random_bool"])
        y_test = test.loc[:, ["srch_id", "prop_id", "booking_bool", "click_bool"]]

        # fit model
        print(f"Fitting model...")
        estimator.fit(X_train, y_train)

        # predict
        print(f"Generating predictions...")
        if type_est == "classifier":
            if target == "book":
                prediction = estimator.predict_proba(X_test)[:, 1]
            elif target == "score":
                # calculate weighted sum (probability of class 5 weighs 5x)
                predict_array = estimator.predict_proba(X_test)
                # weigh click_book instances double
                predict_array[:, 2] = predict_array[:, 2]*pred_weight
                prediction = predict_array[:, [1, 2]].sum(axis=1)
            else:
                print("ERROR. no using classification with score_rank!")
                return

        elif type_est == "regression":
            prediction = estimator.predict(X_test)
        else:
            print("Invalid type_est specified!")
            return


        # score
        score = score_prediction(prediction, y_test, to_print=False)
        scores.append(score)
        print(f"Fold {i+1} finished!")

    if save_model:
        dump(estimator, "model2.joblib")
    if to_print:
            print(f"Prediction scores for {k_folds} folds are:\n {scores}")
    else:
        return scores


def get_sample(data, size):
    srch_ids = shuffle(data["srch_id"].value_counts().index.tolist())
    bound = int(len(srch_ids) * size)
    sample_ids = srch_ids[:bound]
    sample = data.loc[data["srch_id"].isin(sample_ids)]
    return sample


def load_clf_to_prediction(filename, X_test):

    clf = load(filename)
    predict_array = clf.predict_proba(X_test)
    predict_array[:, 2] = predict_array[:, 2] * pred_weight
    prediction = predict_array[:, [1, 2]].sum(axis=1)
    return prediction


def predict_test_set():
    test_data = pd.read_csv("C:/Users/Frede/Dropbox/Master/DM/Assignments/2/DM2/final_test_data.csv")
    test_data = impute_na(test_data)
    prediction = load_clf_to_prediction("model2.joblib", test_data)
    submission = prediction_to_submission(prediction, test_data)
    submission.to_csv("sub1.csv", index=False)


def cross_validate(estimator, data, type_est, pred_weight, target, max_rank, k_folds=3, split=4, to_print=False, save_model=False):
    """
    cross-validate over k-folds
    :param estimator: sklearn estimator instance
    :param data: whole dataset
    :param type_est: type of estimator used (classifier, regression)
    :param pred_weight: weighting of booking_bool and click_bool predictions in classification
    :param k_folds: number of folds to perform
    :param split: split param for split_train_test funct
    :param to_print: if True scores are printed and not returned
    :param save_model: if True save model
    :return: if to_print=False function returns list with scores
    """

    scores = []
    for i in range(k_folds):
        print(f"Fold {i+1} running...")

        train, test = split_train_test(data, split)
        # WITHOUT PRIOR DOWNSAMPLING NOW
        train, _, _, _ = oversample(data=train, max_rank=max_rank)
        X_train = train.drop(columns=["target", "booking_bool", "click_bool", "position", "random_bool"])
        y_train = train["target"]
        X_test = test.drop(columns=["target", "booking_bool", "click_bool", "position", "random_bool"])
        y_test = test.loc[:, ["srch_id", "prop_id", "booking_bool", "click_bool"]]

        # fit model
        print(f"Fitting model...")
        estimator.fit(X_train, y_train)

        # predict
        print(f"Generating predictions...")
        if type_est == "classifier":
            if target == "book":
                prediction = estimator.predict_proba(X_test)[:, 1]
            elif target == "score":
                # calculate weighted sum (probability of class 5 weighs 5x)
                predict_array = estimator.predict_proba(X_test)
                # weigh click_book instances double
                predict_array[:, 2] = predict_array[:, 2]*pred_weight
                prediction = predict_array[:, [1, 2]].sum(axis=1)
            else:
                print("ERROR. no using classification with score_rank!")
                return

        elif type_est == "regression":
            prediction = estimator.predict(X_test)
        else:
            print("Invalid type_est specified!")
            return


        # score
        score = score_prediction(prediction, y_test, to_print=False)
        scores.append(score)
        print(f"Fold {i+1} finished!")

    if save_model:
        dump(estimator, "model2.joblib")
    if to_print:
            print(f"Prediction scores for {k_folds} folds are:\n {scores}")
    else:
        return scores


if __name__ == "__main__":
    # constants
    pd.options.mode.chained_assignment = None
    targets = ["score"]
    n_estimators = [100]
    max_ranks = [5, 10, 15]
    type_est = "classifier"
    pred_weight = 3
    k_folds = 3

    data = pd.read_csv("C:/Users/Frede/Dropbox/Master/DM/Assignments/2/DM2/training_norm_data.csv")
    data = impute_na(data)
    data = get_sample(data=data, size=0.1)

    for target in targets:
        for n_estimator in n_estimators:
            for max_rank in max_ranks:

                print(f"\nCURRENT CONFIGURATION")
                print("########################################################################")
                print(f"Target = {target}")
                print(f"N_trees = {n_estimator}")
                print(f"Max_rank = {max_rank}")
                print(f"Type estimator: {type_est}")
                if type_est == "classifier":
                    print(f"Prediction weight: {pred_weight}")
                print("########################################################################")

                extract_train_features(data=data, target=target, max_rank=max_rank)
                if type_est == "classifier":
                    estimator = RandomForestClassifier(n_estimators=n_estimator, n_jobs=-1)
                elif type_est == "regression":
                    estimator = RandomForestRegressor(n_estimators=n_estimator, n_jobs=-1)
                else:
                    print("Invalid estimator specified!")

                cross_validate(estimator=estimator,
                               data=data,
                               type_est=type_est,
                               target=target,
                               max_rank=max_rank,
                               pred_weight=pred_weight,
                               k_folds=k_folds,
                               to_print=True,
                               save_model=True)



