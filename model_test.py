import numpy as np
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from data_import import oversample
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from feature_selection import forest_feat_select
from NDCG_k import calculate_score
from sklearn.utils import shuffle


def rf_regressor(data, n_estimators):
    """
    use random forest regression to predict propability of being booked for each row
    :param X_train: x_train
    :param y_test: y_train
    :param n_estimators: passed to RandomForestRegressor
    :return: vector with booking probability for each row
    """

    data = data.loc[:, data.columns != "date_time"]
    print("length original data", len(data))

    train, test = split_train_test(data)


    train_down, number_books, number_clicks, id_list = oversample(train, max_rank=5)
    print("length new data", len(train_down))

    X_train = train_down[["prop_starrating", "prop_location_score1", "price_usd", "prop_log_historical_price"]]
    y_train = train_down[["booking_bool", "prop_id", "srch_id", "click_bool"]]

    X_test = test[["prop_starrating", "prop_location_score1", "price_usd", "prop_log_historical_price"]]
    y_test = test[["booking_bool", "prop_id", "srch_id", "click_bool"]]

    rf = RandomForestRegressor(n_estimators=100)

    rf.fit(X_train, y_train["booking_bool"])
    prediction_rf = rf.predict(X_test)

    return y_test, prediction_rf


def prediction_to_submission(prediction, y_test):

    y_test["prediction"] = prediction
    y_test_sorted = y_test.sort_values(["srch_id", "prediction"], ascending=[True, False])
    return y_test_sorted[["srch_id", "prop_id"]]


def split_train_test(data):
    """
    split data into train (0.66) and test (0.33) data
    :param data: pandas df
    :return: two pandas dfs: train and test
    """

    test = shuffle(data.loc[data["srch_id"] % 3 == 0])
    train = shuffle(data.loc[data["srch_id"] % 3 != 0])

    return train, test


def test_submission(data, test_data):
    data = data.loc[:, data.columns != "date_time"]
    print("length original data", len(data))

    new_data, number_books, number_clicks, id_list = oversample(data, max_rank=5)
    print("length new data", len(new_data))

    new_data = new_data.sort_values(["srch_id"])

    X_train = new_data[["prop_starrating", "prop_location_score1", "price_usd", "prop_log_historical_price"]]
    y_train = new_data[["booking_bool", "prop_id", "srch_id", "click_bool"]]

    X_test = test_data[["prop_starrating", "prop_location_score1", "price_usd", "prop_log_historical_price"]]
    y_test = test_data[["prop_id", "srch_id"]]

    rf = RandomForestRegressor(n_estimators=100)

    rf.fit(X_train, y_train["booking_bool"])
    prediction_rf = rf.predict(X_test)

    return y_test, prediction_rf


if __name__ == "__main__":

    data = pd.read_csv("C:/Users/Frede/Dropbox/Master/DM/Assignments/2/DM2/training_set_VU_DM.csv")
    test_data = pd.read_csv("C:/Users/Frede/Dropbox/Master/DM/Assignments/2/DM2/test_set_VU_DM.csv")
    y_test, prediction = test_submission(data, test_data)
    #y_test, prediction = rf_regressor(data, 100)
    submission = prediction_to_submission(prediction, y_test)
    submission.to_csv("test_sub.csv", index=False)

    print(calculate_score(submission, y_test))


