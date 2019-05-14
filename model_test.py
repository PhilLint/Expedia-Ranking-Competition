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

    new_data, number_books, number_clicks, id_list = oversample(data, max_rank=5)
    print("length new data", len(new_data))

    new_data = new_data.sort_values(["srch_id"])

    x = new_data[["prop_starrating", "prop_location_score1", "price_usd", "prop_log_historical_price"]]
    y = new_data[["booking_bool", "prop_id", "srch_id"]]

    #X_train = x[:int(len(x)/2)]
    #X_test = x[int(len(x)/2):]

    #y_train = y[:int(len(y)/2)]
    #y_test = y[int(len(x)/2):]

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=False)

    rf = RandomForestRegressor(n_estimators=100)

    rf.fit(X_train, y_train["booking_bool"])
    rf.feature_importances_
    prediction_rf = rf.predict(X_test)

    return y_test, prediction_rf


def prediction_to_submission(prediction, y_test):

    y_test["prediction"] = prediction
    y_test_sorted = y_test.sort_values(["srch_id", "prediction"], ascending=[True, False])
    return y_test_sorted[["srch_id", "prop_id"]]


#def feature_selection(estimator, data):

if __name__ == "__main__":

    data = pd.read_csv("training_set_VU_DM.csv", nrows=100_000)
    y_test, prediction = rf_regressor(data, 100)
    submission = prediction_to_submission(prediction, y_test)
    calculate_score(submission, y_test)


