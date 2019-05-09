import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import datetime as dt
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from data_import import oversample
from sklearn.ensemble import RandomForestClassifier
from feature_selection import forest_feat_select

if __name__ == "__main__":

    data =  pd.read_csv("C:/Users/Frede/Dropbox/Master/DM/Assignments/2/DM2/training_set_VU_DM.csv", nrows=1_000_000)

    data = data.loc[:, data.columns != "date_time"]
    print("length original data", len(data))

    new_data, number_books, number_clicks, id_list = oversample(data, max_rank=5)
    print("length new data", len(new_data))

    x = new_data[["prop_starrating", "prop_location_score1", "price_usd", "prop_log_historical_price"]]

    y = new_data["booking_bool"]


    forest_feat_select(x,y, 100)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3)


    regressor = LogisticRegression()
    rf = RandomForestClassifier(n_estimators=100)
    selector = RFE(regressor, n_features_to_select=3)
    selector = selector.fit(X_train, y_train)
    selector.ranking_


    regressor.fit(X_train, y_train)

    rf.fit(X_train, y_train)
    rf.feature_importances_
    prediction = regressor.predict(X_test)
    prediction_rf = rf.predict(X_test)
    print("accuracy score", accuracy_score(y_test, prediction))
    print("acc_rf_score", accuracy_score(y_test, prediction_rf))